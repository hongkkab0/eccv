"""
Detection Logger for H1 Experiment
===================================
LVIS val에서 detection 결과를 수집하고 TP/FP 분류 수행
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import pickle
from tqdm import tqdm

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou


@dataclass
class Detection:
    """단일 detection 정보"""
    # 기본 정보
    image_id: str
    det_idx: int  # 이미지 내 detection 인덱스
    
    # Bounding box (xyxy format)
    bbox: np.ndarray  # [x1, y1, x2, y2]
    
    # Prediction 정보
    pred_class: int
    pred_class_name: str
    confidence: float
    
    # Region feature (CLIP space)
    region_feature: Optional[np.ndarray] = None  # [embed_dim]
    
    # Top-K class scores (for gating)
    top_k_classes: Optional[np.ndarray] = None  # [K]
    top_k_scores: Optional[np.ndarray] = None  # [K]
    
    # TP/FP 판정 결과
    is_tp: bool = False
    matched_gt_idx: Optional[int] = None
    matched_gt_class: Optional[int] = None
    matched_gt_iou: float = 0.0
    
    # Triad split 결과
    triad_label: str = "unknown"  # "TP", "Semantic_FP", "Background_FP"
    
    # Semantic FP 상세 정보 (Semantic FP인 경우)
    overlapping_gt_idx: Optional[int] = None
    overlapping_gt_class: Optional[int] = None
    overlapping_gt_class_name: Optional[str] = None
    overlapping_gt_iou: float = 0.0


@dataclass
class ImageGT:
    """이미지의 GT 정보"""
    image_id: str
    bboxes: np.ndarray  # [N, 4] xyxy format
    classes: np.ndarray  # [N]
    class_names: List[str]


class DetectionLogger:
    """Detection 결과 수집 및 분류"""
    
    def __init__(self, 
                 class_names: Dict[int, str],
                 confounder_indices: set,
                 tp_iou_threshold: float = 0.5,
                 semantic_fp_iou_threshold: float = 0.3,
                 top_k: int = 10):
        """
        Args:
            class_names: {class_idx: class_name}
            confounder_indices: confounder 클래스 인덱스 집합
            tp_iou_threshold: TP 판정 IoU 기준
            semantic_fp_iou_threshold: Semantic FP 판정 IoU 기준
            top_k: Top-K 클래스 저장 수
        """
        self.class_names = class_names
        self.confounder_indices = confounder_indices
        self.tp_iou_threshold = tp_iou_threshold
        self.semantic_fp_iou_threshold = semantic_fp_iou_threshold
        self.top_k = top_k
        
        # 수집된 detections
        self.detections: List[Detection] = []
        
        # 통계
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "tp_count": 0,
            "semantic_fp_count": 0,
            "background_fp_count": 0,
        }
    
    def process_batch(self,
                      preds: torch.Tensor,
                      gt_bboxes: torch.Tensor,
                      gt_classes: torch.Tensor,
                      image_ids: List[str],
                      region_features: Optional[torch.Tensor] = None,
                      class_scores: Optional[torch.Tensor] = None):
        """
        배치 처리: detection 결과와 GT를 매칭하여 TP/FP 분류
        
        Args:
            preds: [B, N, 6] (x1, y1, x2, y2, conf, class)
            gt_bboxes: List of [M, 4] per image
            gt_classes: List of [M] per image  
            image_ids: 이미지 ID 리스트
            region_features: [B, N, embed_dim] (optional)
            class_scores: [B, N, num_classes] (optional)
        """
        batch_size = len(image_ids)
        
        for b in range(batch_size):
            # 현재 이미지의 predictions
            pred = preds[b] if preds.dim() == 3 else preds
            if pred.numel() == 0:
                continue
            
            pred_boxes = pred[:, :4]
            pred_confs = pred[:, 4]
            pred_classes = pred[:, 5].int()
            
            # GT
            gt_box = gt_bboxes[b] if isinstance(gt_bboxes, list) else gt_bboxes
            gt_cls = gt_classes[b] if isinstance(gt_classes, list) else gt_classes
            
            # Region features (있는 경우)
            reg_feats = region_features[b] if region_features is not None else None
            cls_scores = class_scores[b] if class_scores is not None else None
            
            # IoU 계산
            if gt_box.numel() > 0 and pred_boxes.numel() > 0:
                ious = box_iou(pred_boxes, gt_box)  # [N_pred, N_gt]
            else:
                ious = torch.zeros(len(pred_boxes), max(1, len(gt_box)))
            
            # 각 detection 처리
            for det_idx in range(len(pred_boxes)):
                detection = self._process_single_detection(
                    image_id=image_ids[b],
                    det_idx=det_idx,
                    pred_box=pred_boxes[det_idx].cpu().numpy(),
                    pred_conf=pred_confs[det_idx].item(),
                    pred_class=pred_classes[det_idx].item(),
                    gt_boxes=gt_box.cpu().numpy() if gt_box.numel() > 0 else np.array([]),
                    gt_classes=gt_cls.cpu().numpy() if gt_cls.numel() > 0 else np.array([]),
                    det_gt_ious=ious[det_idx].cpu().numpy() if ious.numel() > 0 else np.array([]),
                    region_feat=reg_feats[det_idx].cpu().numpy() if reg_feats is not None else None,
                    cls_scores=cls_scores[det_idx].cpu().numpy() if cls_scores is not None else None,
                )
                
                self.detections.append(detection)
                self.stats["total_detections"] += 1
                
                # 통계 업데이트
                if detection.triad_label == "TP":
                    self.stats["tp_count"] += 1
                elif detection.triad_label == "Semantic_FP":
                    self.stats["semantic_fp_count"] += 1
                else:
                    self.stats["background_fp_count"] += 1
    
    def _process_single_detection(self,
                                   image_id: str,
                                   det_idx: int,
                                   pred_box: np.ndarray,
                                   pred_conf: float,
                                   pred_class: int,
                                   gt_boxes: np.ndarray,
                                   gt_classes: np.ndarray,
                                   det_gt_ious: np.ndarray,
                                   region_feat: Optional[np.ndarray],
                                   cls_scores: Optional[np.ndarray]) -> Detection:
        """단일 detection 처리"""
        
        # Top-K 클래스 추출
        top_k_classes, top_k_scores = None, None
        if cls_scores is not None:
            top_k_idx = np.argsort(cls_scores)[-self.top_k:][::-1]
            top_k_classes = top_k_idx
            top_k_scores = cls_scores[top_k_idx]
        
        detection = Detection(
            image_id=image_id,
            det_idx=det_idx,
            bbox=pred_box,
            pred_class=pred_class,
            pred_class_name=self.class_names.get(pred_class, f"class_{pred_class}"),
            confidence=pred_conf,
            region_feature=region_feat,
            top_k_classes=top_k_classes,
            top_k_scores=top_k_scores,
        )
        
        # TP/FP 판정
        if len(gt_boxes) == 0:
            # GT가 없으면 모든 detection은 Background FP
            detection.is_tp = False
            detection.triad_label = "Background_FP"
        else:
            # 가장 높은 IoU를 가진 GT 찾기
            max_iou_idx = np.argmax(det_gt_ious)
            max_iou = det_gt_ious[max_iou_idx]
            matched_gt_class = int(gt_classes[max_iou_idx])
            
            # TP 조건: IoU >= threshold AND 클래스 일치
            if max_iou >= self.tp_iou_threshold and matched_gt_class == pred_class:
                detection.is_tp = True
                detection.matched_gt_idx = int(max_iou_idx)
                detection.matched_gt_class = matched_gt_class
                detection.matched_gt_iou = float(max_iou)
                detection.triad_label = "TP"
            else:
                # FP: Semantic FP vs Background FP 분류
                detection.is_tp = False
                
                # Semantic FP 판정 (확장된 정의):
                # 1) IoU >= semantic_fp_threshold이고 클래스가 다른 경우 (classification error)
                # 2) 또는 GT가 confounder 클래스인 경우
                if max_iou >= self.semantic_fp_iou_threshold:
                    # 클래스가 다르면 Semantic FP (localization OK, classification WRONG)
                    is_class_mismatch = (matched_gt_class != pred_class)
                    is_gt_confounder = matched_gt_class in self.confounder_indices
                    
                    if is_class_mismatch or is_gt_confounder:
                        detection.triad_label = "Semantic_FP"
                        detection.overlapping_gt_idx = int(max_iou_idx)
                        detection.overlapping_gt_class = matched_gt_class
                        detection.overlapping_gt_class_name = self.class_names.get(
                            matched_gt_class, f"class_{matched_gt_class}"
                        )
                        detection.overlapping_gt_iou = float(max_iou)
                    else:
                        detection.triad_label = "Background_FP"
                else:
                    detection.triad_label = "Background_FP"
        
        return detection
    
    def get_triad_split(self) -> Dict[str, List[Detection]]:
        """Triad split 결과 반환"""
        split = {
            "TP": [],
            "Semantic_FP": [],
            "Background_FP": [],
        }
        
        for det in self.detections:
            if det.triad_label in split:
                split[det.triad_label].append(det)
        
        return split
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()
    
    def save(self, output_path: str):
        """결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pickle로 전체 저장 (region features 포함)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'detections': self.detections,
                'stats': self.stats,
                'config': {
                    'tp_iou_threshold': self.tp_iou_threshold,
                    'semantic_fp_iou_threshold': self.semantic_fp_iou_threshold,
                    'top_k': self.top_k,
                }
            }, f)
        
        print(f"Saved {len(self.detections)} detections to {output_path}")
    
    @classmethod
    def load(cls, path: str) -> 'DetectionLogger':
        """저장된 결과 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        logger = cls(
            class_names={},
            confounder_indices=set(),
            **data['config']
        )
        logger.detections = data['detections']
        logger.stats = data['stats']
        
        return logger
    
    def export_summary_json(self, output_path: str):
        """요약 JSON 내보내기 (region features 제외)"""
        summary = {
            'stats': self.stats,
            'triad_counts': {
                'TP': self.stats['tp_count'],
                'Semantic_FP': self.stats['semantic_fp_count'],
                'Background_FP': self.stats['background_fp_count'],
            },
            'detections': []
        }
        
        for det in self.detections:
            summary['detections'].append({
                'image_id': det.image_id,
                'det_idx': det.det_idx,
                'pred_class': det.pred_class,
                'pred_class_name': det.pred_class_name,
                'confidence': det.confidence,
                'triad_label': det.triad_label,
                'is_tp': det.is_tp,
                'overlapping_gt_class_name': det.overlapping_gt_class_name,
            })
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
