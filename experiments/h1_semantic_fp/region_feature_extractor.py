"""
Region Feature Extractor for YOLO-E
====================================
YOLO-E 모델에서 detection별 region feature 추출

Region feature: cv3 출력 (CLIP space에 정렬된 visual embedding)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.modules.head import YOLOEDetect


@dataclass
class DetectionWithFeature:
    """Detection 결과 + Region Feature"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    region_feature: np.ndarray  # [embed_dim]
    class_scores: np.ndarray  # [num_classes]


class RegionFeatureExtractor:
    """
    YOLO-E에서 region feature 추출
    
    YOLOEDetect.cv3의 출력이 region feature (CLIP space)
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: YOLOE 모델
            device: 디바이스
        """
        self.model = model
        self.device = device
        
        # Detect head 참조
        self.detect_head = self._get_detect_head()
    
    def _get_detect_head(self) -> YOLOEDetect:
        """Detect head 가져오기"""
        # model.model[-1]이 detect head
        head = self.model.model.model[-1]
        if isinstance(head, YOLOEDetect):
            return head
        raise ValueError("Model does not have YOLOEDetect head")
    
    @torch.no_grad()
    def extract_features(self,
                        img: torch.Tensor,
                        conf_threshold: float = 0.001,
                        iou_threshold: float = 0.7,
                        max_det: int = 300) -> List[DetectionWithFeature]:
        """
        이미지에서 detection + region feature 추출
        
        Args:
            img: [1, 3, H, W] 입력 이미지
            conf_threshold: confidence 임계값
            iou_threshold: NMS IoU 임계값
            max_det: 최대 detection 수
        
        Returns:
            DetectionWithFeature 리스트
        """
        # 1. Forward pass (feature map 수집을 위해 hook 사용)
        features_by_scale = []
        
        def hook_fn(module, input, output):
            features_by_scale.append(output)
        
        # cv3 (region feature 생성 레이어)에 hook 등록
        hooks = []
        for cv3_layer in self.detect_head.cv3:
            hook = cv3_layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        try:
            # Forward
            results = self.model.predict(img, verbose=False, conf=conf_threshold)
            
        finally:
            # Hook 제거
            for hook in hooks:
                hook.remove()
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        # 2. Detection 결과 처리
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = result.boxes.conf.cpu().numpy()  # [N]
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]
        
        # 3. Region feature 추출 (feature map에서 RoI pooling/sampling)
        detections = []
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            # 각 detection에 대해 region feature 추출
            region_feat = self._extract_region_feature(
                features_by_scale, box, img.shape[-2:]
            )
            
            # 클래스 점수 (cv4 출력에서)
            class_scores = self._get_class_scores(region_feat)
            
            detections.append(DetectionWithFeature(
                bbox=box,
                confidence=float(conf),
                class_id=int(cls_id),
                region_feature=region_feat,
                class_scores=class_scores,
            ))
        
        return detections
    
    def _extract_region_feature(self,
                                features_by_scale: List[torch.Tensor],
                                bbox: np.ndarray,
                                img_size: Tuple[int, int]) -> np.ndarray:
        """
        Feature map에서 bbox 영역의 feature 추출
        
        간단한 구현: 가장 적합한 scale에서 center sampling
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_area = (x2 - x1) * (y2 - y1)
        
        img_h, img_w = img_size
        
        # 가장 적합한 scale 선택 (box 크기 기준)
        best_scale_idx = 0
        best_area_ratio = float('inf')
        
        strides = [8, 16, 32]  # P3, P4, P5
        for i, stride in enumerate(strides):
            feat_area = stride * stride
            ratio = abs(np.log(box_area / feat_area + 1e-6))
            if ratio < best_area_ratio:
                best_area_ratio = ratio
                best_scale_idx = i
        
        # 해당 scale의 feature map에서 center 위치 feature 추출
        feat = features_by_scale[best_scale_idx]  # [B, C, H, W]
        
        stride = strides[best_scale_idx]
        feat_h, feat_w = feat.shape[-2:]
        
        # Center 좌표를 feature map 좌표로 변환
        fx = int(cx / img_w * feat_w)
        fy = int(cy / img_h * feat_h)
        
        fx = max(0, min(fx, feat_w - 1))
        fy = max(0, min(fy, feat_h - 1))
        
        # Feature 추출 및 정규화
        region_feat = feat[0, :, fy, fx].cpu().numpy()
        region_feat = region_feat / (np.linalg.norm(region_feat) + 1e-10)
        
        return region_feat
    
    def _get_class_scores(self, region_feat: np.ndarray) -> np.ndarray:
        """
        Region feature로부터 클래스 점수 계산
        
        pe (prompt embedding)와의 dot product
        """
        if hasattr(self.model.model, 'pe'):
            pe = self.model.model.pe  # [1, num_classes, embed_dim]
            pe_np = pe.squeeze(0).cpu().numpy()
            
            # Dot product
            scores = np.dot(pe_np, region_feat)
            return scores
        else:
            return np.zeros(self.detect_head.nc)


def extract_detections_with_features(model,
                                     dataloader,
                                     class_names: Dict[int, str],
                                     confounder_indices: set,
                                     max_images: int = None,
                                     conf_threshold: float = 0.001,
                                     verbose: bool = False) -> List:
    """
    데이터로더에서 모든 detection + region feature 추출
    
    Args:
        model: YOLOE 모델
        dataloader: 데이터 로더
        class_names: 클래스 이름
        confounder_indices: confounder 클래스 인덱스
        max_images: 최대 이미지 수
        conf_threshold: confidence 임계값
        verbose: 상세 출력
    
    Returns:
        Detection 리스트
    """
    from tqdm import tqdm
    from .detection_logger import Detection, DetectionLogger
    
    extractor = RegionFeatureExtractor(model, device=str(next(model.parameters()).device))
    
    logger = DetectionLogger(
        class_names=class_names,
        confounder_indices=confounder_indices,
        top_k=10,
    )
    
    total = len(dataloader) if max_images is None else min(len(dataloader), max_images)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=total, desc="Extracting features")):
        if max_images and batch_idx >= max_images:
            break
        
        img = batch["img"].to(extractor.device)
        
        # 이미지별 처리
        for b in range(img.shape[0]):
            single_img = img[b:b+1]
            
            # Feature 추출
            dets_with_feat = extractor.extract_features(
                single_img,
                conf_threshold=conf_threshold,
            )
            
            # GT 정보 처리 (있는 경우)
            # TODO: GT와 매칭하여 TP/FP 판정
        
        if verbose and batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: processed")
    
    return logger
