"""
Semantic Uncertainty (u_sem) Calculation
=========================================
JS divergence 기반 semantic uncertainty 계산

u_sem(f) = JS(p^(1), ..., p^(K))

여기서 p^(k)는 k번째 view의 posterior distribution
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from scipy.special import softmax
from scipy.stats import entropy
from PIL import Image
from pathlib import Path

from .detection_logger import Detection
from .attribute_embeddings import AttributeEmbeddingCache, get_view_embeddings_for_classes


def compute_attribute_confidences(
    model,
    image: torch.Tensor,
    bbox: np.ndarray,
    attribute_embeddings: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """
    YOLOE에 attribute text embedding을 넣어서 confidence 획득
    
    Args:
        model: YOLOE 모델
        image: [C, H, W] 또는 [1, C, H, W] 이미지 텐서
        bbox: [x1, y1, x2, y2] detection bbox
        attribute_embeddings: [K, embed_dim] K개 attribute 템플릿 embedding
        device: 디바이스
    
    Returns:
        [K] - 각 attribute에 대한 confidence
    """
    K = attribute_embeddings.shape[0]
    
    # 이미지 준비
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device).float()
    if image.max() > 1.0:
        image = image / 255.0
    
    # Attribute embedding을 tpe로 설정
    tpe = model.model.model[-1].get_tpe(attribute_embeddings.unsqueeze(0))
    
    # 임시로 class 설정
    original_nc = model.model.model[-1].nc
    model.model.model[-1].nc = K
    
    # Inference
    with torch.no_grad():
        results = model.predict(image, verbose=False, conf=0.001)
    
    # 원래대로 복구
    model.model.model[-1].nc = original_nc
    
    # 결과에서 해당 bbox와 겹치는 detection 찾기
    confidences = np.zeros(K)
    
    if len(results) > 0 and results[0].boxes is not None:
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_confs = results[0].boxes.conf.cpu().numpy()
        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # bbox와 IoU 계산
        from ultralytics.utils.metrics import box_iou
        bbox_tensor = torch.tensor([bbox]).float()
        pred_boxes_tensor = torch.tensor(pred_boxes).float()
        
        if len(pred_boxes) > 0:
            ious = box_iou(bbox_tensor, pred_boxes_tensor).squeeze(0).numpy()
            
            # IoU > 0.5인 detection들의 confidence를 class별로 수집
            for i, (cls, conf, iou) in enumerate(zip(pred_classes, pred_confs, ious)):
                if iou > 0.5 and cls < K:
                    confidences[cls] = max(confidences[cls], conf)
    
    return confidences


def compute_yoloe_crop_embedding(image: Image.Image, 
                                bbox: np.ndarray,
                                model,
                                device: str = "cuda") -> np.ndarray:
    """
    이미지 crop의 CLIP embedding 계산
    
    Args:
        image: PIL Image
        bbox: [x1, y1, x2, y2] 좌표
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        device: 디바이스
    
    Returns:
        [512] embedding vector
    """
    # Crop
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros(512)
    
    crop = image.crop((x1, y1, x2, y2))
    
    # CLIP encoding
    with torch.no_grad():
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
        features = clip_model.encode_image(crop_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().squeeze()


def js_divergence(distributions: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Jensen-Shannon Divergence 계산
    
    Args:
        distributions: [K, N] - K개 분포, 각각 N개 클래스에 대한 확률
        weights: [K] - 각 분포의 가중치 (None이면 균등)
    
    Returns:
        JS divergence 값
    """
    K = distributions.shape[0]
    
    if weights is None:
        weights = np.ones(K) / K
    
    # 평균 분포
    mean_dist = np.sum(weights[:, np.newaxis] * distributions, axis=0)
    
    # JS = H(mean) - sum(w_k * H(p_k))
    h_mean = entropy(mean_dist + 1e-10)
    h_individual = np.sum([w * entropy(p + 1e-10) for w, p in zip(weights, distributions)])
    
    return h_mean - h_individual


def compute_attribute_scores_yoloe(
    visual_feature: torch.Tensor,
    attribute_embeddings: torch.Tensor,
    cv4_module=None,
    scale_idx: int = 0,
) -> torch.Tensor:
    """
    Attribute 템플릿 score 계산 (visual feature와 text embedding의 similarity)
    
    Args:
        visual_feature: [embed_dim] - cv3 output at detection location
        attribute_embeddings: [K, embed_dim] - K개 attribute 템플릿
        cv4_module: (optional) YOLOEDetect의 cv4[scale_idx] - logit_scale 사용
        scale_idx: 사용할 scale index
    
    Returns:
        [K] - 각 attribute에 대한 score
    """
    # L2 정규화
    v = F.normalize(visual_feature.view(-1), dim=0, p=2)  # [embed_dim]
    w = F.normalize(attribute_embeddings, dim=-1, p=2)    # [K, embed_dim]
    
    # Cosine similarity (dot product of normalized vectors)
    scores = torch.mv(w, v)  # [K]
    
    # logit_scale 적용 (cv4에서 가져오기, 없으면 기본값 사용)
    if cv4_module is not None and hasattr(cv4_module, 'logit_scale'):
        logit_scale = cv4_module.logit_scale.exp()
        scores = scores * logit_scale
    else:
        # CLIP/MobileCLIP 기본 logit_scale ≈ 100
        scores = scores * 100.0
    
    return scores  # [K]


def compute_view_posterior(region_feature: np.ndarray,
                          view_embeddings: np.ndarray,
                          temperature: float = 1.0) -> np.ndarray:
    """
    View posterior 계산
    
    p^(k)(c | f) ∝ exp(<f, T_k(c)> / τ)
    
    Args:
        region_feature: [embed_dim] - region feature
        view_embeddings: [num_classes, K, embed_dim] - 클래스별, view별 임베딩
        temperature: softmax temperature
    
    Returns:
        [K, num_classes] - 각 view의 posterior distribution
    """
    num_classes, K, embed_dim = view_embeddings.shape
    
    # 정규화
    f_norm = region_feature / (np.linalg.norm(region_feature) + 1e-10)
    
    posteriors = []
    for k in range(K):
        # [num_classes, embed_dim]
        v_k = view_embeddings[:, k, :]
        # [num_classes]
        logits = np.dot(v_k, f_norm) / temperature
        p_k = softmax(logits)
        posteriors.append(p_k)
    
    return np.stack(posteriors, axis=0)  # [K, num_classes]


def compute_u_sem(region_feature: np.ndarray,
                  view_embeddings: np.ndarray,
                  temperature: float = 1.0) -> float:
    """
    Semantic uncertainty u_sem 계산
    
    u_sem(f) = JS(p^(1), ..., p^(K))
    
    Args:
        region_feature: [embed_dim]
        view_embeddings: [num_classes, K, embed_dim]
        temperature: softmax temperature
    
    Returns:
        u_sem 값
    """
    posteriors = compute_view_posterior(region_feature, view_embeddings, temperature)
    return js_divergence(posteriors)


def compute_u_sem_gated(region_feature: np.ndarray,
                        view_embeddings: np.ndarray,
                        top_m_classes: np.ndarray,
                        temperature: float = 1.0) -> float:
    """
    Top-M 클래스 게이팅된 u_sem 계산
    
    상위 M개 클래스에 대해서만 JS divergence 계산
    
    Args:
        region_feature: [embed_dim]
        view_embeddings: [num_classes, K, embed_dim] - 전체 클래스
        top_m_classes: [M] - top-M 클래스 인덱스
        temperature: softmax temperature
    
    Returns:
        u_sem 값
    """
    # Top-M 클래스의 view embeddings만 추출
    gated_embeddings = view_embeddings[top_m_classes]  # [M, K, embed_dim]
    
    return compute_u_sem(region_feature, gated_embeddings, temperature)


class SemanticUncertaintyCalculator:
    """
    Semantic Uncertainty 계산기
    
    YOLOE의 cv4를 사용해서 attribute 템플릿 score를 계산하고
    이 score들의 JS divergence로 u_sem 계산
    """
    
    def __init__(self,
                 attribute_cache: AttributeEmbeddingCache,
                 class_names: Dict[int, str],
                 model = None,
                 top_m: int = 10,
                 temperature: float = 1.0,
                 device: str = "cuda"):
        """
        Args:
            attribute_cache: Attribute embedding 캐시
            class_names: 클래스 이름 딕셔너리
            model: YOLOE 모델 (cv4 접근용)
            top_m: Top-M 클래스 게이팅
            temperature: softmax temperature
            device: 디바이스
        """
        self.attribute_cache = attribute_cache
        self.class_names = class_names
        self.model = model
        self.top_m = top_m
        self.temperature = temperature
        self.device = device
        
        # cv4 모듈 참조 (YOLOE head)
        self.cv4 = None
        if model is not None:
            head = model.model.model[-1]
            self.cv4 = head.cv4  # ModuleList of BNContrastiveHead
        
        # 클래스별 attribute view embeddings를 tensor로 준비
        # {class_idx: [K, embed_dim]}
        self.class_view_embeddings = {}
        for class_idx in class_names.keys():
            if class_idx in attribute_cache.class_views:
                views = attribute_cache.class_views[class_idx]
                # [K, embed_dim]
                emb = np.stack([views.view_embeddings[k] for k in range(len(views.view_embeddings))])
                self.class_view_embeddings[class_idx] = torch.from_numpy(emb).float().to(device)
        
        # Legacy: 기존 방식 호환용
        self.all_class_indices = sorted(class_names.keys())
        self.view_embeddings = get_view_embeddings_for_classes(
            attribute_cache, self.all_class_indices
        )  # [num_classes, K, embed_dim]
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.all_class_indices)}
    
    def compute_for_detection(self, detection: Detection, use_gating: bool = True) -> float:
        """
        단일 detection의 u_sem 계산
        
        YOLOE에 attribute text embedding을 넣어서 inference하고
        각 attribute에 대한 confidence 차이로 u_sem 계산
        
        Args:
            detection: Detection 객체
            use_gating: (미사용, 호환성용)
        
        Returns:
            u_sem 값
        """
        # 예측된 클래스의 attribute embeddings 가져오기
        pred_class = detection.pred_class
        if pred_class not in self.class_view_embeddings:
            return 0.0
        
        attr_embeddings = self.class_view_embeddings[pred_class]  # [K, embed_dim]
        K = attr_embeddings.shape[0]
        
        # 이미지 로드
        if detection.image_path is None:
            # region_feature가 있으면 그것 사용 (fallback)
            if detection.region_feature is not None:
                return self._compute_from_feature(detection, attr_embeddings)
            return 0.0
        
        # 이미지를 로드해서 attribute confidence 계산
        try:
            from PIL import Image as PILImage
            img = PILImage.open(detection.image_path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            
            confidences = compute_attribute_confidences(
                self.model, img_tensor, detection.bbox, 
                attr_embeddings, self.device
            )
        except Exception as e:
            # 실패하면 region_feature 사용
            if detection.region_feature is not None:
                return self._compute_from_feature(detection, attr_embeddings)
            return 0.0
        
        # Confidence 분포의 JS divergence
        # 각 attribute의 confidence를 확률 분포로 변환
        posteriors = []
        for k in range(K):
            conf = confidences[k]
            # confidence를 2-class 분포로 변환
            p = np.array([conf, 1 - conf + 1e-10])
            p = p / p.sum()
            posteriors.append(p)
        
        posteriors = np.stack(posteriors, axis=0)  # [K, 2]
        return js_divergence(posteriors)
    
    def _compute_from_feature(self, detection: Detection, attr_embeddings: torch.Tensor) -> float:
        """region_feature를 사용한 fallback 계산"""
        K = attr_embeddings.shape[0]
        
        # Visual feature를 tensor로 변환
        if isinstance(detection.region_feature, np.ndarray):
            visual_feat = torch.from_numpy(detection.region_feature).float().to(self.device)
        else:
            visual_feat = detection.region_feature.to(self.device)
        
        # L2 정규화 후 dot product
        scores = compute_attribute_scores_yoloe(visual_feat, attr_embeddings, self.cv4[0] if self.cv4 else None)
        scores = scores.detach().cpu().numpy()
        
        # Score를 확률로 변환
        posteriors = []
        for k in range(K):
            s = scores[k]
            p = softmax(np.array([s, 0]) / self.temperature)
            posteriors.append(p)
        
        posteriors = np.stack(posteriors, axis=0)
        return js_divergence(posteriors)
    
    def compute_for_detection_with_image(self, 
                                          detection: Detection,
                                          image: Image.Image) -> float:
        """
        이미지에서 CLIP crop embedding을 계산하여 u_sem 반환
        
        Args:
            detection: Detection 객체
            image: PIL Image
        
        Returns:
            u_sem 값
        """
        if self.clip_model is None:
            return 0.0
        
        # CLIP crop embedding 계산
        region_feature = compute_clip_crop_embedding(
            image, detection.bbox, 
            self.clip_model, self.clip_preprocess,
            self.device
        )
        
        # 전체 클래스 대상 u_sem 계산 (게이팅 없이)
        return compute_u_sem(region_feature, self.view_embeddings, self.temperature)
    
    def compute_for_detections(self, 
                               detections: List[Detection],
                               use_gating: bool = True) -> np.ndarray:
        """
        여러 detection의 u_sem 일괄 계산
        
        Returns:
            [N] - u_sem 값 배열
        """
        u_sems = []
        for det in detections:
            u_sems.append(self.compute_for_detection(det, use_gating))
        return np.array(u_sems)
    
    def compute_for_detections_with_images(self,
                                            detections: List[Detection],
                                            image_dir: str,
                                            show_progress: bool = True) -> np.ndarray:
        """
        이미지에서 CLIP crop embedding을 계산하여 u_sem 일괄 반환
        
        Args:
            detections: Detection 리스트
            image_dir: 이미지 디렉토리
            show_progress: 진행바 표시
        
        Returns:
            [N] - u_sem 값 배열
        """
        from tqdm import tqdm
        
        if self.clip_model is None:
            print("  WARNING: CLIP model not loaded, returning zeros")
            return np.zeros(len(detections))
        
        u_sems = []
        image_cache = {}  # 이미지 캐싱
        
        iterator = tqdm(detections, desc="Computing u_sem") if show_progress else detections
        
        for det in iterator:
            # 이미지 로드 (캐싱)
            if det.image_path and det.image_path not in image_cache:
                try:
                    img_path = Path(image_dir) / det.image_path if not Path(det.image_path).is_absolute() else det.image_path
                    image_cache[det.image_path] = Image.open(img_path).convert("RGB")
                except Exception as e:
                    image_cache[det.image_path] = None
            
            image = image_cache.get(det.image_path) if det.image_path else None
            
            if image is not None:
                u_sem = self.compute_for_detection_with_image(det, image)
            elif det.region_feature is not None:
                u_sem = self.compute_for_detection(det, use_gating=False)
            else:
                u_sem = 0.0
            
            u_sems.append(u_sem)
        
        return np.array(u_sems)
    
    def compute_for_triad_split(self,
                                triad_split: Dict[str, List[Detection]],
                                use_gating: bool = True) -> Dict[str, np.ndarray]:
        """
        Triad split 각 그룹의 u_sem 계산
        
        Returns:
            {"TP": [...], "Semantic_FP": [...], "Background_FP": [...]}
        """
        return {
            group: self.compute_for_detections(dets, use_gating)
            for group, dets in triad_split.items()
        }
    
    def compute_for_triad_split_with_images(self,
                                             triad_split: Dict[str, List[Detection]],
                                             image_dir: str) -> Dict[str, np.ndarray]:
        """
        이미지에서 CLIP crop embedding을 계산하여 triad split의 u_sem 반환
        
        Returns:
            {"TP": [...], "Semantic_FP": [...], "Background_FP": [...]}
        """
        result = {}
        for group, dets in triad_split.items():
            print(f"  Computing u_sem for {group} ({len(dets)} samples)...")
            result[group] = self.compute_for_detections_with_images(dets, image_dir, show_progress=True)
        return result


def compute_paraphrase_disagreement(region_feature: np.ndarray,
                                    paraphrase_embeddings: np.ndarray,
                                    top_m_classes: np.ndarray,
                                    temperature: float = 1.0) -> float:
    """
    대조군: Paraphrase ensemble의 disagreement 계산
    
    동의어 프롬프트 간의 JS divergence (낮을 것으로 예상)
    
    Args:
        region_feature: [embed_dim]
        paraphrase_embeddings: [num_classes, K_para, embed_dim]
        top_m_classes: [M] - top-M 클래스 인덱스
        temperature: softmax temperature
    
    Returns:
        Disagreement 값 (JS divergence)
    """
    gated_embeddings = paraphrase_embeddings[top_m_classes]
    return compute_u_sem(region_feature, gated_embeddings, temperature)


def analyze_u_sem_statistics(u_sem_by_group: Dict[str, np.ndarray]) -> Dict:
    """
    u_sem 통계 분석
    
    Args:
        u_sem_by_group: 그룹별 u_sem 배열
    
    Returns:
        분석 결과 딕셔너리
    """
    stats = {}
    
    for group, values in u_sem_by_group.items():
        if len(values) == 0:
            continue
        values = np.asarray(values)
        stats[group] = {
            "count": int(len(values)),
            "min": float(np.min(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    
    # Cohen's d 계산 (TP vs Semantic_FP)
    if "TP" in u_sem_by_group and "Semantic_FP" in u_sem_by_group:
        tp_vals = u_sem_by_group["TP"]
        sem_fp_vals = u_sem_by_group["Semantic_FP"]
        
        if len(tp_vals) > 0 and len(sem_fp_vals) > 0:
            pooled_std = np.sqrt(
                ((len(tp_vals) - 1) * np.var(tp_vals) + 
                 (len(sem_fp_vals) - 1) * np.var(sem_fp_vals)) /
                (len(tp_vals) + len(sem_fp_vals) - 2)
            )
            
            if pooled_std > 0:
                cohens_d = (np.mean(sem_fp_vals) - np.mean(tp_vals)) / pooled_std
                stats["cohens_d_tp_vs_semfp"] = float(cohens_d)
    
    return stats
