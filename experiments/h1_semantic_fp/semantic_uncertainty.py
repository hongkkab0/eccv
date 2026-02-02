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


class YOLOEFeatureExtractor:
    """
    YOLOE에서 detection 시 visual feature를 추출하는 클래스
    cv4 (BNContrastiveHead)의 input을 캡처하여 embed_dim 차원 feature 추출
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: YOLOE 모델
            device: 디바이스
        """
        self.model = model
        self.device = device
        self.cv3_features = []  # 각 scale의 feature 저장
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """cv4의 input (= cv3 output, embed_dim 차원)을 캡처"""
        head = self.model.model.model[-1]  # YOLOEDetect or YOLOESegment
        
        # cv4 (BNContrastiveHead)의 forward 전에 input을 캡처
        # cv4의 input x가 cv3의 output이며, [B, embed_dim, H, W] 형태
        for i, cv4_module in enumerate(head.cv4):
            # forward_pre_hook으로 input 캡처
            hook = cv4_module.register_forward_pre_hook(
                lambda module, inp, idx=i: self._save_cv4_input(idx, inp)
            )
            self.hooks.append(hook)
    
    def _save_cv4_input(self, idx: int, inputs):
        """cv4의 input (cv3 output) 저장"""
        # inputs는 tuple: (x, w) where x is visual feature, w is text embedding
        if len(inputs) > 0:
            x = inputs[0]  # [B, embed_dim, H, W]
            if len(self.cv3_features) <= idx:
                self.cv3_features.append(x.detach())
            else:
                self.cv3_features[idx] = x.detach()
    
    def _save_cv3_output(self, idx: int, output: torch.Tensor):
        """cv3 output 저장"""
        # output: [B, embed_dim, H, W]
        if len(self.cv3_features) <= idx:
            self.cv3_features.append(output.detach())
        else:
            self.cv3_features[idx] = output.detach()
    
    def clear(self):
        """저장된 feature 초기화"""
        self.cv3_features = []
    
    def remove_hooks(self):
        """Hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_features_for_boxes(self, 
                                    boxes: torch.Tensor,
                                    strides: torch.Tensor) -> torch.Tensor:
        """
        Detection boxes에 대한 visual feature 추출
        
        Args:
            boxes: [N, 4] xyxy 좌표
            strides: [N] 각 detection의 stride (어느 scale에서 왔는지)
        
        Returns:
            [N, embed_dim] visual features
        """
        if len(self.cv3_features) == 0:
            return None
        
        features = []
        head = self.model.model.model[-1]
        head_strides = head.stride  # 각 scale의 stride
        
        for i, (box, stride) in enumerate(zip(boxes, strides)):
            # 해당 stride의 scale index 찾기
            scale_idx = (head_strides == stride).nonzero(as_tuple=True)[0]
            if len(scale_idx) == 0:
                scale_idx = 0
            else:
                scale_idx = scale_idx[0].item()
            
            # 해당 scale의 feature map
            feat_map = self.cv3_features[scale_idx]  # [B, embed_dim, H, W]
            
            # box 중심점의 feature map 위치
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            # stride로 나눠서 feature map 좌표로 변환
            fx = int(cx / stride.item())
            fy = int(cy / stride.item())
            
            # 범위 체크
            _, _, H, W = feat_map.shape
            fx = min(max(fx, 0), W - 1)
            fy = min(max(fy, 0), H - 1)
            
            # feature 추출 (batch=0 가정)
            feat = feat_map[0, :, fy, fx]  # [embed_dim]
            features.append(feat)
        
        if len(features) == 0:
            return None
        
        return torch.stack(features)  # [N, embed_dim]


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
    cv4_module,
    scale_idx: int = 0,
) -> torch.Tensor:
    """
    YOLOE cv4 (BNContrastiveHead)를 사용해서 attribute 템플릿 score 계산
    
    Args:
        visual_feature: [embed_dim] - cv3 output at detection location
        attribute_embeddings: [K, embed_dim] - K개 attribute 템플릿
        cv4_module: YOLOEDetect의 cv4[scale_idx]
        scale_idx: 사용할 scale index
    
    Returns:
        [K] - 각 attribute에 대한 score
    """
    # cv4 (BNContrastiveHead) forward 직접 구현
    # BNContrastiveHead: norm -> logit_scale -> einsum -> bias
    
    x = visual_feature.view(1, -1, 1, 1)  # [1, embed_dim, 1, 1]
    w = attribute_embeddings  # [K, embed_dim]
    
    # BatchNorm 적용
    if hasattr(cv4_module, 'norm'):
        x = cv4_module.norm(x)
    
    # L2 정규화 (text embedding)
    w = F.normalize(w, dim=-1, p=2)
    
    # logit_scale 적용
    if hasattr(cv4_module, 'logit_scale'):
        x = x * cv4_module.logit_scale.exp()
    
    # einsum: [1, embed_dim, 1, 1] x [K, embed_dim] -> [1, K, 1, 1]
    scores = torch.einsum('bchw,nc->bnhw', x, w)
    
    # bias 적용
    if hasattr(cv4_module, 'bias'):
        scores = scores + cv4_module.bias
    
    return scores.squeeze()  # [K]


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
        
        YOLOE cv4를 사용해서 각 attribute 템플릿에 대한 score를 계산하고
        이 score들의 JS divergence로 u_sem 반환
        
        Args:
            detection: Detection 객체
            use_gating: (미사용, 호환성용)
        
        Returns:
            u_sem 값
        """
        if detection.region_feature is None:
            return 0.0
        
        # 예측된 클래스의 attribute embeddings 가져오기
        pred_class = detection.pred_class
        if pred_class not in self.class_view_embeddings:
            return 0.0
        
        attr_embeddings = self.class_view_embeddings[pred_class]  # [K, embed_dim]
        K = attr_embeddings.shape[0]
        
        # Visual feature를 tensor로 변환
        if isinstance(detection.region_feature, np.ndarray):
            visual_feat = torch.from_numpy(detection.region_feature).float().to(self.device)
        else:
            visual_feat = detection.region_feature.to(self.device)
        
        # cv4가 있으면 cv4를 사용해서 정확한 score 계산
        if self.cv4 is not None:
            # cv4[0] 사용 (scale 0)
            scores = compute_attribute_scores_yoloe(
                visual_feat, attr_embeddings, self.cv4[0]
            )
            scores = scores.detach().cpu().numpy()
        else:
            # Fallback: 단순 dot product
            visual_feat_np = visual_feat.cpu().numpy()
            visual_feat_np = visual_feat_np / (np.linalg.norm(visual_feat_np) + 1e-10)
            attr_emb_np = attr_embeddings.cpu().numpy()
            scores = np.dot(attr_emb_np, visual_feat_np)
        
        # 각 attribute view를 별도의 "class"로 취급해서 JS divergence 계산
        # score -> softmax -> posterior
        # 여기서는 단일 클래스에 대한 K개 view의 score 분포를 비교
        # JS divergence of K distributions (각 view에서의 score를 확률로 변환)
        
        # 방법 1: 각 view의 score를 확률로 변환 (softmax)
        # K개 view에서 동일 클래스에 대한 score의 분산을 측정
        posteriors = []
        for k in range(K):
            # 각 view의 score를 2-class 분포로 변환 (해당 attribute vs not)
            # score가 높으면 해당 attribute가 맞음
            s = scores[k]
            p = softmax(np.array([s, 0]) / self.temperature)  # [p(attr), p(not attr)]
            posteriors.append(p)
        
        posteriors = np.stack(posteriors, axis=0)  # [K, 2]
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
