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


class CLIPScorer:
    """
    CLIP을 사용한 image-text similarity scorer
    
    YOLOE로 detection 후, bbox crop을 CLIP으로 인코딩하여
    attribute text embedding과 비교
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None
        self._load_clip()
    
    def _load_clip(self):
        """CLIP 모델 로드"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            print(f"  Loaded CLIP ViT-B/32 on {self.device}")
        except ImportError:
            print("  WARNING: clip not installed. Run: pip install git+https://github.com/openai/CLIP.git")
    
    @torch.no_grad()
    def encode_image_crop(self, image: Image.Image, bbox: np.ndarray) -> np.ndarray:
        """
        이미지 crop의 CLIP embedding
        
        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2]
        
        Returns:
            [512] embedding
        """
        if self.model is None:
            return None
        
        # Crop
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = image.crop((x1, y1, x2, y2))
        
        # CLIP encoding
        crop_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
        features = self.model.encode_image(crop_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트의 CLIP embedding
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            [N, 512] embeddings
        """
        if self.model is None:
            return None
        
        import clip
        tokens = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    def compute_similarities(self, image_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
        """
        Image-text similarity 계산
        
        Args:
            image_emb: [512] image embedding
            text_embs: [K, 512] text embeddings
        
        Returns:
            [K] similarities
        """
        # Cosine similarity (이미 정규화됨)
        return np.dot(text_embs, image_emb)


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
    Semantic Uncertainty 계산기 (CLIP 기반)
    
    YOLOE로 detection 후, CLIP으로 semantic uncertainty 계산:
    1. bbox crop → CLIP image embedding
    2. attribute text → CLIP text embedding
    3. similarity 분포의 JS divergence = u_sem
    
    장점:
    - YOLOE head 해킹 불필요
    - 임베딩 공간 일관성 (같은 CLIP 모델)
    - Artifactness도 같은 feature로 계산 가능
    """
    
    # Attribute view 템플릿
    ATTRIBUTE_TEMPLATES = {
        "material": [
            "a {cls} made of metal",
            "a {cls} made of plastic",
            "a {cls} made of wood",
            "a {cls} made of fabric",
            "a {cls} made of glass",
        ],
        "texture": [
            "a smooth {cls}",
            "a rough {cls}",
            "a shiny {cls}",
            "a fuzzy {cls}",
            "a patterned {cls}",
        ],
        "context": [
            "a {cls} indoors",
            "a {cls} outdoors",
            "a {cls} in nature",
            "a {cls} in urban setting",
        ],
    }
    
    def __init__(self,
                 class_names: Dict[int, str],
                 device: str = "cuda",
                 temperature: float = 1.0):
        """
        Args:
            class_names: 클래스 이름 딕셔너리
            device: 디바이스
            temperature: softmax temperature
        """
        self.class_names = class_names
        self.device = device
        self.temperature = temperature
        
        # CLIP scorer 초기화
        self.clip_scorer = CLIPScorer(device)
        
        # 클래스별 attribute text embeddings 미리 생성
        self.class_attr_embeddings = {}  # {class_idx: [K, 512]}
        self.class_attr_texts = {}       # {class_idx: [K] texts}
        self._build_attribute_embeddings()
    
    def _build_attribute_embeddings(self):
        """모든 클래스의 attribute text embedding 생성"""
        print(f"  Building CLIP attribute embeddings for {len(self.class_names)} classes...")
        
        for class_idx, class_name in self.class_names.items():
            # 클래스 이름 정리
            clean_name = class_name.split("/")[0].strip()
            
            # 모든 attribute view의 텍스트 생성
            texts = []
            for attr_type, templates in self.ATTRIBUTE_TEMPLATES.items():
                for template in templates:
                    texts.append(template.format(cls=clean_name))
            
            self.class_attr_texts[class_idx] = texts
            
            # CLIP text embedding
            embeddings = self.clip_scorer.encode_texts(texts)
            if embeddings is not None:
                self.class_attr_embeddings[class_idx] = embeddings
        
        print(f"  Built embeddings for {len(self.class_attr_embeddings)} classes")
    
    def compute_for_detection(self, detection: Detection, use_gating: bool = True) -> float:
        """
        단일 detection의 u_sem 계산 (CLIP 기반)
        
        1. bbox crop → CLIP image embedding
        2. attribute texts → CLIP text embeddings (미리 계산됨)
        3. similarity 분포의 JS divergence = u_sem
        
        Args:
            detection: Detection 객체
            use_gating: (미사용)
        
        Returns:
            u_sem 값
        """
        # 예측된 클래스의 attribute embeddings 가져오기
        pred_class = detection.pred_class
        if pred_class not in self.class_attr_embeddings:
            return 0.0
        
        attr_embeddings = self.class_attr_embeddings[pred_class]  # [K, 512]
        K = attr_embeddings.shape[0]
        
        # 이미지 로드
        if detection.image_path is None:
            return 0.0
        
        try:
            img = Image.open(detection.image_path).convert('RGB')
            
            # CLIP image embedding
            image_emb = self.clip_scorer.encode_image_crop(img, detection.bbox)
            if image_emb is None:
                return 0.0
            
            # Image-attribute similarities
            similarities = self.clip_scorer.compute_similarities(image_emb, attr_embeddings)
            
        except Exception as e:
            return 0.0
        
        # Similarity를 확률 분포로 변환 (softmax)
        # 각 attribute type별로 그룹핑해서 분포 생성
        # 여기서는 전체 K개 attribute에 대한 softmax 분포 사용
        probs = softmax(similarities / self.temperature)
        
        # JS divergence: attribute type별로 분포를 비교
        # 간단히 전체 분포의 entropy를 u_sem으로 사용
        # (높은 entropy = 불확실성 높음 = semantic FP 가능성)
        u_sem = entropy(probs + 1e-10)
        
        return u_sem
    
    def get_image_embedding(self, detection: Detection) -> Optional[np.ndarray]:
        """
        Detection의 CLIP image embedding 반환 (artifactness 계산용)
        """
        if detection.image_path is None:
            return None
        
        try:
            img = Image.open(detection.image_path).convert('RGB')
            return self.clip_scorer.encode_image_crop(img, detection.bbox)
        except:
            return None
    
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
