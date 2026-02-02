"""
Semantic Uncertainty (u_sem) Calculation
=========================================
MobileCLIP 기반 JS divergence semantic uncertainty

구조:
- YOLOE: detection만 (bbox, class, conf)
- MobileCLIP: u_sem 계산
  - bbox crop → MobileCLIP image encoder
  - attribute text → MobileCLIP text encoder  
  - Top-M gating + JS divergence

장점:
- YOLOE/MobileCLIP 임베딩 공간 일관성
- Head 해킹 불필요
- 포화 방지 (Top-M gating)
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


# =============================================================================
# DEPICTION/REPLICA 클래스 정의 (H1 가설용)
# =============================================================================

# LVIS에서 "실물 아닌" depiction/replica 카테고리
DEPICTION_REPLICA_CLASSES = {
    # Toys & Figurines
    "toy", "doll", "puppet", "action_figure", "figurine", "stuffed_animal",
    "teddy_bear", "plush_toy", "model_car", "model_airplane",
    # Statues & Sculptures
    "statue", "sculpture", "bust_(sculpture)", "mannequin",
    # 2D Depictions
    "poster", "painting", "photograph", "picture", "drawing", "print",
    "portrait", "mural",
    # Others
    "cartoon", "mascot", "inflatable",
}


def is_depiction_replica_class(class_name: str) -> bool:
    """클래스가 depiction/replica인지 확인"""
    name_lower = class_name.lower().replace(" ", "_")
    for dep_class in DEPICTION_REPLICA_CLASSES:
        if dep_class in name_lower:
            return True
    return False


# =============================================================================
# MobileCLIP Scorer
# =============================================================================

class MobileCLIPScorer:
    """
    MobileCLIP을 사용한 image-text similarity scorer
    
    YOLOE와 동일한 MobileCLIP 공간에서 u_sem 계산
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.embed_dim = 512
        self._load_mobileclip()
    
    def _load_mobileclip(self):
        """MobileCLIP 모델 로드"""
        try:
            import mobileclip
            
            # MobileCLIP-B (blt) - YOLOE default
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                'mobileclip_blt', 
                pretrained='checkpoints/mobileclip_blt.pt'
            )
            self.tokenizer = mobileclip.get_tokenizer('mobileclip_blt')
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embed_dim = 512
            
            print(f"  Loaded MobileCLIP-BLT on {self.device}")
            
        except Exception as e:
            print(f"  WARNING: MobileCLIP load failed ({e}), falling back to CLIP")
            self._load_clip_fallback()
    
    def _load_clip_fallback(self):
        """CLIP fallback"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.tokenizer = clip.tokenize
            self.model.eval()
            self.embed_dim = 512
            print(f"  Loaded CLIP ViT-B/32 (fallback) on {self.device}")
        except ImportError:
            print("  WARNING: Neither MobileCLIP nor CLIP available")
    
    @torch.no_grad()
    def encode_image_crop(self, image: Image.Image, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        이미지 crop의 MobileCLIP embedding
        
        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2]
        
        Returns:
            [embed_dim] embedding
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
        
        # MobileCLIP encoding
        crop_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
        features = self.model.encode_image(crop_tensor)
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트의 MobileCLIP embedding
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            [N, embed_dim] embeddings
        """
        if self.model is None:
            return None
        
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()


# =============================================================================
# JS Divergence 계산
# =============================================================================

def js_divergence(distributions: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Jensen-Shannon Divergence 계산
    
    Args:
        distributions: [K, N] - K개 분포, 각각 N개 클래스에 대한 확률
        weights: [K] - 각 분포의 가중치 (None이면 균등)
    
    Returns:
        JS divergence 값 (0 ~ log(K))
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


# =============================================================================
# Semantic Uncertainty Calculator (MobileCLIP 기반)
# =============================================================================

class SemanticUncertaintyCalculator:
    """
    Semantic Uncertainty 계산기 (MobileCLIP 기반, Top-M Gating)
    
    파이프라인:
    1. bbox crop → MobileCLIP image embedding f
    2. 1203개 클래스 이름 → MobileCLIP text embeddings → f와 similarity
       → softmax → base posterior p_base
       → top-M 클래스 추출
    3. 각 attribute view k에 대해:
       top-M 클래스의 attribute template → similarity → softmax → p^(k)
    4. JS(p^(1), ..., p^(K)) = u_sem
    
    이점:
    - Top-M gating으로 JS 포화 방지
    - MobileCLIP 공간 일관성 (YOLOE와 동일)
    """
    
    # Attribute view 템플릿 (K개 view)
    ATTRIBUTE_VIEWS = {
        "base": "a photo of a {cls}",
        "material_metal": "a {cls} made of metal",
        "material_plastic": "a {cls} made of plastic", 
        "material_wood": "a {cls} made of wood",
        "texture_smooth": "a smooth {cls}",
        "texture_rough": "a rough {cls}",
        "context_indoor": "a {cls} indoors",
        "context_outdoor": "a {cls} outdoors",
    }
    
    def __init__(self,
                 class_names: Dict[int, str],
                 device: str = "cuda",
                 temperature: float = 10.0,  # 높여서 분포 부드럽게
                 top_m: int = 20):
        """
        Args:
            class_names: 클래스 이름 딕셔너리 {idx: name}
            device: 디바이스
            temperature: softmax temperature (높을수록 부드러운 분포)
            top_m: Top-M 클래스 gating
        """
        self.class_names = class_names
        self.device = device
        self.temperature = temperature
        self.top_m = top_m
        
        # MobileCLIP scorer
        self.scorer = MobileCLIPScorer(device)
        
        # 클래스 이름 리스트 (정렬된 순서)
        self.sorted_class_indices = sorted(class_names.keys())
        self.sorted_class_names = [class_names[i] for i in self.sorted_class_indices]
        self.num_classes = len(self.sorted_class_names)
        
        # View 이름 리스트 (base 제외)
        self.view_names = [k for k in self.ATTRIBUTE_VIEWS.keys() if k != "base"]
        self.num_views = len(self.view_names)
        
        # 텍스트 임베딩 캐시
        self.base_embeddings = None      # [num_classes, embed_dim]
        self.view_embeddings = {}        # {view_name: [num_classes, embed_dim]}
        
        self._build_text_embeddings()
    
    def _clean_class_name(self, name: str) -> str:
        """클래스 이름 정리 (LVIS format → 단순 이름)"""
        # "person/human" → "person"
        return name.split("/")[0].strip().replace("_", " ")
    
    def _build_text_embeddings(self):
        """모든 클래스/view의 텍스트 임베딩 생성"""
        print(f"\n  Building MobileCLIP text embeddings for {self.num_classes} classes, {self.num_views} views...")
        
        clean_names = [self._clean_class_name(n) for n in self.sorted_class_names]
        
        # Base embeddings (클래스 이름만)
        base_texts = [self.ATTRIBUTE_VIEWS["base"].format(cls=n) for n in clean_names]
        self.base_embeddings = self.scorer.encode_texts(base_texts)
        print(f"    Base embeddings: {self.base_embeddings.shape}")
        
        # View embeddings (각 attribute view)
        for view_name in self.view_names:
            template = self.ATTRIBUTE_VIEWS[view_name]
            view_texts = [template.format(cls=n) for n in clean_names]
            self.view_embeddings[view_name] = self.scorer.encode_texts(view_texts)
        
        print(f"    View embeddings: {self.num_views} views x {self.num_classes} classes")
        print(f"    Temperature: {self.temperature}, Top-M: {self.top_m}")
    
    def compute_for_detection(self, detection: Detection) -> Tuple[float, Optional[np.ndarray]]:
        """
        단일 detection의 u_sem 계산
        
        Args:
            detection: Detection 객체
        
        Returns:
            (u_sem, image_embedding) - image_embedding은 artifactness 계산용
        """
        if detection.image_path is None:
            return 0.0, None
        
        try:
            # 1. Image embedding
            img = Image.open(detection.image_path).convert('RGB')
            image_emb = self.scorer.encode_image_crop(img, detection.bbox)
            
            if image_emb is None:
                return 0.0, None
            
            # 2. Base posterior → Top-M 클래스
            base_logits = np.dot(self.base_embeddings, image_emb) / self.temperature
            base_posterior = softmax(base_logits)
            
            # Top-M 클래스 인덱스
            top_m_indices = np.argsort(base_posterior)[-self.top_m:]
            
            # 3. 각 view의 posterior (Top-M만)
            posteriors = []
            for view_name in self.view_names:
                view_emb = self.view_embeddings[view_name]
                
                # Top-M 클래스만
                top_m_emb = view_emb[top_m_indices]  # [M, embed_dim]
                
                # Logits & softmax (Top-M 내에서)
                logits = np.dot(top_m_emb, image_emb) / self.temperature
                posterior = softmax(logits)  # [M]
                posteriors.append(posterior)
            
            posteriors = np.stack(posteriors, axis=0)  # [K, M]
            
            # 4. JS divergence
            u_sem = js_divergence(posteriors)
            
            return u_sem, image_emb
            
        except Exception as e:
            return 0.0, None
    
    def compute_for_detections(self, detections: List[Detection], 
                               verbose: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        여러 detection의 u_sem 일괄 계산
        
        Args:
            detections: Detection 리스트
            verbose: 진행상황 출력
        
        Returns:
            (u_sem_array, image_embeddings)
        """
        u_sems = []
        image_embs = []
        
        for i, det in enumerate(detections):
            if verbose and (i + 1) % 500 == 0:
                print(f"    Processing {i+1}/{len(detections)}...")
            
            u_sem, emb = self.compute_for_detection(det)
            u_sems.append(u_sem)
            image_embs.append(emb)
        
        return np.array(u_sems), image_embs


# =============================================================================
# Triad Split 개선: Depiction/Replica FP 분리
# =============================================================================

class EnhancedTriadSplit:
    """
    개선된 Triad Split: Semantic FP를 세분화
    
    - TP: True Positives
    - ClassConfusion_FP: 미세 클래스 혼동 (chair↔armchair)
    - Depiction_FP: depiction/replica 혼동 (H1 가설 대상)
    - Background_FP: 배경 오검출
    """
    
    def __init__(self, class_names: Dict[int, str]):
        self.class_names = class_names
        
        # Depiction/replica 클래스 인덱스 캐시
        self.depiction_class_indices = set()
        for idx, name in class_names.items():
            if is_depiction_replica_class(name):
                self.depiction_class_indices.add(idx)
        
        print(f"  Found {len(self.depiction_class_indices)} depiction/replica classes")
    
    def classify_detection(self, detection: Detection) -> str:
        """
        Detection을 4개 카테고리로 분류
        
        Returns:
            "TP", "ClassConfusion_FP", "Depiction_FP", "Background_FP"
        """
        # TP 판정
        if detection.is_tp:
            return "TP"
        
        # Background FP: GT와 매칭 안 됨
        if detection.matched_gt_idx is None:
            return "Background_FP"
        
        # Semantic FP → 세분화
        # matched_gt_class 사용 (gt_class가 아님)
        gt_class = detection.matched_gt_class
        pred_class = detection.pred_class
        
        # GT나 Pred 중 하나라도 depiction/replica면 Depiction_FP
        if gt_class in self.depiction_class_indices or pred_class in self.depiction_class_indices:
            return "Depiction_FP"
        
        # 그 외는 일반 클래스 혼동
        return "ClassConfusion_FP"
    
    def split_detections(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """
        Detection 리스트를 4개 그룹으로 분할
        """
        groups = {
            "TP": [],
            "ClassConfusion_FP": [],
            "Depiction_FP": [],
            "Background_FP": [],
        }
        
        for det in detections:
            category = self.classify_detection(det)
            groups[category].append(det)
        
        return groups
