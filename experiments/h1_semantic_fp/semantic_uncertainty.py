"""
Semantic Uncertainty (u_sem) Calculation
=========================================
YOLOE region feature + MobileCLIP attribute embeddings → JS divergence

핵심:
1. YOLOE head에서 cv3 출력 (512-dim region feature) 캡처
2. detection bbox 중심에 해당하는 anchor 위치의 feature 추출
3. MobileCLIP attribute embedding table과 matmul → view별 posterior → JS

장점:
- YOLOE 파이프라인 유지
- MobileCLIP 임베딩 공간 일관성
- CLIP fallback 없음
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
# YOLOE Region Feature Extractor
# =============================================================================

class YOLOERegionFeatureExtractor:
    """
    YOLOE에서 region feature 추출
    
    head._region_features: [B, embed, num_anchors]
    anchors, strides로 bbox 중심에 가장 가까운 anchor 찾아서 feature 추출
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.head = model.model.model[-1]  # YOLOEDetect
        
    def get_feature_for_bbox(self, bbox: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        bbox 중심에 해당하는 anchor 위치의 region feature 추출
        
        Args:
            bbox: [x1, y1, x2, y2] xyxy 좌표
            img_shape: (H, W) 이미지 크기
            
        Returns:
            [embed] feature vector 또는 None
        """
        if not hasattr(self.head, '_region_features') or self.head._region_features is None:
            return None
        
        region_features = self.head._region_features[0]  # [embed, num_anchors]
        
        if not hasattr(self.head, 'anchors') or self.head.anchors is None:
            return None
        
        anchors = self.head.anchors  # [2, num_anchors]
        strides = self.head.strides  # [num_anchors]
        
        # bbox 중심점
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # anchor 좌표를 pixel 좌표로 변환
        anchor_px = anchors.T * strides.unsqueeze(1)  # [num_anchors, 2]
        
        # 가장 가까운 anchor 찾기
        dist = ((anchor_px[:, 0] - cx) ** 2 + (anchor_px[:, 1] - cy) ** 2)
        nearest_idx = dist.argmin().item()
        
        # 해당 anchor의 feature
        feature = region_features[:, nearest_idx].cpu().numpy()
        
        return feature


# =============================================================================
# MobileCLIP Attribute Embedding Builder
# =============================================================================

class MobileCLIPAttributeEmbeddings:
    """
    MobileCLIP text encoder로 attribute embeddings 생성
    
    YOLOE가 사용하는 것과 동일한 MobileCLIP 임베딩 공간
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
    
    # Depiction/Real 프롬프트 (artifactness용)
    DEPICTION_PROMPTS = [
        "a toy", "a toy version", "a statue", "a sculpture",
        "a poster", "a painting", "a drawing", "a figurine",
        "a doll", "a puppet", "a stuffed animal", "a plush toy",
        "a mannequin", "a model replica", "a printed image",
    ]
    
    REAL_PROMPTS = [
        "a real photo of the object", "a real object",
        "a living thing", "an actual object",
    ]
    
    def __init__(self, class_names: Dict[int, str], device: str = "cuda"):
        self.class_names = class_names
        self.device = device
        
        # 클래스 이름 리스트
        self.sorted_class_indices = sorted(class_names.keys())
        self.sorted_class_names = [class_names[i] for i in self.sorted_class_indices]
        self.num_classes = len(self.sorted_class_names)
        
        # View 이름 리스트 (base 제외)
        self.view_names = [k for k in self.ATTRIBUTE_VIEWS.keys() if k != "base"]
        self.num_views = len(self.view_names)
        
        # MobileCLIP text encoder
        self.text_model = None
        self.tokenizer = None
        
        # Embeddings
        self.base_embeddings = None      # [num_classes, embed]
        self.view_embeddings = {}        # {view_name: [num_classes, embed]}
        self.depiction_embeddings = None # [len(DEPICTION_PROMPTS), embed]
        self.real_embeddings = None      # [len(REAL_PROMPTS), embed]
        
        self._load_mobileclip_text()
        self._build_embeddings()
    
    def _load_mobileclip_text(self):
        """MobileCLIP text encoder 로드 (YOLOE와 동일)"""
        try:
            from ultralytics.nn.text_model import build_text_model
            
            self.text_model = build_text_model("mobileclip:blt", device=self.device)
            self.text_model.eval()
            self.tokenizer = self.text_model.tokenize
            print(f"  Loaded MobileCLIP text encoder (via YOLOE's build_text_model)")
            
        except Exception as e:
            print(f"  ERROR: Failed to load MobileCLIP text encoder: {e}")
    
    def _clean_class_name(self, name: str) -> str:
        """클래스 이름 정리"""
        return name.split("/")[0].strip().replace("_", " ")
    
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트 인코딩"""
        if self.text_model is None:
            return None
        
        tokens = self.tokenizer(texts)
        features = self.text_model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()
    
    def _build_embeddings(self):
        """모든 embeddings 생성"""
        if self.text_model is None:
            print("  ERROR: Text model not loaded")
            return
        
        print(f"\n  Building MobileCLIP embeddings for {self.num_classes} classes, {self.num_views} views...")
        
        clean_names = [self._clean_class_name(n) for n in self.sorted_class_names]
        
        # Base embeddings
        base_texts = [self.ATTRIBUTE_VIEWS["base"].format(cls=n) for n in clean_names]
        self.base_embeddings = self._encode_texts(base_texts)
        print(f"    Base embeddings: {self.base_embeddings.shape}")
        
        # View embeddings
        for view_name in self.view_names:
            template = self.ATTRIBUTE_VIEWS[view_name]
            view_texts = [template.format(cls=n) for n in clean_names]
            self.view_embeddings[view_name] = self._encode_texts(view_texts)
        
        print(f"    View embeddings: {self.num_views} views x {self.num_classes} classes")
        
        # Depiction/Real embeddings
        self.depiction_embeddings = self._encode_texts(self.DEPICTION_PROMPTS)
        self.real_embeddings = self._encode_texts(self.REAL_PROMPTS)
        print(f"    Depiction: {self.depiction_embeddings.shape}, Real: {self.real_embeddings.shape}")


# =============================================================================
# Semantic Uncertainty Calculator (YOLOE region feature 기반)
# =============================================================================

class SemanticUncertaintyCalculator:
    """
    Semantic Uncertainty 계산기 (YOLOE region feature + MobileCLIP embeddings)
    
    파이프라인:
    1. YOLOE head에서 region feature 캡처 (512-dim)
    2. bbox 중심에 해당하는 anchor 위치의 feature 추출
    3. MobileCLIP attribute embeddings과 matmul → view별 posterior
    4. JS(p^(1), ..., p^(K)) = u_sem
    """
    
    def __init__(self,
                 model,
                 class_names: Dict[int, str],
                 device: str = "cuda",
                 temperature: float = 10.0,
                 top_m: int = 20):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.temperature = temperature
        self.top_m = top_m
        
        # Region feature extractor
        self.feature_extractor = YOLOERegionFeatureExtractor(model, device)
        
        # MobileCLIP attribute embeddings
        self.attr_emb = MobileCLIPAttributeEmbeddings(class_names, device)
        
        print(f"    Temperature: {self.temperature}, Top-M: {self.top_m}")
    
    def compute_for_detection(self, detection: Detection, 
                               img_shape: Tuple[int, int],
                               debug: bool = False) -> Tuple[float, Optional[np.ndarray], Dict]:
        """
        단일 detection의 u_sem 계산
        
        Args:
            detection: Detection 객체
            img_shape: (H, W) 이미지 크기
            debug: True면 디버그 정보 반환
        
        Returns:
            (u_sem, region_feature, debug_info)
        """
        debug_info = {"status": "ok", "drop_reason": None}
        
        # 1. Region feature 추출
        region_feat = self.feature_extractor.get_feature_for_bbox(detection.bbox, img_shape)
        
        if region_feat is None:
            debug_info = {"status": "dropped", "drop_reason": "no_region_feature"}
            return 0.0, None, debug_info
        
        if self.attr_emb.base_embeddings is None:
            debug_info = {"status": "dropped", "drop_reason": "no_embeddings"}
            return 0.0, None, debug_info
        
        # 2. Base posterior → Top-M 클래스
        # region_feat: [embed], base_embeddings: [num_classes, embed]
        base_logits = np.dot(self.attr_emb.base_embeddings, region_feat) / self.temperature
        base_posterior = softmax(base_logits)
        
        # Top-M 클래스 인덱스
        top_m_indices = np.argsort(base_posterior)[-self.top_m:]
        top_m_probs = base_posterior[top_m_indices]
        
        if debug:
            top_m_names = [self.attr_emb.sorted_class_names[i] for i in top_m_indices[-5:]]
            debug_info["top5_base"] = list(zip(top_m_names, top_m_probs[-5:].tolist()))
        
        # 3. 각 view의 posterior (Top-M만)
        posteriors = []
        view_top5s = {}
        
        for view_name in self.attr_emb.view_names:
            view_emb = self.attr_emb.view_embeddings[view_name]
            
            # Top-M 클래스만
            top_m_emb = view_emb[top_m_indices]  # [M, embed]
            
            # Logits & softmax (Top-M 내에서)
            logits = np.dot(top_m_emb, region_feat) / self.temperature
            posterior = softmax(logits)  # [M]
            posteriors.append(posterior)
            
            if debug:
                view_ranking = np.argsort(posterior)[-5:]
                view_top5s[view_name] = [
                    (self.attr_emb.sorted_class_names[top_m_indices[i]], posterior[i])
                    for i in view_ranking
                ]
        
        posteriors = np.stack(posteriors, axis=0)  # [K, M]
        
        if debug:
            debug_info["view_top5s"] = view_top5s
        
        # 4. JS divergence
        u_sem = js_divergence(posteriors)
        debug_info["u_sem"] = u_sem
        
        return u_sem, region_feat, debug_info
    
    def compute_artifactness(self, region_feat: np.ndarray) -> float:
        """
        Region feature의 artifactness score 계산
        
        Returns:
            margin = max(depiction_sim) - max(real_sim)
        """
        if region_feat is None:
            return 0.0
        
        if self.attr_emb.depiction_embeddings is None:
            return 0.0
        
        dep_sims = np.dot(self.attr_emb.depiction_embeddings, region_feat)
        real_sims = np.dot(self.attr_emb.real_embeddings, region_feat)
        
        return float(np.max(dep_sims) - np.max(real_sims))
    
    def run_inference_and_compute(self, 
                                   image_path: str,
                                   detections: List[Detection],
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
        """
        이미지에 대해 YOLOE inference 후 u_sem, artifactness 계산
        
        Args:
            image_path: 이미지 경로
            detections: 해당 이미지의 detection 리스트
            verbose: 진행상황 출력
        
        Returns:
            (u_sems, s_arts, region_features, drop_stats)
        """
        if not detections:
            return np.array([]), np.array([]), [], {"total": 0, "valid": 0}
        
        # 1. YOLOE inference (region feature 캡처)
        # 이미 detection phase에서 inference가 끝났으므로, 
        # head에 저장된 _region_features를 사용
        # 단, fuse 전에만 유효함
        
        # 이미지 크기
        img = Image.open(image_path)
        img_shape = (img.height, img.width)
        
        # 2. 각 detection에 대해 u_sem, artifactness 계산
        u_sems = []
        s_arts = []
        region_feats = []
        
        drop_stats = {"total": len(detections), "valid": 0, "no_region_feature": 0, "no_embeddings": 0}
        
        for det in detections:
            u_sem, feat, debug_info = self.compute_for_detection(det, img_shape)
            art = self.compute_artifactness(feat) if feat is not None else 0.0
            
            u_sems.append(u_sem)
            s_arts.append(art)
            region_feats.append(feat)
            
            if debug_info["status"] == "ok":
                drop_stats["valid"] += 1
            else:
                reason = debug_info.get("drop_reason", "unknown")
                if "no_region_feature" in reason:
                    drop_stats["no_region_feature"] += 1
                elif "no_embeddings" in reason:
                    drop_stats["no_embeddings"] += 1
        
        return np.array(u_sems), np.array(s_arts), region_feats, drop_stats
    
    def unit_test_single_detection(self, detection: Detection, img_shape: Tuple[int, int]):
        """
        단일 detection에 대한 유닛테스트 - view 간 차이 확인
        """
        print("\n" + "="*60)
        print("U_SEM UNIT TEST (YOLOE Region Feature)")
        print("="*60)
        
        u_sem, feat, debug_info = self.compute_for_detection(detection, img_shape, debug=True)
        
        print(f"Detection: {detection.pred_class_name} (conf={detection.confidence:.3f})")
        print(f"Status: {debug_info['status']}")
        
        if debug_info['status'] != 'ok':
            print(f"Drop reason: {debug_info['drop_reason']}")
            return
        
        print(f"Region feature shape: {feat.shape if feat is not None else 'None'}")
        print(f"\nu_sem = {u_sem:.6f}")
        
        if feat is not None:
            art = self.compute_artifactness(feat)
            print(f"artifactness = {art:.6f}")
        
        print(f"\nBase view top-5:")
        for name, prob in debug_info.get("top5_base", []):
            print(f"  {name}: {prob:.4f}")
        
        print(f"\nPer-view top-5 (view별로 다르면 u_sem이 커야 함):")
        view_top5s = debug_info.get("view_top5s", {})
        for view_name, top5 in view_top5s.items():
            print(f"\n  [{view_name}]")
            for name, prob in top5:
                print(f"    {name}: {prob:.4f}")
        
        # View 간 일치도 분석
        print(f"\n분석:")
        all_top1s = []
        for view_name, top5 in view_top5s.items():
            if top5:
                all_top1s.append(top5[-1][0])
        
        unique_top1s = set(all_top1s)
        print(f"  View별 top-1 클래스: {all_top1s}")
        print(f"  Unique top-1 수: {len(unique_top1s)}/{len(all_top1s)}")
        
        if len(unique_top1s) == 1:
            print(f"  → 모든 view가 동일한 top-1 → u_sem ≈ 0 정상")
        else:
            print(f"  → View마다 top-1이 다름 → u_sem > 0 예상")


# =============================================================================
# Enhanced Triad Split (Artifactness Score 기반)
# =============================================================================

class EnhancedTriadSplit:
    """
    개선된 Triad Split: Semantic FP를 artifactness score로 세분화
    
    분류:
    - TP: True Positives
    - Depiction_FP: artifactness score > threshold인 Semantic FP (H1 대상)
    - ClassConfusion_FP: 나머지 Semantic FP
    - Background_FP: GT와 매칭 안 됨
    """
    
    def __init__(self, depiction_threshold: float = 0.0):
        """
        Args:
            depiction_threshold: artifactness score > threshold면 Depiction_FP
        """
        self.depiction_threshold = depiction_threshold
    
    def split_detections(self, 
                         detections: List[Detection],
                         artifactness_scores: np.ndarray) -> Dict[str, List[Tuple[Detection, float]]]:
        """
        Detection 리스트를 4개 그룹으로 분할
        
        Args:
            detections: Detection 리스트
            artifactness_scores: 각 detection의 artifactness score
        
        Returns:
            {group_name: [(Detection, artifactness_score), ...]}
        """
        groups = {
            "TP": [],
            "Depiction_FP": [],
            "ClassConfusion_FP": [],
            "Background_FP": [],
        }
        
        for det, art_score in zip(detections, artifactness_scores):
            if det.is_tp:
                groups["TP"].append((det, art_score))
            elif det.matched_gt_idx is None:
                groups["Background_FP"].append((det, art_score))
            else:
                # Semantic FP → artifactness score로 세분화
                if art_score > self.depiction_threshold:
                    groups["Depiction_FP"].append((det, art_score))
                else:
                    groups["ClassConfusion_FP"].append((det, art_score))
        
        return groups
    
    def print_split_stats(self, groups: Dict):
        """Split 통계 출력"""
        print("\n  Enhanced Triad Split (artifactness-based):")
        for name, items in groups.items():
            if items:
                scores = [s for _, s in items]
                print(f"    {name}: {len(items)} detections, "
                      f"mean_art={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            else:
                print(f"    {name}: 0 detections")


# =============================================================================
# Backward compatibility
# =============================================================================

class MobileCLIPScorer:
    """Backward compatibility - 이제 사용하지 않음"""
    def __init__(self, device: str = "cuda"):
        print("  WARNING: MobileCLIPScorer is deprecated. Use SemanticUncertaintyCalculator instead.")
        self.model_name = "DEPRECATED"
