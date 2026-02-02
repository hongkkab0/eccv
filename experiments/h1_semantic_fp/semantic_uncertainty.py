"""
Semantic Uncertainty (u_sem) Calculation
=========================================
YOLOE region feature + 캐시된 MobileCLIP embeddings → JS divergence

핵심:
1. YOLOE head에서 cv3 출력 (512-dim anchor feature) 캡처 (fuse 전!)
2. 캐시된 MobileCLIP text embeddings 사용 (모델 로드 X)
3. anchor feature · text_emb → view별 posterior → JS

장점:
- MobileCLIP 모델 로드 불필요 (캐시만 사용)
- YOLOE 파이프라인 유지
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
# 캐시된 MobileCLIP Embeddings 로더
# =============================================================================

class CachedEmbeddings:
    """
    캐시된 MobileCLIP text embeddings 로더
    
    MobileCLIP 모델 로드 없이 미리 생성된 embeddings만 사용
    """
    
    # Attribute view 템플릿 (캐시 생성 시 사용된 것과 동일해야 함)
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
    
    # Depiction/Real 프롬프트
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
    
    def __init__(self, 
                 class_names: Dict[int, str],
                 cache_dir: str = "tools/mobileclip_blt",
                 device: str = "cuda"):
        self.class_names = class_names
        self.cache_dir = Path(cache_dir)
        self.device = device
        
        # 클래스 이름 리스트 (정렬된 순서)
        self.sorted_class_indices = sorted(class_names.keys())
        self.sorted_class_names = [class_names[i] for i in self.sorted_class_indices]
        self.num_classes = len(self.sorted_class_names)
        
        # View 이름 리스트 (base 제외)
        self.view_names = [k for k in self.ATTRIBUTE_VIEWS.keys() if k != "base"]
        self.num_views = len(self.view_names)
        
        # Embeddings
        self.base_embeddings = None      # [num_classes, embed_dim]
        self.view_embeddings = {}        # {view_name: [num_classes, embed_dim]}
        self.depiction_embeddings = None
        self.real_embeddings = None
        
        self._load_or_generate_embeddings()
    
    def _clean_class_name(self, name: str) -> str:
        """클래스 이름 정리"""
        return name.split("/")[0].strip().replace("_", " ")
    
    def _load_or_generate_embeddings(self):
        """캐시 로드 또는 생성"""
        label_cache = self.cache_dir / "lvis_label_embeddings.pt"
        attr_cache = self.cache_dir / "lvis_attribute_embeddings.pt"
        art_cache = self.cache_dir / "artifactness_embeddings.pt"
        
        # 캐시 존재 여부 확인
        if label_cache.exists() and attr_cache.exists():
            print(f"  Loading cached embeddings from {self.cache_dir}")
            self._load_from_cache(label_cache, attr_cache, art_cache)
        else:
            print(f"  Cache not found at {self.cache_dir}")
            print(f"  Generating embeddings using YOLOE's text model...")
            self._generate_embeddings()
    
    def _load_from_cache(self, label_cache: Path, attr_cache: Path, art_cache: Path):
        """캐시에서 로드"""
        # Label embeddings (base view)
        label_emb = torch.load(label_cache, map_location=self.device)
        # label_emb는 {class_name: embedding} 또는 tensor 형태일 수 있음
        
        if isinstance(label_emb, dict):
            # {name: emb} 형태
            base_list = []
            for name in self.sorted_class_names:
                clean = self._clean_class_name(name)
                prompt = self.ATTRIBUTE_VIEWS["base"].format(cls=clean)
                if prompt in label_emb:
                    base_list.append(label_emb[prompt].cpu().numpy())
                elif clean in label_emb:
                    base_list.append(label_emb[clean].cpu().numpy())
                else:
                    # fallback: 0 벡터
                    base_list.append(np.zeros(512))
            self.base_embeddings = np.stack(base_list)
        else:
            # tensor 형태 [num_classes, embed_dim]
            self.base_embeddings = label_emb.cpu().numpy()
        
        print(f"    Base embeddings: {self.base_embeddings.shape}")
        
        # Attribute embeddings (각 view)
        attr_emb = torch.load(attr_cache, map_location=self.device)
        
        if isinstance(attr_emb, dict):
            for view_name in self.view_names:
                template = self.ATTRIBUTE_VIEWS[view_name]
                view_list = []
                for name in self.sorted_class_names:
                    clean = self._clean_class_name(name)
                    prompt = template.format(cls=clean)
                    if prompt in attr_emb:
                        view_list.append(attr_emb[prompt].cpu().numpy())
                    else:
                        view_list.append(np.zeros(512))
                self.view_embeddings[view_name] = np.stack(view_list)
        
        print(f"    View embeddings: {len(self.view_embeddings)} views")
        
        # Artifactness embeddings
        if art_cache.exists():
            art_emb = torch.load(art_cache, map_location=self.device)
            if isinstance(art_emb, dict):
                dep_list = [art_emb[p].cpu().numpy() for p in self.DEPICTION_PROMPTS if p in art_emb]
                real_list = [art_emb[p].cpu().numpy() for p in self.REAL_PROMPTS if p in art_emb]
                if dep_list:
                    self.depiction_embeddings = np.stack(dep_list)
                if real_list:
                    self.real_embeddings = np.stack(real_list)
            print(f"    Artifactness embeddings loaded")
    
    def _generate_embeddings(self):
        """YOLOE의 text model로 embeddings 생성"""
        try:
            from ultralytics.nn.text_model import build_text_model
            
            print(f"  Building text model...")
            text_model = build_text_model("mobileclip:blt", device=self.device)
            text_model.eval()
            
            clean_names = [self._clean_class_name(n) for n in self.sorted_class_names]
            
            with torch.no_grad():
                # Base embeddings
                base_texts = [self.ATTRIBUTE_VIEWS["base"].format(cls=n) for n in clean_names]
                base_tokens = text_model.tokenize(base_texts)
                base_feats = text_model.encode_text(base_tokens)
                base_feats = F.normalize(base_feats, dim=-1)
                self.base_embeddings = base_feats.cpu().numpy()
                print(f"    Base embeddings: {self.base_embeddings.shape}")
                
                # View embeddings
                for view_name in self.view_names:
                    template = self.ATTRIBUTE_VIEWS[view_name]
                    view_texts = [template.format(cls=n) for n in clean_names]
                    view_tokens = text_model.tokenize(view_texts)
                    view_feats = text_model.encode_text(view_tokens)
                    view_feats = F.normalize(view_feats, dim=-1)
                    self.view_embeddings[view_name] = view_feats.cpu().numpy()
                print(f"    View embeddings: {len(self.view_embeddings)} views")
                
                # Depiction/Real embeddings
                dep_tokens = text_model.tokenize(self.DEPICTION_PROMPTS)
                dep_feats = text_model.encode_text(dep_tokens)
                dep_feats = F.normalize(dep_feats, dim=-1)
                self.depiction_embeddings = dep_feats.cpu().numpy()
                
                real_tokens = text_model.tokenize(self.REAL_PROMPTS)
                real_feats = text_model.encode_text(real_tokens)
                real_feats = F.normalize(real_feats, dim=-1)
                self.real_embeddings = real_feats.cpu().numpy()
                print(f"    Artifactness embeddings generated")
            
            # 캐시 저장
            self._save_cache()
            
        except Exception as e:
            print(f"  ERROR: Failed to generate embeddings: {e}")
            # Fallback: 빈 embeddings
            self.base_embeddings = np.zeros((self.num_classes, 512))
            for view_name in self.view_names:
                self.view_embeddings[view_name] = np.zeros((self.num_classes, 512))
    
    def _save_cache(self):
        """캐시 저장"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장 형식: {prompt: embedding}
        clean_names = [self._clean_class_name(n) for n in self.sorted_class_names]
        
        # Label embeddings
        label_dict = {}
        for i, name in enumerate(clean_names):
            prompt = self.ATTRIBUTE_VIEWS["base"].format(cls=name)
            label_dict[prompt] = torch.from_numpy(self.base_embeddings[i])
        torch.save(label_dict, self.cache_dir / "lvis_label_embeddings.pt")
        
        # Attribute embeddings
        attr_dict = {}
        for view_name in self.view_names:
            template = self.ATTRIBUTE_VIEWS[view_name]
            for i, name in enumerate(clean_names):
                prompt = template.format(cls=name)
                attr_dict[prompt] = torch.from_numpy(self.view_embeddings[view_name][i])
        torch.save(attr_dict, self.cache_dir / "lvis_attribute_embeddings.pt")
        
        # Artifactness embeddings
        art_dict = {}
        if self.depiction_embeddings is not None:
            for i, p in enumerate(self.DEPICTION_PROMPTS):
                art_dict[p] = torch.from_numpy(self.depiction_embeddings[i])
        if self.real_embeddings is not None:
            for i, p in enumerate(self.REAL_PROMPTS):
                art_dict[p] = torch.from_numpy(self.real_embeddings[i])
        torch.save(art_dict, self.cache_dir / "artifactness_embeddings.pt")
        
        print(f"  Saved embeddings cache to {self.cache_dir}")


# =============================================================================
# Semantic Uncertainty Calculator
# =============================================================================

class SemanticUncertaintyCalculator:
    """
    Semantic Uncertainty 계산기
    
    사용하는 것:
    - detection.region_feature (YOLOE cv3 output, fuse 전 캡처)
    - 캐시된 MobileCLIP text embeddings
    
    계산:
    - region_feature · base_embeddings → top-M 클래스 선택
    - region_feature · view_embeddings[k] → view별 posterior
    - JS(posteriors) = u_sem
    """
    
    def __init__(self,
                 class_names: Dict[int, str],
                 device: str = "cuda",
                 temperature: float = 10.0,
                 top_m: int = 20,
                 cache_dir: str = "tools/mobileclip_blt"):
        self.class_names = class_names
        self.device = device
        self.temperature = temperature
        self.top_m = top_m
        
        # 캐시된 embeddings 로드
        print(f"\n--- Loading/Generating Embeddings ---")
        self.emb = CachedEmbeddings(class_names, cache_dir, device)
        
        print(f"    Temperature: {self.temperature}, Top-M: {self.top_m}")
    
    def compute_u_sem_from_logits(self, cls_logits: np.ndarray) -> float:
        """
        cls_logits에서 uncertainty 계산 (fused 모델용)
        
        fused 모델에서는 view별 분포를 얻을 수 없으므로,
        base logits의 top-M 엔트로피를 사용
        
        Args:
            cls_logits: [nc] pre-sigmoid logits
        
        Returns:
            uncertainty 값 (normalized entropy)
        """
        if cls_logits is None:
            return 0.0
        
        # Top-M logits → posterior
        top_m_indices = np.argsort(cls_logits)[-self.top_m:]
        top_m_logits = cls_logits[top_m_indices] / self.temperature
        top_m_posterior = softmax(top_m_logits)
        
        # Normalized entropy (0~1)
        h = entropy(top_m_posterior + 1e-10)
        h_max = np.log(self.top_m)  # maximum entropy
        
        return float(h / h_max) if h_max > 0 else 0.0
    
    def compute_u_sem(self, region_feature: np.ndarray) -> float:
        """
        단일 region feature의 u_sem 계산 (unfused 모델용)
        
        Args:
            region_feature: [512] anchor feature
        
        Returns:
            u_sem 값
        """
        if region_feature is None or self.emb.base_embeddings is None:
            return 0.0
        
        # 1. Base posterior → Top-M 클래스
        base_logits = np.dot(self.emb.base_embeddings, region_feature) / self.temperature
        base_posterior = softmax(base_logits)
        
        # Top-M 클래스 인덱스
        top_m_indices = np.argsort(base_posterior)[-self.top_m:]
        
        # 2. 각 view의 posterior (Top-M만)
        posteriors = []
        for view_name in self.emb.view_names:
            view_emb = self.emb.view_embeddings.get(view_name)
            if view_emb is None:
                continue
            
            # Top-M 클래스만
            top_m_emb = view_emb[top_m_indices]  # [M, 512]
            
            # Logits & softmax (Top-M 내에서)
            logits = np.dot(top_m_emb, region_feature) / self.temperature
            posterior = softmax(logits)  # [M]
            posteriors.append(posterior)
        
        if len(posteriors) < 2:
            return 0.0
        
        posteriors = np.stack(posteriors, axis=0)  # [K, M]
        
        # 3. JS divergence
        return js_divergence(posteriors)
    
    def compute_artifactness(self, region_feature: np.ndarray) -> float:
        """
        Artifactness score 계산
        
        Returns:
            margin = max(depiction_sim) - max(real_sim)
        """
        if region_feature is None:
            return 0.0
        if self.emb.depiction_embeddings is None or self.emb.real_embeddings is None:
            return 0.0
        
        dep_sims = np.dot(self.emb.depiction_embeddings, region_feature)
        real_sims = np.dot(self.emb.real_embeddings, region_feature)
        
        return float(np.max(dep_sims) - np.max(real_sims))
    
    def compute_for_detection(self, detection: Detection) -> Tuple[float, float, Dict]:
        """
        단일 detection의 u_sem, s_art 계산
        
        우선순위:
        1. region_feature가 있으면: full u_sem (JS divergence) + artifactness
        2. cls_logits만 있으면: entropy-based uncertainty + artifactness=0
        3. 둘 다 없으면: drop
        
        Returns:
            (u_sem, s_art, debug_info)
        """
        debug_info = {"status": "ok", "source": "none"}
        
        # 1. region_feature가 있으면 (unfused 모델)
        if detection.region_feature is not None:
            feat = detection.region_feature
            u_sem = self.compute_u_sem(feat)
            s_art = self.compute_artifactness(feat)
            debug_info["source"] = "region_feature"
            return u_sem, s_art, debug_info
        
        # 2. cls_logits만 있으면 (fused 모델)
        if detection.cls_logits is not None:
            u_sem = self.compute_u_sem_from_logits(detection.cls_logits)
            s_art = 0.0  # fused 모델에서는 artifactness 계산 불가
            debug_info["source"] = "cls_logits"
            return u_sem, s_art, debug_info
        
        # 3. 둘 다 없으면 drop
        return 0.0, 0.0, {"status": "dropped", "reason": "no_feature_or_logits"}
    
    def compute_for_detections(self, detections: List[Detection], 
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        여러 detection의 u_sem, s_art 일괄 계산
        
        Returns:
            (u_sem_array, s_art_array, drop_stats)
        """
        u_sems = []
        s_arts = []
        
        drop_stats = {
            "total": len(detections),
            "valid": 0,
            "from_region_feature": 0,
            "from_cls_logits": 0,
            "dropped": 0,
        }
        
        for i, det in enumerate(detections):
            if verbose and (i + 1) % 100 == 0:
                print(f"    Processing {i+1}/{len(detections)}...")
            
            u_sem, s_art, info = self.compute_for_detection(det)
            u_sems.append(u_sem)
            s_arts.append(s_art)
            
            if info["status"] == "ok":
                drop_stats["valid"] += 1
                if info.get("source") == "region_feature":
                    drop_stats["from_region_feature"] += 1
                elif info.get("source") == "cls_logits":
                    drop_stats["from_cls_logits"] += 1
            else:
                drop_stats["dropped"] += 1
        
        if verbose:
            print(f"    Drop stats: valid={drop_stats['valid']}/{drop_stats['total']}")
            print(f"      from_region_feature: {drop_stats['from_region_feature']}")
            print(f"      from_cls_logits: {drop_stats['from_cls_logits']}")
            print(f"      dropped: {drop_stats['dropped']}")
        
        return np.array(u_sems), np.array(s_arts), drop_stats
    
    def sanity_check(self, detection: Detection):
        """
        Sanity check: posterior가 균등이 아닌지, u_sem이 의미 있는지 확인
        """
        print("\n" + "="*60)
        print("SANITY CHECK")
        print("="*60)
        
        if detection.region_feature is None:
            print("ERROR: No region feature")
            return
        
        feat = detection.region_feature
        print(f"Region feature: shape={feat.shape}, norm={np.linalg.norm(feat):.4f}")
        
        # Base posterior
        base_logits = np.dot(self.emb.base_embeddings, feat) / self.temperature
        base_posterior = softmax(base_logits)
        
        # Uniform vs actual
        uniform_prob = 1.0 / len(base_posterior)
        max_prob = np.max(base_posterior)
        top5_idx = np.argsort(base_posterior)[-5:]
        
        print(f"\nBase posterior (temperature={self.temperature}):")
        print(f"  Uniform prob: {uniform_prob:.6f}")
        print(f"  Max prob: {max_prob:.6f}")
        print(f"  Max/Uniform ratio: {max_prob/uniform_prob:.2f}x")
        
        if max_prob < uniform_prob * 2:
            print("  WARNING: Posterior too uniform! Check temperature or feature normalization.")
        
        print(f"\nTop-5 classes:")
        for idx in reversed(top5_idx):
            name = self.emb.sorted_class_names[idx]
            prob = base_posterior[idx]
            print(f"    {name}: {prob:.6f}")
        
        # u_sem
        u_sem = self.compute_u_sem(feat)
        s_art = self.compute_artifactness(feat)
        
        print(f"\nu_sem = {u_sem:.6f}")
        print(f"s_art = {s_art:.6f}")
        
        if u_sem < 1e-4:
            print("WARNING: u_sem very small. Views might be too similar.")


# =============================================================================
# Enhanced Triad Split (Artifactness Score 기반)
# =============================================================================

class EnhancedTriadSplit:
    """
    개선된 Triad Split: Semantic FP를 artifactness score로 세분화
    
    분류:
    - TP: True Positives
    - Depiction_FP: artifactness score > threshold (H1 대상)
    - ClassConfusion_FP: 나머지 Semantic FP
    - Background_FP: GT와 매칭 안 됨
    """
    
    def __init__(self, depiction_threshold: float = 0.0):
        self.depiction_threshold = depiction_threshold
    
    def split_detections(self, 
                         detections: List[Detection],
                         artifactness_scores: np.ndarray) -> Dict[str, List[Tuple[Detection, float]]]:
        """
        Detection 리스트를 4개 그룹으로 분할
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

class YOLOERegionFeatureExtractor:
    """Deprecated - region feature는 detection phase에서 직접 캡처"""
    pass

class MobileCLIPAttributeEmbeddings:
    """Deprecated - CachedEmbeddings 사용"""
    pass

class MobileCLIPScorer:
    """Deprecated"""
    pass
