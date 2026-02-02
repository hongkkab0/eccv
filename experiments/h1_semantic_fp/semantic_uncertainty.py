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
    
    def compute_u_sem(self, region_feature: np.ndarray, cls_logits: np.ndarray = None) -> float:
        """
        캘리브레이션된 u_sem 계산
        
        핵심: cls_logits의 스케일에 맞춰서 view posterior를 계산
        - cls_logits가 있으면: 스케일 캘리브레이션 적용
        - cls_logits가 없으면: 기존 방식 (temperature 사용)
        
        Args:
            region_feature: [512] anchor feature (L2 normalized됨)
            cls_logits: [nc] pre-sigmoid logits from YOLOE head
        
        Returns:
            u_sem 값
        """
        if region_feature is None or self.emb.base_embeddings is None:
            return 0.0
        
        # ===== Step 1: region feature L2 normalize =====
        region_norm = region_feature / (np.linalg.norm(region_feature) + 1e-8)
        
        # ===== Step 2: base posterior (cls_logits 또는 dot-product) =====
        if cls_logits is not None:
            # cls_logits가 있으면 그대로 사용 (YOLOE 스케일)
            base_logits = cls_logits
            
            # 스케일 캘리브레이션: alpha = (sim · logits) / (sim · sim)
            # normalized dot-product로 raw similarity 계산
            raw_sim = np.dot(self.emb.base_embeddings, region_norm)  # [nc]
            
            # alpha 계산 (least squares)
            sim_flat = raw_sim.flatten()
            logits_flat = base_logits.flatten()
            alpha = np.dot(sim_flat, logits_flat) / (np.dot(sim_flat, sim_flat) + 1e-8)
            alpha = max(alpha, 1.0)  # alpha가 너무 작으면 클리핑
        else:
            # cls_logits가 없으면 기존 방식
            base_logits = np.dot(self.emb.base_embeddings, region_norm) / self.temperature
            alpha = 1.0 / self.temperature
        
        base_posterior = softmax(base_logits)
        
        # Top-M 클래스 인덱스
        top_m_indices = np.argsort(base_posterior)[-self.top_m:]
        
        # ===== Step 3: base view posterior (Top-M만) =====
        # base view도 posteriors에 포함
        base_top_m_logits = base_logits[top_m_indices]
        posteriors = [softmax(base_top_m_logits)]
        
        # ===== Step 4: 각 attribute view의 posterior (Top-M만, 캘리브레이션된 스케일) =====
        for view_name in self.emb.view_names:
            view_emb = self.emb.view_embeddings.get(view_name)
            if view_emb is None:
                continue
            
            # Top-M 클래스만
            top_m_emb = view_emb[top_m_indices]  # [M, 512]
            
            # normalized dot-product → alpha로 스케일 맞춤
            raw_sim_view = np.dot(top_m_emb, region_norm)  # [M]
            logits_view = alpha * raw_sim_view
            
            posterior = softmax(logits_view)  # [M]
            posteriors.append(posterior)
        
        if len(posteriors) < 2:
            return 0.0
        
        posteriors = np.stack(posteriors, axis=0)  # [K+1, M] (base + K views)
        
        # ===== Step 5: JS divergence =====
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
        1. region_feature + cls_logits: 캘리브레이션된 u_sem (best)
        2. region_feature만: temperature 기반 u_sem
        3. cls_logits만: entropy-based uncertainty
        4. 둘 다 없으면: drop
        
        Returns:
            (u_sem, s_art, debug_info)
        """
        debug_info = {"status": "ok", "source": "none"}
        
        # 1. region_feature + cls_logits (best: 캘리브레이션된 스케일)
        if detection.region_feature is not None and detection.cls_logits is not None:
            feat = detection.region_feature
            logits = detection.cls_logits
            u_sem = self.compute_u_sem(feat, logits)
            s_art = self.compute_artifactness(feat)
            debug_info["source"] = "region_feature+cls_logits"
            return u_sem, s_art, debug_info
        
        # 2. region_feature만 (cls_logits 없음)
        if detection.region_feature is not None:
            feat = detection.region_feature
            u_sem = self.compute_u_sem(feat, None)
            s_art = self.compute_artifactness(feat)
            debug_info["source"] = "region_feature"
            return u_sem, s_art, debug_info
        
        # 3. cls_logits만 있으면 (fused 모델)
        if detection.cls_logits is not None:
            u_sem = self.compute_u_sem_from_logits(detection.cls_logits)
            s_art = 0.0  # fused 모델에서는 artifactness 계산 불가
            debug_info["source"] = "cls_logits"
            return u_sem, s_art, debug_info
        
        # 4. 둘 다 없으면 drop
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
                source = info.get("source", "")
                if "region_feature" in source:
                    drop_stats["from_region_feature"] += 1
                elif source == "cls_logits":
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
        캘리브레이션 효과 비교
        """
        print("\n" + "="*60)
        print("SANITY CHECK (with calibration)")
        print("="*60)
        
        if detection.region_feature is None:
            print("ERROR: No region feature")
            return
        
        feat = detection.region_feature
        cls_logits = detection.cls_logits
        
        print(f"Region feature: shape={feat.shape}, norm={np.linalg.norm(feat):.4f}")
        if cls_logits is not None:
            print(f"Cls logits: shape={cls_logits.shape}, min={cls_logits.min():.2f}, max={cls_logits.max():.2f}")
        
        # L2 normalize
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        
        # ===== 방법 1: 기존 (temperature만) =====
        base_logits_old = np.dot(self.emb.base_embeddings, feat_norm) / self.temperature
        base_posterior_old = softmax(base_logits_old)
        
        uniform_prob = 1.0 / len(base_posterior_old)
        max_prob_old = np.max(base_posterior_old)
        
        print(f"\n[OLD] temperature={self.temperature}:")
        print(f"  Uniform prob: {uniform_prob:.6f}")
        print(f"  Max prob: {max_prob_old:.6f}")
        print(f"  Max/Uniform ratio: {max_prob_old/uniform_prob:.2f}x")
        
        # ===== 방법 2: 캘리브레이션 (cls_logits 기준) =====
        if cls_logits is not None:
            # alpha 계산
            raw_sim = np.dot(self.emb.base_embeddings, feat_norm)
            alpha = np.dot(raw_sim, cls_logits) / (np.dot(raw_sim, raw_sim) + 1e-8)
            alpha = max(alpha, 1.0)
            
            # cls_logits를 base posterior로 사용
            base_posterior_new = softmax(cls_logits)
            max_prob_new = np.max(base_posterior_new)
            
            print(f"\n[NEW] calibrated (alpha={alpha:.2f}):")
            print(f"  Max prob: {max_prob_new:.6f}")
            print(f"  Max/Uniform ratio: {max_prob_new/uniform_prob:.2f}x")
            print(f"  Improvement: {max_prob_new/max_prob_old:.2f}x sharper")
            
            top5_idx = np.argsort(base_posterior_new)[-5:]
        else:
            top5_idx = np.argsort(base_posterior_old)[-5:]
            print("\n  (No cls_logits - using old method)")
        
        print(f"\nTop-5 classes:")
        posterior_to_show = softmax(cls_logits) if cls_logits is not None else base_posterior_old
        for idx in reversed(top5_idx):
            name = self.emb.sorted_class_names[idx]
            prob = posterior_to_show[idx]
            print(f"    {name}: {prob:.6f}")
        
        # u_sem 비교
        u_sem_old = self.compute_u_sem(feat, None)
        u_sem_new = self.compute_u_sem(feat, cls_logits) if cls_logits is not None else u_sem_old
        s_art = self.compute_artifactness(feat)
        
        print(f"\nu_sem (old): {u_sem_old:.6f}")
        print(f"u_sem (calibrated): {u_sem_new:.6f}")
        print(f"s_art: {s_art:.6f}")
        
        if cls_logits is not None and u_sem_new > u_sem_old * 1.5:
            print("OK: Calibration improved u_sem significantly")
        elif u_sem_new < 1e-3:
            print("WARNING: u_sem still very small even with calibration")


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
# Semantic FP 재분해: Near-miss vs Hard-negative
# =============================================================================

class SemanticFPSplitter:
    """
    Semantic FP를 pred/gt 텍스트 유사도 기준으로 분해
    
    Near-miss FP: pred와 gt가 의미적으로 유사 (chair vs armchair)
    Hard-negative FP: pred와 gt가 의미적으로 다름 (chair vs table)
    """
    
    def __init__(self, 
                 class_embeddings: np.ndarray,  # [nc, 512]
                 class_names: Dict[int, str],
                 near_miss_threshold: float = 0.7,  # cosine similarity
                 hard_negative_threshold: float = 0.3):
        """
        Args:
            class_embeddings: 클래스별 텍스트 임베딩
            class_names: {idx: name}
            near_miss_threshold: 이 이상이면 near-miss
            hard_negative_threshold: 이 이하면 hard-negative
        """
        self.class_emb = class_embeddings
        self.class_names = class_names
        self.near_miss_threshold = near_miss_threshold
        self.hard_negative_threshold = hard_negative_threshold
        
        # L2 normalize embeddings
        norms = np.linalg.norm(self.class_emb, axis=1, keepdims=True)
        self.class_emb_norm = self.class_emb / (norms + 1e-8)
    
    def get_class_similarity(self, class_a: int, class_b: int) -> float:
        """두 클래스 간 코사인 유사도"""
        if class_a >= len(self.class_emb_norm) or class_b >= len(self.class_emb_norm):
            return 0.0
        return float(np.dot(self.class_emb_norm[class_a], self.class_emb_norm[class_b]))
    
    def split_semantic_fps(self, 
                           detections: List[Detection],
                           u_sems: np.ndarray) -> Dict[str, List[Tuple[Detection, float]]]:
        """
        Semantic FP를 3가지로 분해
        
        Returns:
            {
                "NearMiss_FP": [(det, u_sem), ...],  # pred/gt 유사 (동의어/계층 충돌)
                "HardNegative_FP": [(det, u_sem), ...],  # pred/gt 다름 (진짜 오류)
                "Ambiguous_FP": [(det, u_sem), ...],  # 중간
            }
        """
        groups = {
            "NearMiss_FP": [],
            "HardNegative_FP": [],
            "Ambiguous_FP": [],
        }
        
        for det, u_sem in zip(detections, u_sems):
            # Semantic FP만 처리
            if det.triad_label != "Semantic_FP":
                continue
            
            # pred/gt 클래스 유사도 계산
            pred_cls = det.pred_class
            gt_cls = det.overlapping_gt_class if det.overlapping_gt_class is not None else det.matched_gt_class
            
            if gt_cls is None:
                groups["Ambiguous_FP"].append((det, u_sem))
                continue
            
            sim = self.get_class_similarity(pred_cls, gt_cls)
            
            if sim >= self.near_miss_threshold:
                groups["NearMiss_FP"].append((det, u_sem))
            elif sim <= self.hard_negative_threshold:
                groups["HardNegative_FP"].append((det, u_sem))
            else:
                groups["Ambiguous_FP"].append((det, u_sem))
        
        return groups
    
    def print_split_stats(self, groups: Dict, u_sems: np.ndarray = None):
        """Split 통계 출력"""
        print("\n  Semantic FP Split (by pred/gt similarity):")
        for name, items in groups.items():
            if items:
                u_vals = [u for _, u in items]
                print(f"    {name}: {len(items)} detections")
                print(f"      u_sem: mean={np.mean(u_vals):.4f}, std={np.std(u_vals):.4f}")
                
                # 예시 출력
                if len(items) > 0:
                    det, u = items[0]
                    gt_cls = det.overlapping_gt_class if det.overlapping_gt_class is not None else det.matched_gt_class
                    pred_name = self.class_names.get(det.pred_class, "?")
                    gt_name = self.class_names.get(gt_cls, "?") if gt_cls is not None else "?"
                    sim = self.get_class_similarity(det.pred_class, gt_cls) if gt_cls else 0
                    print(f"      Example: {pred_name} vs {gt_name} (sim={sim:.2f})")
            else:
                print(f"    {name}: 0 detections")
    
    def get_confusion_matrix_subset(self, detections: List[Detection], top_k: int = 20):
        """
        가장 많이 혼동되는 클래스 쌍 반환
        """
        confusion_pairs = {}
        
        for det in detections:
            if det.triad_label != "Semantic_FP":
                continue
            
            pred_cls = det.pred_class
            gt_cls = det.overlapping_gt_class if det.overlapping_gt_class is not None else det.matched_gt_class
            
            if gt_cls is None:
                continue
            
            pair = (min(pred_cls, gt_cls), max(pred_cls, gt_cls))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # 상위 K개 반환
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:top_k]
        
        result = []
        for (cls_a, cls_b), count in sorted_pairs:
            name_a = self.class_names.get(cls_a, f"cls_{cls_a}")
            name_b = self.class_names.get(cls_b, f"cls_{cls_b}")
            sim = self.get_class_similarity(cls_a, cls_b)
            result.append({
                "classes": (cls_a, cls_b),
                "names": (name_a, name_b),
                "count": count,
                "similarity": sim,
            })
        
        return result


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
