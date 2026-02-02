"""
Semantic Uncertainty (u_sem) Calculation
=========================================
MobileCLIP 기반 JS divergence semantic uncertainty

핵심 변경:
1. MobileCLIP 올바른 로드 (mobileclip 라이브러리 사용)
2. Depiction_FP를 artifactness score 기반으로 슬라이싱 (클래스 리스트 X)
3. u_sem 유닛테스트 함수 추가
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
# MobileCLIP Scorer (올바른 로드)
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
        self.model_name = None
        self._load_mobileclip()
    
    def _load_mobileclip(self):
        """MobileCLIP 모델 로드 - 올바른 모델 이름 사용"""
        try:
            import mobileclip
            
            # mobileclip 라이브러리의 올바른 모델 이름들:
            # 'mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b', 'mobileclip_blt'
            # 체크포인트 경로도 확인
            checkpoint_path = Path('checkpoints/mobileclip_blt.pt')
            
            if not checkpoint_path.exists():
                # 대안 경로들 시도
                alt_paths = [
                    Path('mobileclip_blt.pt'),
                    Path('../checkpoints/mobileclip_blt.pt'),
                    Path('weights/mobileclip_blt.pt'),
                ]
                for alt in alt_paths:
                    if alt.exists():
                        checkpoint_path = alt
                        break
            
            if checkpoint_path.exists():
                print(f"  Found MobileCLIP checkpoint: {checkpoint_path}")
                self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                    'mobileclip_blt',
                    pretrained=str(checkpoint_path)
                )
                self.tokenizer = mobileclip.get_tokenizer('mobileclip_blt')
                self.model = self.model.to(self.device)
                self.model.eval()
                self.embed_dim = 512
                self.model_name = "MobileCLIP-BLT"
                print(f"  Loaded MobileCLIP-BLT on {self.device}")
                return
            else:
                print(f"  WARNING: MobileCLIP checkpoint not found at {checkpoint_path}")
                raise FileNotFoundError("MobileCLIP checkpoint not found")
                
        except Exception as e:
            print(f"  WARNING: MobileCLIP load failed ({e})")
            print(f"  Falling back to OpenAI CLIP ViT-B/32")
            self._load_clip_fallback()
    
    def _load_clip_fallback(self):
        """CLIP fallback"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.tokenizer = clip.tokenize
            self.model.eval()
            self.embed_dim = 512
            self.model_name = "CLIP-ViT-B/32"
            print(f"  Loaded CLIP ViT-B/32 (fallback) on {self.device}")
        except ImportError:
            print("  ERROR: Neither MobileCLIP nor CLIP available")
            self.model_name = "NONE"
    
    @torch.no_grad()
    def encode_image_crop(self, image: Image.Image, bbox: np.ndarray) -> Optional[np.ndarray]:
        """이미지 crop의 embedding"""
        if self.model is None:
            return None
        
        # Crop with bounds checking
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        
        # 최소 크기 체크
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        crop = image.crop((x1, y1, x2, y2))
        
        # Encoding
        crop_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
        features = self.model.encode_image(crop_tensor)
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트의 embedding"""
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
# Semantic Uncertainty Calculator
# =============================================================================

class SemanticUncertaintyCalculator:
    """
    Semantic Uncertainty 계산기
    
    파이프라인:
    1. bbox crop → image embedding f
    2. 1203개 클래스 이름 → text embeddings → f와 similarity
       → softmax → base posterior p_base
       → top-M 클래스 추출
    3. 각 attribute view k에 대해:
       top-M 클래스의 attribute template → similarity → softmax → p^(k)
    4. JS(p^(1), ..., p^(K)) = u_sem
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
                 temperature: float = 10.0,
                 top_m: int = 20):
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
        """클래스 이름 정리"""
        return name.split("/")[0].strip().replace("_", " ")
    
    def _build_text_embeddings(self):
        """모든 클래스/view의 텍스트 임베딩 생성"""
        print(f"\n  Building text embeddings for {self.num_classes} classes, {self.num_views} views...")
        print(f"  Using: {self.scorer.model_name}")
        
        clean_names = [self._clean_class_name(n) for n in self.sorted_class_names]
        
        # Base embeddings
        base_texts = [self.ATTRIBUTE_VIEWS["base"].format(cls=n) for n in clean_names]
        self.base_embeddings = self.scorer.encode_texts(base_texts)
        
        if self.base_embeddings is not None:
            print(f"    Base embeddings: {self.base_embeddings.shape}")
        
        # View embeddings
        for view_name in self.view_names:
            template = self.ATTRIBUTE_VIEWS[view_name]
            view_texts = [template.format(cls=n) for n in clean_names]
            self.view_embeddings[view_name] = self.scorer.encode_texts(view_texts)
        
        print(f"    View embeddings: {self.num_views} views x {self.num_classes} classes")
        print(f"    Temperature: {self.temperature}, Top-M: {self.top_m}")
    
    def compute_for_detection(self, detection: Detection, 
                               debug: bool = False) -> Tuple[float, Optional[np.ndarray], Dict]:
        """
        단일 detection의 u_sem 계산
        
        Args:
            detection: Detection 객체
            debug: True면 디버그 정보 반환
        
        Returns:
            (u_sem, image_embedding, debug_info)
        """
        debug_info = {"status": "ok", "drop_reason": None}
        
        if detection.image_path is None:
            debug_info = {"status": "dropped", "drop_reason": "no_image_path"}
            return 0.0, None, debug_info
        
        if self.base_embeddings is None:
            debug_info = {"status": "dropped", "drop_reason": "no_embeddings"}
            return 0.0, None, debug_info
        
        try:
            # 1. Image embedding
            img = Image.open(detection.image_path).convert('RGB')
            image_emb = self.scorer.encode_image_crop(img, detection.bbox)
            
            if image_emb is None:
                debug_info = {"status": "dropped", "drop_reason": "invalid_crop"}
                return 0.0, None, debug_info
            
            # 2. Base posterior → Top-M 클래스
            base_logits = np.dot(self.base_embeddings, image_emb) / self.temperature
            base_posterior = softmax(base_logits)
            
            # Top-M 클래스 인덱스
            top_m_indices = np.argsort(base_posterior)[-self.top_m:]
            top_m_probs = base_posterior[top_m_indices]
            
            if debug:
                top_m_names = [self.sorted_class_names[i] for i in top_m_indices[-5:]]
                debug_info["top5_base"] = list(zip(top_m_names, top_m_probs[-5:].tolist()))
            
            # 3. 각 view의 posterior (Top-M만)
            posteriors = []
            view_top5s = {}
            
            for view_name in self.view_names:
                view_emb = self.view_embeddings[view_name]
                
                # Top-M 클래스만
                top_m_emb = view_emb[top_m_indices]  # [M, embed_dim]
                
                # Logits & softmax (Top-M 내에서)
                logits = np.dot(top_m_emb, image_emb) / self.temperature
                posterior = softmax(logits)  # [M]
                posteriors.append(posterior)
                
                if debug:
                    # 이 view에서 top-5 (top_m_indices 내에서의 순위)
                    view_ranking = np.argsort(posterior)[-5:]
                    view_top5s[view_name] = [
                        (self.sorted_class_names[top_m_indices[i]], posterior[i])
                        for i in view_ranking
                    ]
            
            posteriors = np.stack(posteriors, axis=0)  # [K, M]
            
            if debug:
                debug_info["view_top5s"] = view_top5s
            
            # 4. JS divergence
            u_sem = js_divergence(posteriors)
            debug_info["u_sem"] = u_sem
            
            return u_sem, image_emb, debug_info
            
        except Exception as e:
            debug_info = {"status": "dropped", "drop_reason": f"exception: {str(e)}"}
            return 0.0, None, debug_info
    
    def compute_for_detections(self, detections: List[Detection], 
                               verbose: bool = True) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        여러 detection의 u_sem 일괄 계산
        
        Returns:
            (u_sem_array, image_embeddings, stats)
        """
        u_sems = []
        image_embs = []
        
        # Drop 통계
        drop_stats = {
            "total": len(detections),
            "valid": 0,
            "no_image_path": 0,
            "no_embeddings": 0,
            "invalid_crop": 0,
            "exception": 0,
        }
        
        for i, det in enumerate(detections):
            if verbose and (i + 1) % 100 == 0:
                print(f"    Processing {i+1}/{len(detections)}...")
            
            u_sem, emb, debug_info = self.compute_for_detection(det)
            u_sems.append(u_sem)
            image_embs.append(emb)
            
            # 통계
            if debug_info["status"] == "ok":
                drop_stats["valid"] += 1
            else:
                reason = debug_info.get("drop_reason", "unknown")
                if "no_image_path" in reason:
                    drop_stats["no_image_path"] += 1
                elif "invalid_crop" in reason:
                    drop_stats["invalid_crop"] += 1
                elif "no_embeddings" in reason:
                    drop_stats["no_embeddings"] += 1
                else:
                    drop_stats["exception"] += 1
        
        if verbose:
            print(f"    Drop stats: {drop_stats}")
        
        return np.array(u_sems), image_embs, drop_stats
    
    def unit_test_single_detection(self, detection: Detection):
        """
        단일 detection에 대한 유닛테스트 - view 간 차이 확인
        """
        print("\n" + "="*60)
        print("U_SEM UNIT TEST")
        print("="*60)
        
        u_sem, emb, debug_info = self.compute_for_detection(detection, debug=True)
        
        print(f"Detection: {detection.pred_class_name} (conf={detection.confidence:.3f})")
        print(f"Status: {debug_info['status']}")
        
        if debug_info['status'] != 'ok':
            print(f"Drop reason: {debug_info['drop_reason']}")
            return
        
        print(f"\nu_sem = {u_sem:.6f}")
        
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
                all_top1s.append(top5[-1][0])  # top-1 class name
        
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
    
    기존: 클래스 리스트로 depiction 판단 → Depiction_FP = 0 문제
    개선: artifactness score threshold로 판단 → 실제로 depiction 같은 FP 포착
    
    분류:
    - TP: True Positives
    - Depiction_FP: artifactness score > threshold인 Semantic FP (H1 대상)
    - ClassConfusion_FP: 나머지 Semantic FP
    - Background_FP: GT와 매칭 안 됨
    """
    
    # Depiction/Real 프롬프트 (artifactness scorer와 동일)
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
                 scorer: MobileCLIPScorer,
                 depiction_threshold: float = 0.0):
        """
        Args:
            scorer: MobileCLIPScorer (image embedding용)
            depiction_threshold: artifactness score > threshold면 Depiction_FP
                                 (margin 방식: depiction - real, 0 이상이면 depiction 우세)
        """
        self.scorer = scorer
        self.depiction_threshold = depiction_threshold
        
        # Text embeddings
        self.depiction_emb = None
        self.real_emb = None
        self._build_text_embeddings()
    
    def _build_text_embeddings(self):
        """Depiction/Real 텍스트 임베딩 생성"""
        print(f"  Building depiction/real embeddings...")
        self.depiction_emb = self.scorer.encode_texts(self.DEPICTION_PROMPTS)
        self.real_emb = self.scorer.encode_texts(self.REAL_PROMPTS)
        print(f"    Depiction: {self.depiction_emb.shape}, Real: {self.real_emb.shape}")
    
    def compute_artifactness(self, image_emb: np.ndarray) -> float:
        """
        Image embedding의 artifactness score 계산
        
        Returns:
            margin = max(depiction_sim) - max(real_sim)
            양수면 depiction 우세
        """
        if image_emb is None:
            return 0.0
        
        dep_sims = np.dot(self.depiction_emb, image_emb)
        real_sims = np.dot(self.real_emb, image_emb)
        
        return float(np.max(dep_sims) - np.max(real_sims))
    
    def split_detections(self, 
                         detections: List[Detection],
                         image_embs: List[Optional[np.ndarray]]) -> Dict[str, List[Tuple[Detection, float]]]:
        """
        Detection 리스트를 4개 그룹으로 분할 (artifactness score와 함께)
        
        Args:
            detections: Detection 리스트
            image_embs: 각 detection의 image embedding (u_sem 계산에서 재활용)
        
        Returns:
            {group_name: [(Detection, artifactness_score), ...]}
        """
        groups = {
            "TP": [],
            "Depiction_FP": [],
            "ClassConfusion_FP": [],
            "Background_FP": [],
        }
        
        for det, emb in zip(detections, image_embs):
            # Artifactness score
            art_score = self.compute_artifactness(emb) if emb is not None else 0.0
            
            # 분류
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
