"""
Artifactness Score Calculation (Track B)
=========================================
Visual Primitives → Artifactness score

"is it a toy?", "is it a statue?" 등의 판별 질문을 사용한
depiction/replica 탐지 점수
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

from .detection_logger import Detection


@dataclass
class ArtifactnessPrompts:
    """Artifactness 판별용 프롬프트"""
    
    # Depiction 관련 프롬프트
    depiction_prompts: List[str] = None
    
    # Real object 프롬프트
    real_prompts: List[str] = None
    
    def __post_init__(self):
        if self.depiction_prompts is None:
            self.depiction_prompts = [
                "a toy",
                "a toy version",
                "a statue",
                "a sculpture", 
                "a poster",
                "a painting",
                "a drawing",
                "a figurine",
                "a doll",
                "a puppet",
                "a cartoon character",
                "a stuffed animal",
                "a plush toy",
                "a mannequin",
                "a model replica",
                "a printed image",
                "a 2D depiction",
                "an artificial representation",
            ]
        
        if self.real_prompts is None:
            self.real_prompts = [
                "a real photo of the object",
                "a real object",
                "a living thing",
                "an actual object",
                "a genuine item",
                "a real animal",
                "a real person",
                "a real vehicle",
            ]


class ArtifactnessScorer:
    """
    Artifactness score 계산기 (CLIP 기반)
    
    CLIP으로 bbox crop embedding과 depiction/real prompt의 similarity 비교
    - s_art = max_j <f, T(depiction_j)> - mean(<f, T(real)>)
    
    SemanticUncertaintyCalculator와 같은 CLIP 모델 사용
    """
    
    def __init__(self,
                 device: str = "cuda",
                 method: str = "margin"):
        """
        Args:
            device: 디바이스
            method: "max" 또는 "margin"
        """
        self.device = device
        self.method = method
        
        self.prompts = ArtifactnessPrompts()
        
        # CLIP 모델 로드
        self.clip_model = None
        self.clip_preprocess = None
        
        # 임베딩 캐시
        self.depiction_embeddings: Optional[np.ndarray] = None
        self.real_embeddings: Optional[np.ndarray] = None
        
        self._initialize_clip()
    
    def _initialize_clip(self):
        """CLIP 모델 및 임베딩 초기화"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print(f"  Loaded CLIP ViT-B/32 for artifactness scoring")
            
            # Text embeddings 생성
            with torch.no_grad():
                # Depiction 임베딩
                dep_tokens = clip.tokenize(self.prompts.depiction_prompts).to(self.device)
                dep_feats = self.clip_model.encode_text(dep_tokens)
                dep_feats = dep_feats / dep_feats.norm(dim=-1, keepdim=True)
                self.depiction_embeddings = dep_feats.cpu().numpy()
                
                # Real 임베딩
                real_tokens = clip.tokenize(self.prompts.real_prompts).to(self.device)
                real_feats = self.clip_model.encode_text(real_tokens)
                real_feats = real_feats / real_feats.norm(dim=-1, keepdim=True)
                self.real_embeddings = real_feats.cpu().numpy()
            
            print(f"  Built {len(self.prompts.depiction_prompts)} depiction + {len(self.prompts.real_prompts)} real embeddings")
            
        except ImportError:
            print("  WARNING: clip not installed")
    
    def compute_score(self, region_feature: np.ndarray) -> float:
        """
        단일 region feature의 artifactness score 계산
        
        Args:
            region_feature: [embed_dim]
        
        Returns:
            Artifactness score
        """
        # 정규화
        f_norm = region_feature / (np.linalg.norm(region_feature) + 1e-10)
        
        # Depiction 유사도
        dep_sims = np.dot(self.depiction_embeddings, f_norm)
        max_dep_sim = np.max(dep_sims)
        
        if self.method == "max":
            # s_art = max_j <f, T(depiction_j)>
            return float(max_dep_sim)
        
        elif self.method == "margin":
            # s_art = max_j <f, T(depiction_j)> - max_k <f, T(real_k)>
            # 높을수록 depiction일 가능성 높음
            real_sims = np.dot(self.real_embeddings, f_norm)
            max_real_sim = np.max(real_sims)
            return float(max_dep_sim - max_real_sim)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_detailed_scores(self, region_feature: np.ndarray) -> Dict:
        """
        상세 점수 계산 (분석용)
        
        Returns:
            각 프롬프트별 유사도 딕셔너리
        """
        f_norm = region_feature / (np.linalg.norm(region_feature) + 1e-10)
        
        dep_sims = np.dot(self.depiction_embeddings, f_norm)
        real_sims = np.dot(self.real_embeddings, f_norm)
        
        return {
            "depiction_scores": {
                prompt: float(sim) 
                for prompt, sim in zip(self.prompts.depiction_prompts, dep_sims)
            },
            "real_scores": {
                prompt: float(sim)
                for prompt, sim in zip(self.prompts.real_prompts, real_sims)
            },
            "max_depiction": float(np.max(dep_sims)),
            "max_real": float(np.max(real_sims)),
            "margin": float(np.max(dep_sims) - np.max(real_sims)),
        }
    
    def compute_for_detection(self, detection: Detection, image_emb: np.ndarray = None) -> float:
        """
        Detection의 artifactness score 계산
        
        Args:
            detection: Detection 객체
            image_emb: 미리 계산된 CLIP image embedding (없으면 직접 계산)
        """
        # image_emb가 제공되면 사용
        if image_emb is not None:
            return self.compute_score(image_emb)
        
        # 없으면 직접 CLIP encoding
        if detection.image_path is None or self.clip_model is None:
            return 0.0
        
        try:
            from PIL import Image
            img = Image.open(detection.image_path).convert('RGB')
            
            # Crop
            x1, y1, x2, y2 = map(int, detection.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            crop = img.crop((x1, y1, x2, y2))
            
            # CLIP encoding
            with torch.no_grad():
                crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
                features = self.clip_model.encode_image(crop_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                image_emb = features.cpu().numpy().squeeze()
            
            return self.compute_score(image_emb)
            
        except Exception as e:
            return 0.0
    
    def compute_for_detections(self, detections: List[Detection], 
                                image_embs: List[np.ndarray] = None) -> np.ndarray:
        """
        여러 detection의 artifactness score 일괄 계산
        
        Args:
            detections: Detection 리스트
            image_embs: 미리 계산된 CLIP image embeddings (없으면 직접 계산)
        """
        scores = []
        for i, det in enumerate(detections):
            emb = image_embs[i] if image_embs is not None else None
            scores.append(self.compute_for_detection(det, emb))
        return np.array(scores)
    
    def compute_for_triad_split(self,
                                triad_split: Dict[str, List[Detection]]) -> Dict[str, np.ndarray]:
        """Triad split 각 그룹의 artifactness score 계산"""
        return {
            group: self.compute_for_detections(dets)
            for group, dets in triad_split.items()
        }


class CombinedErrorScorer:
    """
    최종 error probability 계산
    
    u = σ(w1 * u_sem + w2 * s_art + w3 * u_ret + w4 * u_loc)
    
    현재는 u_sem + s_art 조합만 구현
    """
    
    def __init__(self,
                 w_sem: float = 1.0,
                 w_art: float = 1.0,
                 normalize: bool = True):
        """
        Args:
            w_sem: u_sem 가중치
            w_art: s_art 가중치
            normalize: 점수 정규화 여부
        """
        self.w_sem = w_sem
        self.w_art = w_art
        self.normalize = normalize
    
    def compute_combined_score(self,
                               u_sem: float,
                               s_art: float,
                               u_sem_stats: Optional[Dict] = None,
                               s_art_stats: Optional[Dict] = None) -> float:
        """
        결합 점수 계산
        
        Args:
            u_sem: Semantic uncertainty
            s_art: Artifactness score
            u_sem_stats: u_sem 정규화용 통계 (mean, std)
            s_art_stats: s_art 정규화용 통계
        
        Returns:
            결합 점수
        """
        if self.normalize:
            if u_sem_stats:
                u_sem = (u_sem - u_sem_stats.get("mean", 0)) / (u_sem_stats.get("std", 1) + 1e-10)
            if s_art_stats:
                s_art = (s_art - s_art_stats.get("mean", 0)) / (s_art_stats.get("std", 1) + 1e-10)
        
        # 선형 조합
        combined = self.w_sem * u_sem + self.w_art * s_art
        
        # Sigmoid로 확률화
        prob = 1.0 / (1.0 + np.exp(-combined))
        
        return prob
    
    def compute_for_arrays(self,
                           u_sem_array: np.ndarray,
                           s_art_array: np.ndarray) -> np.ndarray:
        """배열 단위 계산"""
        # 정규화
        if self.normalize:
            u_sem_array = (u_sem_array - np.mean(u_sem_array)) / (np.std(u_sem_array) + 1e-10)
            s_art_array = (s_art_array - np.mean(s_art_array)) / (np.std(s_art_array) + 1e-10)
        
        combined = self.w_sem * u_sem_array + self.w_art * s_art_array
        probs = 1.0 / (1.0 + np.exp(-combined))
        
        return probs


def analyze_artifactness_statistics(s_art_by_group: Dict[str, np.ndarray]) -> Dict:
    """Artifactness score 통계 분석"""
    stats = {}
    
    for group, values in s_art_by_group.items():
        if len(values) == 0:
            continue
        
        stats[group] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    
    return stats
