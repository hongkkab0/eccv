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
    Artifactness score 계산기
    
    Track B: Visual Primitives
    - s_art = max_j <f, T(q_j)> (depiction 최대 유사도)
    - 또는 margin: s_art = <f, T(real)> - max_j <f, T(depiction)>
    """
    
    def __init__(self,
                 text_model_name: str = "clip:ViT-B/32",
                 device: str = "cuda",
                 method: str = "margin"):
        """
        Args:
            text_model_name: 텍스트 모델 이름
            device: 디바이스
            method: "max" 또는 "margin"
        """
        self.text_model_name = text_model_name
        self.device = device
        self.method = method
        
        self.prompts = ArtifactnessPrompts()
        
        # 임베딩 캐시
        self.depiction_embeddings: Optional[np.ndarray] = None
        self.real_embeddings: Optional[np.ndarray] = None
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """프롬프트 임베딩 초기화"""
        from ultralytics.nn.text_model import build_text_model
        
        text_model = build_text_model(self.text_model_name, device=self.device)
        text_model.eval()
        
        with torch.no_grad():
            # Depiction 임베딩
            dep_tokens = text_model.tokenize(self.prompts.depiction_prompts)
            self.depiction_embeddings = text_model.encode_text(dep_tokens).cpu().numpy()
            
            # Real 임베딩
            real_tokens = text_model.tokenize(self.prompts.real_prompts)
            self.real_embeddings = text_model.encode_text(real_tokens).cpu().numpy()
    
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
    
    def compute_for_detection(self, detection: Detection) -> float:
        """Detection의 artifactness score 계산"""
        if detection.region_feature is None:
            return 0.0
        return self.compute_score(detection.region_feature)
    
    def compute_for_detections(self, detections: List[Detection]) -> np.ndarray:
        """여러 detection의 artifactness score 일괄 계산"""
        scores = []
        for det in detections:
            scores.append(self.compute_for_detection(det))
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
