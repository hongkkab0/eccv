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

from .detection_logger import Detection
from .attribute_embeddings import AttributeEmbeddingCache, get_view_embeddings_for_classes


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
    
    Detection 리스트에 대해 u_sem을 일괄 계산
    """
    
    def __init__(self,
                 attribute_cache: AttributeEmbeddingCache,
                 class_names: Dict[int, str],
                 top_m: int = 10,
                 temperature: float = 1.0):
        """
        Args:
            attribute_cache: Attribute embedding 캐시
            class_names: 클래스 이름 딕셔너리
            top_m: Top-M 클래스 게이팅
            temperature: softmax temperature
        """
        self.attribute_cache = attribute_cache
        self.class_names = class_names
        self.top_m = top_m
        self.temperature = temperature
        
        # 전체 클래스의 view embeddings 준비
        self.all_class_indices = sorted(class_names.keys())
        self.view_embeddings = get_view_embeddings_for_classes(
            attribute_cache, self.all_class_indices
        )  # [num_classes, K, embed_dim]
        
        # 인덱스 매핑
        self.idx_to_pos = {idx: pos for pos, idx in enumerate(self.all_class_indices)}
    
    def compute_for_detection(self, detection: Detection, use_gating: bool = True) -> float:
        """
        단일 detection의 u_sem 계산
        
        Args:
            detection: Detection 객체
            use_gating: Top-M 게이팅 사용 여부
        
        Returns:
            u_sem 값
        """
        if detection.region_feature is None:
            return 0.0
        
        if use_gating and detection.top_k_classes is not None:
            # Top-M 게이팅
            top_m_classes = detection.top_k_classes[:self.top_m]
            # 인덱스 변환 (클래스 인덱스 -> 배열 위치)
            top_m_positions = [self.idx_to_pos.get(c, 0) for c in top_m_classes]
            return compute_u_sem_gated(
                detection.region_feature,
                self.view_embeddings,
                np.array(top_m_positions),
                self.temperature
            )
        else:
            # 전체 클래스 대상
            return compute_u_sem(
                detection.region_feature,
                self.view_embeddings,
                self.temperature
            )
    
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
        
        stats[group] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
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
