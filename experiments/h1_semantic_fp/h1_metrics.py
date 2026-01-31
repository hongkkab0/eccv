"""
H1 Verification Metrics
========================
H1 검증을 위한 메트릭 계산

- AUROC / AUPR: u_sem의 변별력 측정
- Spearman ρ: u_sem과 human violation count의 상관관계
- Cohen's d: 그룹 간 효과 크기
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve,
    precision_recall_curve,
)
from dataclasses import dataclass


@dataclass
class H1VerificationResult:
    """H1 검증 결과"""
    
    # 변별력 메트릭 (TP vs Semantic FP)
    auroc_u_sem: float
    aupr_u_sem: float
    auroc_confidence: float  # 대조군: confidence
    
    # Effect size
    cohens_d_u_sem: float
    cohens_d_confidence: float
    
    # 상관관계 (human annotation 있는 경우)
    spearman_rho_u_sem: Optional[float] = None
    spearman_p_value: Optional[float] = None
    
    # Paraphrase 대조군
    spearman_rho_paraphrase: Optional[float] = None
    
    # 추가 정보
    n_tp: int = 0
    n_semantic_fp: int = 0


def compute_auroc_aupr(scores_positive: np.ndarray,
                       scores_negative: np.ndarray) -> Tuple[float, float]:
    """
    AUROC와 AUPR 계산
    
    Args:
        scores_positive: positive 클래스 (Semantic FP) 점수
        scores_negative: negative 클래스 (TP) 점수
    
    Returns:
        (AUROC, AUPR)
    """
    # 라벨 생성: positive=1 (Semantic FP), negative=0 (TP)
    y_true = np.concatenate([
        np.ones(len(scores_positive)),
        np.zeros(len(scores_negative))
    ])
    y_scores = np.concatenate([scores_positive, scores_negative])
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5
    
    try:
        aupr = average_precision_score(y_true, y_scores)
    except ValueError:
        aupr = 0.5
    
    return auroc, aupr


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d 계산 (효과 크기)
    
    Args:
        group1: 첫 번째 그룹 값
        group2: 두 번째 그룹 값
    
    Returns:
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_spearman_correlation(scores: np.ndarray,
                                 violations: np.ndarray) -> Tuple[float, float]:
    """
    Spearman 상관계수 계산
    
    Args:
        scores: u_sem 점수
        violations: human violation count
    
    Returns:
        (rho, p-value)
    """
    if len(scores) < 3:
        return 0.0, 1.0
    
    rho, p_value = stats.spearmanr(scores, violations)
    return rho, p_value


class H1Evaluator:
    """
    H1 가설 검증 평가기
    
    H1: Semantic FP는 attribute violation이 높다
    → u_sem이 TP와 Semantic FP를 잘 구분해야 함
    """
    
    def __init__(self):
        pass
    
    def evaluate(self,
                 u_sem_tp: np.ndarray,
                 u_sem_semantic_fp: np.ndarray,
                 confidence_tp: np.ndarray,
                 confidence_semantic_fp: np.ndarray,
                 u_sem_with_violations: Optional[np.ndarray] = None,
                 violations: Optional[np.ndarray] = None,
                 paraphrase_disagreement: Optional[np.ndarray] = None) -> H1VerificationResult:
        """
        H1 검증 수행
        
        Args:
            u_sem_tp: TP의 u_sem 값
            u_sem_semantic_fp: Semantic FP의 u_sem 값
            confidence_tp: TP의 confidence
            confidence_semantic_fp: Semantic FP의 confidence
            u_sem_with_violations: violation annotation이 있는 샘플의 u_sem
            violations: human violation count
            paraphrase_disagreement: 대조군 (paraphrase JS)
        
        Returns:
            H1VerificationResult
        """
        # 빈 배열 체크
        if len(u_sem_tp) == 0 or len(u_sem_semantic_fp) == 0:
            print("WARNING: Empty arrays - cannot compute metrics")
            return H1VerificationResult(
                auroc_u_sem=0.5,
                aupr_u_sem=0.5,
                auroc_confidence=0.5,
                cohens_d_u_sem=0.0,
                cohens_d_confidence=0.0,
                n_tp=len(u_sem_tp),
                n_semantic_fp=len(u_sem_semantic_fp),
            )
        
        # 1. AUROC/AUPR for u_sem (Semantic FP를 positive로)
        auroc_u_sem, aupr_u_sem = compute_auroc_aupr(
            u_sem_semantic_fp, u_sem_tp
        )
        
        # 2. AUROC for confidence (대조군 - 낮을 것으로 예상)
        # Confidence는 높을수록 TP일 가능성이 높으므로 부호 반전 필요 없음
        # 하지만 "Semantic FP 검출" 관점에서는 confidence가 낮아야 positive
        # → 실제로 confidence-matched이면 ~0.5가 나와야 함
        auroc_conf, _ = compute_auroc_aupr(
            1 - confidence_semantic_fp,  # 반전: 낮은 confidence = Semantic FP?
            1 - confidence_tp
        )
        
        # 3. Cohen's d
        cohens_d_u_sem = compute_cohens_d(u_sem_semantic_fp, u_sem_tp)
        cohens_d_conf = compute_cohens_d(
            1 - confidence_semantic_fp, 
            1 - confidence_tp
        )
        
        result = H1VerificationResult(
            auroc_u_sem=auroc_u_sem,
            aupr_u_sem=aupr_u_sem,
            auroc_confidence=auroc_conf,
            cohens_d_u_sem=cohens_d_u_sem,
            cohens_d_confidence=cohens_d_conf,
            n_tp=len(u_sem_tp),
            n_semantic_fp=len(u_sem_semantic_fp),
        )
        
        # 4. Spearman correlation (human annotation 있는 경우)
        if u_sem_with_violations is not None and violations is not None:
            rho, p_value = compute_spearman_correlation(u_sem_with_violations, violations)
            result.spearman_rho_u_sem = rho
            result.spearman_p_value = p_value
        
        # 5. Paraphrase 대조군 (있는 경우)
        if paraphrase_disagreement is not None and violations is not None:
            rho_para, _ = compute_spearman_correlation(paraphrase_disagreement, violations)
            result.spearman_rho_paraphrase = rho_para
        
        return result
    
    def get_roc_curve_data(self,
                           u_sem_tp: np.ndarray,
                           u_sem_semantic_fp: np.ndarray) -> Dict:
        """ROC curve 데이터 반환 (시각화용)"""
        y_true = np.concatenate([
            np.ones(len(u_sem_semantic_fp)),
            np.zeros(len(u_sem_tp))
        ])
        y_scores = np.concatenate([u_sem_semantic_fp, u_sem_tp])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
    
    def get_pr_curve_data(self,
                          u_sem_tp: np.ndarray,
                          u_sem_semantic_fp: np.ndarray) -> Dict:
        """PR curve 데이터 반환 (시각화용)"""
        y_true = np.concatenate([
            np.ones(len(u_sem_semantic_fp)),
            np.zeros(len(u_sem_tp))
        ])
        y_scores = np.concatenate([u_sem_semantic_fp, u_sem_tp])
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }


def format_h1_results(result: H1VerificationResult) -> str:
    """H1 검증 결과 포맷팅"""
    lines = [
        "=" * 50,
        "H1 Verification Results",
        "=" * 50,
        "",
        f"Sample sizes: TP={result.n_tp}, Semantic FP={result.n_semantic_fp}",
        "",
        "--- Discriminative Power (TP vs Semantic FP) ---",
        f"u_sem AUROC:      {result.auroc_u_sem:.4f}",
        f"u_sem AUPR:       {result.aupr_u_sem:.4f}",
        f"Confidence AUROC: {result.auroc_confidence:.4f} (baseline, should be ~0.5)",
        "",
        "--- Effect Size ---",
        f"u_sem Cohen's d:      {result.cohens_d_u_sem:.4f}",
        f"Confidence Cohen's d: {result.cohens_d_confidence:.4f}",
        "",
    ]
    
    if result.spearman_rho_u_sem is not None:
        lines.extend([
            "--- Correlation with Human Violations ---",
            f"u_sem Spearman ρ:      {result.spearman_rho_u_sem:.4f} (p={result.spearman_p_value:.4e})",
        ])
        
        if result.spearman_rho_paraphrase is not None:
            lines.append(
                f"Paraphrase Spearman ρ: {result.spearman_rho_paraphrase:.4f} (baseline, should be lower)"
            )
    
    lines.extend([
        "",
        "--- Interpretation ---",
    ])
    
    # 해석
    if result.auroc_u_sem > 0.7:
        lines.append("✓ u_sem shows good discriminative power (AUROC > 0.7)")
    elif result.auroc_u_sem > 0.6:
        lines.append("△ u_sem shows moderate discriminative power (0.6 < AUROC ≤ 0.7)")
    else:
        lines.append("✗ u_sem shows weak discriminative power (AUROC ≤ 0.6)")
    
    if abs(result.auroc_confidence - 0.5) < 0.05:
        lines.append("✓ Confidence is near chance (AUROC ~0.5) as expected with matching")
    else:
        lines.append("! Confidence deviates from chance - check matching quality")
    
    if result.cohens_d_u_sem > 0.8:
        lines.append("✓ Large effect size (d > 0.8)")
    elif result.cohens_d_u_sem > 0.5:
        lines.append("△ Medium effect size (0.5 < d ≤ 0.8)")
    else:
        lines.append("✗ Small effect size (d ≤ 0.5)")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def run_full_h1_evaluation(triad_split: Dict[str, List],
                           u_sem_by_group: Dict[str, np.ndarray],
                           verbose: bool = True) -> H1VerificationResult:
    """
    전체 H1 평가 실행 (convenience function)
    
    Args:
        triad_split: {"TP": [Detection, ...], "Semantic_FP": [...], ...}
        u_sem_by_group: {"TP": np.array, "Semantic_FP": np.array, ...}
        verbose: 결과 출력 여부
    
    Returns:
        H1VerificationResult
    """
    # Confidence 추출
    conf_tp = np.array([d.confidence for d in triad_split["TP"]])
    conf_sem_fp = np.array([d.confidence for d in triad_split["Semantic_FP"]])
    
    # 평가
    evaluator = H1Evaluator()
    result = evaluator.evaluate(
        u_sem_tp=u_sem_by_group["TP"],
        u_sem_semantic_fp=u_sem_by_group["Semantic_FP"],
        confidence_tp=conf_tp,
        confidence_semantic_fp=conf_sem_fp,
    )
    
    if verbose:
        print(format_h1_results(result))
    
    return result
