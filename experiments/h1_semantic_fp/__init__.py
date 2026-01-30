"""
H1 Semantic FP Experiment
==========================
Fig.1 주장을 증거로 만드는 핵심 실험

Hypothesis H1: Semantic FP는 attribute violation이 높다
→ u_sem (JS divergence)이 TP와 Semantic FP를 구분할 수 있다

실행:
    python -m experiments.h1_semantic_fp.run_experiment --checkpoint yoloe-v8l-seg.pt

모듈 구성:
- config.py: 실험 설정
- confounder_classes.py: LVIS confounder 클래스 정의
- detection_logger.py: Detection 로깅 및 Triad Split
- confidence_matching.py: Confidence-matched 샘플링
- attribute_embeddings.py: Attribute view 임베딩 생성
- semantic_uncertainty.py: u_sem (JS divergence) 계산
- artifactness_score.py: Track B artifactness score
- h1_metrics.py: H1 검증 메트릭
- visualize.py: 시각화 도구
- run_experiment.py: 메인 실행 스크립트
"""

from .config import (
    ExperimentConfig,
    ConfounderConfig,
    AttributeViewConfig,
    ArtifactnessConfig,
    get_default_config,
    get_confounder_config,
    get_attribute_view_config,
    get_artifactness_config,
)

from .confounder_classes import (
    CONFOUNDER_KEYWORDS,
    EXPLICIT_CONFOUNDER_INDICES,
    load_lvis_class_names,
    build_confounder_set,
    get_confounder_class_info,
    is_confounder_class,
    analyze_confounder_coverage,
)

from .detection_logger import (
    Detection,
    ImageGT,
    DetectionLogger,
)

from .confidence_matching import (
    ConfidenceMatchedSample,
    ConfidenceMatchedSampler,
    create_confidence_matched_dataset,
)

from .attribute_embeddings import (
    AttributeViewSet,
    AttributeEmbeddingCache,
    AttributeEmbeddingGenerator,
    get_view_embeddings_for_class,
    get_view_embeddings_for_classes,
)

from .semantic_uncertainty import (
    js_divergence,
    compute_view_posterior,
    compute_u_sem,
    compute_u_sem_gated,
    SemanticUncertaintyCalculator,
    analyze_u_sem_statistics,
)

from .artifactness_score import (
    ArtifactnessPrompts,
    ArtifactnessScorer,
    CombinedErrorScorer,
    analyze_artifactness_statistics,
)

from .h1_metrics import (
    H1VerificationResult,
    compute_auroc_aupr,
    compute_cohens_d,
    compute_spearman_correlation,
    H1Evaluator,
    format_h1_results,
    run_full_h1_evaluation,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "ExperimentConfig",
    "ConfounderConfig", 
    "AttributeViewConfig",
    "ArtifactnessConfig",
    "get_default_config",
    
    # Confounder
    "CONFOUNDER_KEYWORDS",
    "load_lvis_class_names",
    "build_confounder_set",
    
    # Detection
    "Detection",
    "DetectionLogger",
    
    # Confidence Matching
    "ConfidenceMatchedSampler",
    "create_confidence_matched_dataset",
    
    # Embeddings
    "AttributeEmbeddingGenerator",
    "AttributeEmbeddingCache",
    
    # Semantic Uncertainty
    "js_divergence",
    "compute_u_sem",
    "SemanticUncertaintyCalculator",
    
    # Artifactness
    "ArtifactnessScorer",
    "CombinedErrorScorer",
    
    # H1 Metrics
    "H1Evaluator",
    "H1VerificationResult",
    "format_h1_results",
]
