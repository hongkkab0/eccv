"""
H1 Semantic FP Experiment Configuration
=======================================
Fig.1 주장을 증거로 만드는 핵심 실험 설정
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """실험 전체 설정"""
    
    # === A-1: 데이터셋/추론 세팅 ===
    dataset: str = "lvis"  # LVIS v1 val
    data_yaml: str = "lvis.yaml"
    split: str = "val"  # val 전체
    
    # Detector 설정
    checkpoint: str = "yoloe-v8l-seg.pt"  # YOLO-E checkpoint
    device: str = "cuda:0"
    batch_size: int = 1
    imgsz: int = 640
    
    # Detection match 설정
    iou_threshold: float = 0.5  # TP/FP 판정 IoU 기준
    conf_threshold: float = 0.001  # 낮은 confidence도 수집
    
    # === A-2: Triad Split 설정 ===
    semantic_fp_iou_threshold: float = 0.3  # Semantic FP 판정 IoU
    
    # === A-3: Confidence-matched 평가 설정 ===
    conf_bin_start: float = 0.80
    conf_bin_end: float = 0.95
    conf_bin_step: float = 0.02
    samples_per_bin: int = 50  # 각 bin에서 그룹별 샘플링 수
    
    # === A-4: H1 검증 설정 ===
    top_m_classes: int = 10  # Top-M 클래스 게이팅
    num_attribute_views: int = 5  # K개의 attribute view
    text_model: str = "clip:ViT-B/32"  # 텍스트 임베딩 모델 (MobileCLIP 의존성 제거)
    
    # === 출력 설정 ===
    output_dir: str = "experiments/h1_semantic_fp/outputs"
    save_detection_logs: bool = True
    save_visualizations: bool = True
    
    # === 기타 ===
    seed: int = 42
    num_workers: int = 4


@dataclass
class ConfounderConfig:
    """
    Confounder 클래스 설정 (Semantic FP 자동 분류용)
    
    LVIS 클래스명에서 키워드로 confounder set을 정의
    이 클래스들은 depiction/replica 계열로, 실제 객체의 "모형"이나 "그림"을 나타냄
    """
    
    # 키워드 기반 confounder 정의
    confounder_keywords: List[str] = field(default_factory=lambda: [
        # 3D 모형류
        "statue", "sculpture", "figurine", "doll", "toy", "plush", 
        "mannequin", "puppet", "marionette", "teddy",
        
        # 마스크/가면류
        "mask",
        
        # 2D 표현류
        "poster", "sign", "logo", "emblem", "drawing", "painting", 
        "print", "cartoon", "graffiti", "relief",
        
        # 이미지/사진류
        "picture", "photo", "image", "portrait",
        
        # 장식류 (동물/사람 형상)
        "ornament", "decoration", "carving",
        
        # 반사/영상류
        "reflection", "shadow",
    ])
    
    # LVIS 1203 클래스 중 confounder로 명시적으로 지정할 클래스 인덱스
    # (키워드 매칭 외에 추가로 지정)
    explicit_confounder_indices: List[int] = field(default_factory=lambda: [
        379,   # doll
        436,   # figurine
        834,   # poster/placard
        747,   # painting
        927,   # sculpture
        1007,  # statue/statue sculpture
        1109,  # toy
        1070,  # teddy bear
        855,   # puppet/marionette
        674,   # mask/facemask
    ])


@dataclass  
class AttributeViewConfig:
    """
    Attribute View 설정 (Track A: Hierarchical Attributes)
    
    u_sem = JS(p^(1), ..., p^(K)) 계산을 위한 설정
    """
    
    # 슈퍼클래스 정의 (LLM 호출 대상)
    superclasses: List[str] = field(default_factory=lambda: [
        "animal", "vehicle", "tool", "furniture", "food", "clothing",
        "electronics", "container", "sports_equipment", "musical_instrument",
        "kitchen_item", "office_item", "outdoor_item", "building_part",
        "body_part", "plant", "weapon", "toy", "art", "text_sign"
    ])
    
    # 각 슈퍼클래스별 attribute view 템플릿
    # K=5 views: material, shape, texture, function, context
    view_templates: Dict[str, List[str]] = field(default_factory=lambda: {
        "material": [
            "a {cls} made of metal",
            "a {cls} made of plastic", 
            "a {cls} made of wood",
            "a {cls} made of fabric",
            "a {cls} made of glass",
        ],
        "texture": [
            "a smooth {cls}",
            "a rough {cls}",
            "a shiny {cls}",
            "a matte {cls}",
            "a textured {cls}",
        ],
        "shape": [
            "a round {cls}",
            "a rectangular {cls}",
            "a elongated {cls}",
            "a compact {cls}",
            "a irregular shaped {cls}",
        ],
        "context": [
            "a {cls} indoors",
            "a {cls} outdoors",
            "a {cls} in natural setting",
            "a {cls} in urban setting",
            "a {cls} in domestic setting",
        ],
        "state": [
            "a new {cls}",
            "a used {cls}",
            "a clean {cls}",
            "a dirty {cls}",
            "a damaged {cls}",
        ],
    })
    
    # 동의어 paraphrase (대조군용)
    paraphrase_templates: List[str] = field(default_factory=lambda: [
        "a photo of a {cls}",
        "an image of a {cls}",
        "a picture of a {cls}",
        "a {cls}",
        "one {cls}",
    ])


@dataclass
class ArtifactnessConfig:
    """
    Track B: Visual Primitives → Artifactness score
    
    "is it a toy?", "is it a statue?" 등의 판별 질문 설정
    """
    
    # Depiction 판별 프롬프트
    depiction_prompts: List[str] = field(default_factory=lambda: [
        "a toy",
        "a statue",
        "a poster",
        "a painting",
        "a drawing",
        "a figurine",
        "a sculpture",
        "a doll",
        "a puppet",
        "a cartoon character",
        "a stuffed animal",
        "a plush toy",
        "a mannequin",
        "a model replica",
        "a printed image",
    ])
    
    # Real 판별 프롬프트
    real_prompts: List[str] = field(default_factory=lambda: [
        "a real photo",
        "a real object",
        "a living thing",
        "an actual object",
        "a genuine item",
    ])
    
    # Artifactness score 계산 방식
    # "max": s_art = max_j <f, T(q_j)>
    # "margin": s_art = <f, T(real)> - max_j <f, T(depiction)>
    score_method: str = "margin"


# LVIS 1203 클래스 이름 (lvis.yaml에서 추출)
LVIS_CLASS_NAMES = None  # 런타임에 로드


def get_default_config() -> ExperimentConfig:
    """기본 실험 설정 반환"""
    return ExperimentConfig()


def get_confounder_config() -> ConfounderConfig:
    """Confounder 설정 반환"""
    return ConfounderConfig()


def get_attribute_view_config() -> AttributeViewConfig:
    """Attribute view 설정 반환"""
    return AttributeViewConfig()


def get_artifactness_config() -> ArtifactnessConfig:
    """Artifactness 설정 반환"""
    return ArtifactnessConfig()
