"""
LVIS Confounder Classes Definition
===================================
Semantic FP 자동 분류를 위한 confounder 클래스 정의

Confounder set: depiction/replica/print 계열 클래스
- 이 클래스들의 GT 박스와 IoU >= 0.3인 FP는 Semantic FP로 분류
"""

import re
from typing import List, Dict, Set, Tuple
from pathlib import Path
import yaml


# Confounder 키워드 (클래스명에서 매칭)
CONFOUNDER_KEYWORDS = [
    # === 3D 모형류 ===
    "statue", "sculpture", "figurine", "doll", "toy", "plush",
    "mannequin", "puppet", "marionette", "teddy",
    "snowman",  # 눈사람도 depiction
    
    # === 마스크/가면류 ===
    "mask",
    
    # === 2D 표현류 ===
    "poster", "sign", "logo", "emblem", "drawing", "painting",
    "print", "cartoon", "graffiti", "relief",
    "picture", "photo", "image", "portrait",
    "billboard", "banner",
    
    # === 장식류 ===
    "ornament", "decoration", "carving",
    "scarecrow",  # 허수아비
    
    # === 기타 ===
    "rag doll",
]

# LVIS 클래스 중 명시적 confounder 인덱스 (확인된 것들)
EXPLICIT_CONFOUNDER_INDICES = {
    379: "doll",
    436: "figurine", 
    834: "poster/placard",
    747: "painting",
    927: "sculpture",
    1007: "statue/statue sculpture",
    1109: "toy",
    1070: "teddy bear",
    855: "puppet/marionette",
    674: "mask/facemask",
    976: "snowman",
    868: "rag doll",
    919: "scarecrow/strawman",
    95: "billboard",
    49: "banner/streamer",
    381: "dollhouse/doll's house",
}


def load_lvis_class_names(lvis_yaml_path: str = None) -> Dict[int, str]:
    """LVIS 클래스 이름 로드"""
    if lvis_yaml_path is None:
        # 기본 경로
        lvis_yaml_path = Path(__file__).parent.parent.parent / "ultralytics/cfg/datasets/lvis.yaml"
    
    with open(lvis_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data.get('names', {})


def build_confounder_set(class_names: Dict[int, str], 
                         keywords: List[str] = None,
                         explicit_indices: Dict[int, str] = None) -> Set[int]:
    """
    Confounder 클래스 인덱스 집합 생성
    
    Args:
        class_names: {index: class_name} 딕셔너리
        keywords: 매칭할 키워드 리스트
        explicit_indices: 명시적으로 추가할 인덱스
    
    Returns:
        confounder 클래스 인덱스 집합
    """
    if keywords is None:
        keywords = CONFOUNDER_KEYWORDS
    if explicit_indices is None:
        explicit_indices = EXPLICIT_CONFOUNDER_INDICES
    
    confounder_indices = set()
    
    # 1. 키워드 기반 매칭
    for idx, name in class_names.items():
        name_lower = name.lower()
        for keyword in keywords:
            # 단어 경계를 고려한 매칭
            if re.search(rf'\b{re.escape(keyword.lower())}\b', name_lower):
                confounder_indices.add(idx)
                break
    
    # 2. 명시적 인덱스 추가
    for idx in explicit_indices.keys():
        if idx in class_names:
            confounder_indices.add(idx)
    
    return confounder_indices


def get_confounder_class_info(class_names: Dict[int, str]) -> Tuple[Set[int], Dict[int, str]]:
    """
    Confounder 클래스 정보 반환
    
    Returns:
        (confounder_indices, confounder_names) 튜플
    """
    confounder_indices = build_confounder_set(class_names)
    confounder_names = {idx: class_names[idx] for idx in confounder_indices}
    return confounder_indices, confounder_names


def is_confounder_class(class_idx: int, confounder_set: Set[int]) -> bool:
    """클래스가 confounder인지 확인"""
    return class_idx in confounder_set


def analyze_confounder_coverage(class_names: Dict[int, str]) -> Dict:
    """
    Confounder 분석 리포트 생성
    
    Returns:
        분석 결과 딕셔너리
    """
    confounder_indices, confounder_names = get_confounder_class_info(class_names)
    
    # 키워드별 매칭 통계
    keyword_matches = {}
    for keyword in CONFOUNDER_KEYWORDS:
        matches = []
        for idx, name in class_names.items():
            if re.search(rf'\b{re.escape(keyword.lower())}\b', name.lower()):
                matches.append((idx, name))
        if matches:
            keyword_matches[keyword] = matches
    
    return {
        "total_classes": len(class_names),
        "confounder_count": len(confounder_indices),
        "confounder_ratio": len(confounder_indices) / len(class_names),
        "confounder_indices": sorted(confounder_indices),
        "confounder_names": confounder_names,
        "keyword_matches": keyword_matches,
    }


if __name__ == "__main__":
    # 테스트: confounder 클래스 분석
    class_names = load_lvis_class_names()
    report = analyze_confounder_coverage(class_names)
    
    print(f"=== LVIS Confounder Analysis ===")
    print(f"Total classes: {report['total_classes']}")
    print(f"Confounder classes: {report['confounder_count']} ({report['confounder_ratio']*100:.1f}%)")
    print(f"\nConfounder class names:")
    for idx in sorted(report['confounder_indices']):
        print(f"  {idx}: {class_names[idx]}")
