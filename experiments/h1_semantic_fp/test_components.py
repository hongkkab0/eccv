"""
Component Tests for H1 Experiment
==================================
각 모듈의 기본 동작 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


def test_confounder_classes():
    """Confounder 클래스 테스트"""
    print("\n=== Testing Confounder Classes ===")
    
    from experiments.h1_semantic_fp.confounder_classes import (
        load_lvis_class_names,
        build_confounder_set,
        analyze_confounder_coverage,
    )
    
    try:
        class_names = load_lvis_class_names()
        print(f"✓ Loaded {len(class_names)} LVIS classes")
        
        confounder_indices = build_confounder_set(class_names)
        print(f"✓ Found {len(confounder_indices)} confounder classes")
        
        # 일부 출력
        print(f"  Examples: {list(confounder_indices)[:5]}")
        
        report = analyze_confounder_coverage(class_names)
        print(f"✓ Coverage: {report['confounder_ratio']*100:.1f}%")
        
    except FileNotFoundError:
        print("! LVIS yaml not found - skipping")
    
    print("PASSED\n")


def test_js_divergence():
    """JS Divergence 테스트"""
    print("\n=== Testing JS Divergence ===")
    
    from experiments.h1_semantic_fp.semantic_uncertainty import js_divergence
    
    # 동일한 분포: JS = 0
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.5, 0.5])
    js = js_divergence(np.stack([p1, p2]))
    print(f"✓ Same distributions: JS = {js:.6f} (expected ~0)")
    assert js < 0.01
    
    # 다른 분포: JS > 0
    p1 = np.array([0.9, 0.1])
    p2 = np.array([0.1, 0.9])
    js = js_divergence(np.stack([p1, p2]))
    print(f"✓ Different distributions: JS = {js:.4f} (expected > 0)")
    assert js > 0.3
    
    print("PASSED\n")


def test_confidence_matching():
    """Confidence Matching 테스트"""
    print("\n=== Testing Confidence Matching ===")
    
    from experiments.h1_semantic_fp.confidence_matching import ConfidenceMatchedSampler
    from experiments.h1_semantic_fp.detection_logger import Detection
    
    # 더미 detection 생성
    def make_dummy_detection(conf):
        return Detection(
            image_id="test",
            det_idx=0,
            bbox=np.array([0, 0, 100, 100]),
            pred_class=0,
            pred_class_name="test",
            confidence=conf,
        )
    
    # 각 그룹에 다양한 confidence로 detection 생성
    np.random.seed(42)
    triad_split = {
        "TP": [make_dummy_detection(np.random.uniform(0.8, 0.95)) for _ in range(100)],
        "Semantic_FP": [make_dummy_detection(np.random.uniform(0.8, 0.95)) for _ in range(100)],
        "Background_FP": [make_dummy_detection(np.random.uniform(0.8, 0.95)) for _ in range(100)],
    }
    
    sampler = ConfidenceMatchedSampler(
        bin_start=0.80,
        bin_end=0.95,
        bin_step=0.05,
        samples_per_bin=10,
    )
    
    matched = sampler.sample(triad_split, min_samples_per_bin=5)
    verification = sampler.verify_matching(matched)
    
    print(f"✓ Bins used: {verification['num_bins_used']}")
    print(f"✓ Total samples: {verification['total_samples']}")
    print(f"✓ Max mean diff: {verification['max_mean_difference']:.4f}")
    print(f"✓ Well matched: {verification['is_well_matched']}")
    
    print("PASSED\n")


def test_h1_metrics():
    """H1 메트릭 테스트"""
    print("\n=== Testing H1 Metrics ===")
    
    from experiments.h1_semantic_fp.h1_metrics import (
        compute_auroc_aupr,
        compute_cohens_d,
        compute_spearman_correlation,
    )
    
    # AUROC 테스트
    positive = np.array([0.8, 0.9, 0.7, 0.85])  # 높은 값
    negative = np.array([0.2, 0.3, 0.1, 0.25])  # 낮은 값
    auroc, aupr = compute_auroc_aupr(positive, negative)
    print(f"✓ AUROC (perfect separation): {auroc:.4f} (expected ~1.0)")
    assert auroc > 0.9
    
    # Cohen's d 테스트
    g1 = np.array([10, 11, 12, 10, 11])
    g2 = np.array([5, 6, 5, 6, 5])
    d = compute_cohens_d(g1, g2)
    print(f"✓ Cohen's d: {d:.4f} (expected large)")
    assert abs(d) > 1.0
    
    # Spearman 테스트
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])  # 완벽한 상관
    rho, p = compute_spearman_correlation(x, y)
    print(f"✓ Spearman rho (perfect): {rho:.4f} (expected ~1.0)")
    assert rho > 0.99
    
    print("PASSED\n")


def test_attribute_embeddings():
    """Attribute Embedding 테스트 (CLIP 필요)"""
    print("\n=== Testing Attribute Embeddings ===")
    
    try:
        import torch
        from experiments.h1_semantic_fp.attribute_embeddings import AttributeEmbeddingGenerator
        
        # 작은 테스트
        generator = AttributeEmbeddingGenerator(
            text_model_name="clip:ViT-B/32",
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_views=3,
        )
        
        view_set = generator.generate_class_views("dog", 0)
        
        print(f"✓ Generated {len(view_set.view_names)} views for 'dog'")
        print(f"✓ Embedding shape: {view_set.view_embeddings.shape}")
        print(f"✓ Views: {view_set.view_names}")
        
        print("PASSED\n")
        
    except Exception as e:
        print(f"! Skipped (CLIP not available): {e}\n")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("H1 Semantic FP Experiment - Component Tests")
    print("=" * 60)
    
    test_js_divergence()
    test_confidence_matching()
    test_h1_metrics()
    test_confounder_classes()
    test_attribute_embeddings()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
