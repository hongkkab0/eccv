# H1 Semantic FP Experiment

**Fig.1 주장을 증거로 만드는 핵심 실험**

## 개요

이 실험은 다음 가설을 검증합니다:

**H1**: Semantic False Positive는 attribute violation이 높다  
→ `u_sem` (JS divergence 기반 semantic uncertainty)이 TP와 Semantic FP를 구분할 수 있다

## 실험 설계

### A-1. 데이터셋/추론 세팅
- **Dataset**: LVIS v1 val (전체)
- **Vocabulary**: LVIS 1,203 클래스 (YOLO-E 텍스트 임베딩)
- **Detector**: YOLO-E (frozen)
- **Detection match**: IoU ≥ 0.5 기준 TP/FP 판정

### A-2. Triad Split
Detection을 세 그룹으로 자동 분류:
1. **TP (True Positive)**: 예측 class = GT class, IoU ≥ 0.5
2. **Semantic FP**: FP 중 confounder 클래스 GT와 IoU ≥ 0.3인 경우
3. **Background FP**: 그 외 FP

**Confounder 클래스**: statue, sculpture, figurine, doll, toy, poster, painting 등
(depiction/replica 계열 - `confounder_classes.py` 참조)

### A-3. Confidence-Matched 평가
세 그룹의 confidence 분포를 동일하게 매칭하여 "confidence로는 구분 불가"를 증명:
- Bin 구간: 0.80~0.95 (0.02 step)
- 각 bin에서 그룹별 동일 수 샘플링

### A-4. H1 검증
```
u_sem(f) = JS(p^(1), ..., p^(K))
```

- **변별력**: AUROC, AUPR, Cohen's d로 측정
- **상관관계**: Spearman ρ(u_sem, human violation count)
- **대조군**: Paraphrase ensemble disagreement

## 사용법

### 기본 실행
```bash
python -m experiments.h1_semantic_fp.run_experiment \
    --checkpoint yoloe-v8l-seg.pt \
    --device cuda:0
```

### 옵션
```bash
python -m experiments.h1_semantic_fp.run_experiment \
    --checkpoint yoloe-v8l-seg.pt \
    --device cuda:0 \
    --batch-size 1 \
    --conf-threshold 0.001 \
    --iou-threshold 0.5 \
    --top-m 10 \
    --num-views 5 \
    --output-dir experiments/h1_semantic_fp/outputs/my_exp \
    --max-images 1000 \
    --verbose
```

### 캐시 사용 (재실행 시)
```bash
# Detection만 스킵
python -m experiments.h1_semantic_fp.run_experiment \
    --skip-detection \
    --detection-cache path/to/detection_cache.pkl

# Embedding도 스킵
python -m experiments.h1_semantic_fp.run_experiment \
    --skip-detection --skip-embedding \
    --detection-cache path/to/detection_cache.pkl \
    --embedding-cache path/to/attribute_cache.json
```

## 출력 파일

```
outputs/
├── results.json           # 전체 결과 요약
├── confounder_analysis.json
├── detection_cache.pkl    # Detection 캐시
├── attribute_cache.json   # Attribute embedding 캐시
└── figures/
    ├── confidence_histogram.png   # Confidence matching 증명
    ├── u_sem_distribution.png     # u_sem 분포 비교
    ├── roc_curve.png              # ROC curve
    ├── triad_split.png            # Triad split 분포
    └── summary_figure.png         # 논문용 종합 Figure
```

## 모듈 구성

| 모듈 | 설명 |
|------|------|
| `config.py` | 실험 설정 |
| `confounder_classes.py` | LVIS confounder 클래스 정의 |
| `detection_logger.py` | Detection 로깅 및 Triad Split |
| `confidence_matching.py` | Confidence-matched 샘플링 |
| `attribute_embeddings.py` | Attribute view 임베딩 생성 (Track A) |
| `semantic_uncertainty.py` | u_sem (JS divergence) 계산 |
| `artifactness_score.py` | Track B: Artifactness score |
| `h1_metrics.py` | H1 검증 메트릭 |
| `region_feature_extractor.py` | YOLO-E region feature 추출 |
| `visualize.py` | 시각화 도구 |
| `run_experiment.py` | 메인 실행 스크립트 |

## 기대 결과

### 성공 기준
- `u_sem` AUROC > 0.7 (TP vs Semantic FP)
- Confidence AUROC ≈ 0.5 (matching 증명)
- Cohen's d > 0.5 (medium effect size)
- Spearman ρ(u_sem, violations) > 0 (p < 0.05)

### 해석 가이드
```
=== H1 Verification Results ===
u_sem AUROC:      0.78    # 좋음 (> 0.7)
Confidence AUROC: 0.51    # 예상대로 ~0.5 (matching 성공)
Cohen's d:        0.85    # Large effect size

✓ u_sem shows good discriminative power
✓ Confidence is near chance as expected
✓ Large effect size
```

## Track A vs Track B

**Track A (u_sem)**: Class-consistent semantic probe
- Attribute view embedding으로 semantic consistency 측정
- 논문의 메인 기여

**Track B (s_art)**: Depiction/replica detector
- "is it a toy?", "is it a statue?" 등 판별 질문
- u_sem과 결합하여 최종 error probability 계산

```python
u = σ(w1 * u_sem + w2 * s_art + w3 * u_ret + w4 * u_loc)
```

## 참고

- LVIS Dataset: https://www.lvisdataset.org/
- YOLO-E: This repository
