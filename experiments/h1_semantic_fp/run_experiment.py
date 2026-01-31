"""
H1 Semantic FP Experiment - Main Runner
========================================
Fig.1 주장을 증거로 만드는 핵심 실험 실행 스크립트

실행 방법:
    python -m experiments.h1_semantic_fp.run_experiment --checkpoint yoloe-v8l-seg.pt
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLOE
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .config import (
    ExperimentConfig, 
    get_default_config,
)
from .confounder_classes import (
    load_lvis_class_names,
    build_confounder_set,
    analyze_confounder_coverage,
)
from .detection_logger import DetectionLogger
from .confidence_matching import (
    ConfidenceMatchedSampler,
    create_confidence_matched_dataset,
)
from .attribute_embeddings import (
    AttributeEmbeddingGenerator,
    AttributeEmbeddingCache,
)
from .semantic_uncertainty import (
    SemanticUncertaintyCalculator,
    analyze_u_sem_statistics,
)
from .artifactness_score import (
    ArtifactnessScorer,
    analyze_artifactness_statistics,
)
from .h1_metrics import (
    H1Evaluator,
    format_h1_results,
)
from .visualize import save_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description="H1 Semantic FP Experiment")
    
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt",
                        help="YOLO-E checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP")
    parser.add_argument("--top-m", type=int, default=10,
                        help="Top-M classes for gating")
    parser.add_argument("--num-views", type=int, default=5,
                        help="Number of attribute views (K)")
    parser.add_argument("--text-model", type=str, default="clip:ViT-B/32",
                        help="Text model for embeddings (default: clip:ViT-B/32)")
    
    # 단계별 실행
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip detection and load from cache")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding generation and load from cache")
    parser.add_argument("--detection-cache", type=str, default=None,
                        help="Path to detection cache")
    parser.add_argument("--embedding-cache", type=str, default=None,
                        help="Path to embedding cache")
    
    # 디버깅
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process (for debugging)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def get_text_embeddings_with_clip(names: list, device: str = "cuda", 
                                   text_model_name: str = "clip:ViT-B/32") -> torch.Tensor:
    """
    CLIP을 사용하여 텍스트 임베딩 생성 (MobileCLIP 의존성 제거)
    
    Args:
        names: 클래스 이름 리스트
        device: 디바이스
        text_model_name: 텍스트 모델 이름 (default: clip:ViT-B/32)
    """
    from ultralytics.nn.text_model import build_text_model
    
    print(f"  Building text embeddings for {len(names)} classes using {text_model_name}...")
    text_model = build_text_model(text_model_name, device=device)
    text_model.eval()
    
    with torch.no_grad():
        tokens = text_model.tokenize(names)
        txt_feats = text_model.encode_text(tokens)
        # [1, num_classes, embed_dim] 형태로 변환, float32로 명시적 변환
        txt_feats = txt_feats.unsqueeze(0).to(torch.float32)
    
    return txt_feats


def run_detection_phase(config: ExperimentConfig,
                        model: YOLOE,
                        class_names: dict,
                        confounder_indices: set,
                        max_images: int = None,
                        verbose: bool = False) -> DetectionLogger:
    """
    Phase 1: LVIS val에서 detection 수행 및 Triad Split
    """
    print("\n" + "="*60)
    print("Phase 1: Detection and Triad Split")
    print("="*60)
    
    # 데이터 로더 준비
    data = check_det_dataset(config.data_yaml)
    
    # Detection logger 초기화
    logger = DetectionLogger(
        class_names=class_names,
        confounder_indices=confounder_indices,
        tp_iou_threshold=config.iou_threshold,
        semantic_fp_iou_threshold=config.semantic_fp_iou_threshold,
        top_k=config.top_m_classes,
    )
    
    # 모델 설정 - CLIP 직접 사용 (MobileCLIP 의존성 제거)
    names = [name.split("/")[0] for name in list(class_names.values())]
    tpe = get_text_embeddings_with_clip(names, device=config.device, text_model_name=config.text_model)
    model.set_classes(names, tpe)
    
    # Validation 데이터셋 로드
    # cfg 객체 생성 (build_yolo_dataset에 필요한 모든 속성)
    from types import SimpleNamespace
    dataset_cfg = SimpleNamespace(
        imgsz=config.imgsz,
        rect=True,
        cache=False,
        single_cls=False,
        task='detect',
        classes=None,
        fraction=1.0,
        load_vp=False,
        # augmentation 관련 (validation이므로 비활성화)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # mask 관련
        mask_ratio=4,
        overlap_mask=False,
        # 기타
        bgr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        crop_fraction=1.0,
    )
    
    dataset = build_yolo_dataset(
        dataset_cfg,
        data.get('val'),
        batch=config.batch_size,
        data=data,
        mode='val',
        stride=32,
        rect=True,
    )
    
    dataloader = build_dataloader(
        dataset,
        batch=config.batch_size,
        workers=config.num_workers,
        shuffle=False,
        rank=-1,
    )
    
    # Detection 수행
    total_images = len(dataloader)
    if max_images:
        total_images = min(total_images, max_images)
    
    print(f"Processing {total_images} images...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, total=total_images)):
        if max_images and batch_idx >= max_images:
            break
        
        # 이미지 처리 - float32로 변환 및 정규화 (0-1 범위)
        imgs = batch["img"].to(model.device).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        
        # 추론
        with torch.no_grad():
            results = model.predict(imgs, verbose=False, conf=config.conf_threshold)
        
        # 이미지 파일 경로 추출
        im_files = batch.get("im_file", [None] * len(results))
        
        # Detection 로깅 - GT와 매칭
        for b, result in enumerate(results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # Prediction 정보 추출
            pred_boxes = result.boxes.xyxy.cpu()  # [N, 4]
            pred_confs = result.boxes.conf.cpu()  # [N]
            pred_classes = result.boxes.cls.cpu().int()  # [N]
            
            # GT 정보 추출 (batch에서)
            batch_mask = batch["batch_idx"] == b
            gt_boxes_norm = batch["bboxes"][batch_mask]  # normalized xywh
            gt_cls_raw = batch["cls"][batch_mask]
            # cls가 [N, 1] 또는 [N] 형태일 수 있음
            gt_classes = gt_cls_raw.squeeze(-1).int() if gt_cls_raw.dim() > 1 else gt_cls_raw.int()
            
            # GT boxes를 xyxy로 변환 (이미지 크기 기준)
            img_h, img_w = imgs.shape[2], imgs.shape[3]
            if len(gt_boxes_norm) > 0:
                gt_boxes = gt_boxes_norm.clone()
                # xywh -> xyxy
                gt_boxes[:, 0] = (gt_boxes_norm[:, 0] - gt_boxes_norm[:, 2] / 2) * img_w
                gt_boxes[:, 1] = (gt_boxes_norm[:, 1] - gt_boxes_norm[:, 3] / 2) * img_h
                gt_boxes[:, 2] = (gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2] / 2) * img_w
                gt_boxes[:, 3] = (gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3] / 2) * img_h
            else:
                gt_boxes = torch.zeros((0, 4))
            
            # 이미지 ID 및 경로
            img_id = f"batch{batch_idx}_img{b}"
            img_path = im_files[b] if b < len(im_files) else None
            
            # 첫 배치에서 디버깅 정보 출력 (클래스 이름으로)
            if batch_idx == 0 and b == 0:
                print(f"\n[DEBUG] First batch sanity check:")
                print(f"  Image path: {img_path}")
                print(f"  Pred boxes: {len(pred_boxes)}, GT boxes: {len(gt_boxes)}")
                
                # 클래스 이름으로 변환하여 출력
                pred_cls_list = pred_classes[:5].tolist()
                gt_cls_list = gt_classes[:5].tolist() if len(gt_classes) > 0 else []
                
                pred_names = [class_names.get(c, f"?{c}") for c in pred_cls_list]
                gt_names = [class_names.get(c, f"?{c}") for c in gt_cls_list]
                
                print(f"  Pred classes (first 5): {pred_cls_list} -> {pred_names}")
                print(f"  GT classes (first 5): {gt_cls_list} -> {gt_names}")
                
                # IoU 샘플 출력
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    from ultralytics.utils.metrics import box_iou
                    sample_ious = box_iou(pred_boxes[:3], gt_boxes[:3])
                    print(f"  Sample IoU (3x3): max={sample_ious.max().item():.3f}")
            
            # Logger에 전달
            logger.process_batch(
                preds=torch.cat([pred_boxes, pred_confs.unsqueeze(1), pred_classes.unsqueeze(1).float()], dim=1).unsqueeze(0),
                gt_bboxes=[gt_boxes],
                gt_classes=[gt_classes],
                image_ids=[img_id],
                image_paths=[img_path],
            )
        
        if verbose and batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: {logger.stats}")
    
    print(f"\nDetection complete!")
    print(f"Stats: {logger.get_stats()}")
    
    return logger


def run_embedding_phase(config: ExperimentConfig,
                        class_names: dict,
                        verbose: bool = False) -> AttributeEmbeddingCache:
    """
    Phase 2: Attribute view 임베딩 생성
    """
    print("\n" + "="*60)
    print("Phase 2: Attribute Embedding Generation")
    print("="*60)
    
    generator = AttributeEmbeddingGenerator(
        text_model_name=config.text_model,
        device=config.device,
        num_views=config.num_attribute_views,
    )
    
    print(f"Generating {config.num_attribute_views} attribute views for {len(class_names)} classes...")
    
    cache = generator.generate_all_class_views(
        class_names=class_names,
        use_paraphrase=False,
        show_progress=True,
    )
    
    print(f"Embedding generation complete!")
    
    return cache


def run_evaluation_phase(config: ExperimentConfig,
                         logger: DetectionLogger,
                         attribute_cache: AttributeEmbeddingCache,
                         class_names: dict,
                         output_dir: Path,
                         verbose: bool = False):
    """
    Phase 3: Confidence matching, u_sem 계산, H1 검증
    """
    print("\n" + "="*60)
    print("Phase 3: Evaluation")
    print("="*60)
    
    # 3.1 Triad Split
    print("\n--- Triad Split ---")
    triad_split = logger.get_triad_split()
    
    for group, dets in triad_split.items():
        print(f"  {group}: {len(dets)} detections")
    
    # 3.2 Confidence Matching
    print("\n--- Confidence Matching ---")
    matched_data, verification = create_confidence_matched_dataset(
        triad_split,
        config={
            "bin_start": config.conf_bin_start,
            "bin_end": config.conf_bin_end,
            "bin_step": config.conf_bin_step,
            "samples_per_bin": config.samples_per_bin,
            "seed": config.seed,
        }
    )
    
    print(f"  Matching verification:")
    print(f"    Total samples: {verification['total_samples']}")
    print(f"    Bins used: {verification['num_bins_used']}")
    print(f"    Max mean difference: {verification['max_mean_difference']:.4f}")
    print(f"    Well matched: {verification['is_well_matched']}")
    
    # Matching 실패 시 원본 데이터 사용 (fallback)
    if verification['total_samples'] == 0:
        print("  WARNING: Confidence matching failed. Using unmatched data.")
        matched_data = triad_split
    
    # 3.3 u_sem 계산 (CLIP crop embedding 사용)
    print("\n--- Semantic Uncertainty Calculation (CLIP crop) ---")
    u_sem_calculator = SemanticUncertaintyCalculator(
        attribute_cache=attribute_cache,
        class_names=class_names,
        top_m=config.top_m_classes,
        device=config.device,
    )
    
    # 이미지 디렉토리 (data yaml에서 가져옴)
    data = check_det_dataset(config.data_yaml)
    image_dir = Path(data.get("path", "")) / "val2017"  # COCO/LVIS 기본 경로
    
    # CLIP crop embedding으로 u_sem 계산
    u_sem_by_group = u_sem_calculator.compute_for_triad_split_with_images(
        matched_data, str(image_dir)
    )
    u_sem_stats = analyze_u_sem_statistics(u_sem_by_group)
    
    print(f"  u_sem statistics:")
    for group, stats in u_sem_stats.items():
        if isinstance(stats, dict):
            print(f"    {group}: mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f}")
    
    # 3.4 Artifactness Score (Track B)
    print("\n--- Artifactness Score Calculation ---")
    art_scorer = ArtifactnessScorer(
        text_model_name=config.text_model,
        device=config.device,
        method="margin",
    )
    
    s_art_by_group = art_scorer.compute_for_triad_split(matched_data)
    s_art_stats = analyze_artifactness_statistics(s_art_by_group)
    
    print(f"  Artifactness statistics:")
    for group, stats in s_art_stats.items():
        print(f"    {group}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # 3.5 H1 검증
    print("\n--- H1 Verification ---")
    evaluator = H1Evaluator()
    
    conf_tp = np.array([d.confidence for d in matched_data["TP"]])
    conf_sem_fp = np.array([d.confidence for d in matched_data["Semantic_FP"]])
    
    h1_result = evaluator.evaluate(
        u_sem_tp=u_sem_by_group["TP"],
        u_sem_semantic_fp=u_sem_by_group["Semantic_FP"],
        confidence_tp=conf_tp,
        confidence_semantic_fp=conf_sem_fp,
    )
    
    print(format_h1_results(h1_result))
    
    # 3.6 결과 저장
    print("\n--- Saving Results ---")
    
    # JSON 결과
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "checkpoint": config.checkpoint,
            "iou_threshold": config.iou_threshold,
            "top_m": config.top_m_classes,
            "num_views": config.num_attribute_views,
        },
        "detection_stats": logger.get_stats(),
        "matching_verification": verification,
        "u_sem_stats": u_sem_stats,
        "s_art_stats": s_art_stats,
        "h1_result": {
            "auroc_u_sem": h1_result.auroc_u_sem,
            "aupr_u_sem": h1_result.aupr_u_sem,
            "auroc_confidence": h1_result.auroc_confidence,
            "cohens_d_u_sem": h1_result.cohens_d_u_sem,
            "cohens_d_confidence": h1_result.cohens_d_confidence,
            "n_tp": h1_result.n_tp,
            "n_semantic_fp": h1_result.n_semantic_fp,
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 시각화
    print("\n--- Generating Visualizations ---")
    confidence_data = {
        "TP": conf_tp,
        "Semantic_FP": conf_sem_fp,
        "Background_FP": np.array([d.confidence for d in matched_data["Background_FP"]]),
    }
    
    roc_data = evaluator.get_roc_curve_data(
        u_sem_by_group["TP"],
        u_sem_by_group["Semantic_FP"],
    )
    
    save_all_figures(
        output_dir / "figures",
        confidence_data,
        u_sem_by_group,
        roc_data,
        h1_result.auroc_u_sem,
        logger.get_stats(),
    )
    
    print(f"\nAll results saved to {output_dir}")
    
    return h1_result


def main():
    args = parse_args()
    
    # 설정
    config = get_default_config()
    config.checkpoint = args.checkpoint
    config.device = args.device
    config.batch_size = args.batch_size
    config.conf_threshold = args.conf_threshold
    config.iou_threshold = args.iou_threshold
    config.top_m_classes = args.top_m
    config.num_attribute_views = args.num_views
    config.text_model = args.text_model
    
    # 출력 디렉토리
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/h1_semantic_fp/outputs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("H1 Semantic FP Experiment")
    print("="*60)
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Device: {config.device}")
    print(f"Output: {output_dir}")
    
    # 클래스 로드 - data yaml에서 직접 가져옴 (인덱스 체계 통일)
    print("\n--- Loading Classes from Data YAML ---")
    data = check_det_dataset(config.data_yaml)
    
    # data["names"]는 {0: 'name0', 1: 'name1', ...} 또는 ['name0', 'name1', ...] 형태
    if isinstance(data["names"], dict):
        class_names = data["names"]
    else:
        class_names = {i: name for i, name in enumerate(data["names"])}
    
    # Confounder indices도 동일한 인덱스 체계로 구축
    confounder_indices = build_confounder_set(class_names)
    
    print(f"Total classes: {len(class_names)}")
    print(f"Confounder classes: {len(confounder_indices)}")
    
    # Confounder 분석 저장
    confounder_report = analyze_confounder_coverage(class_names)
    with open(output_dir / "confounder_analysis.json", "w") as f:
        # numpy를 json serializable하게 변환
        report_serializable = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in confounder_report.items()
        }
        report_serializable["confounder_names"] = {
            str(k): v for k, v in confounder_report["confounder_names"].items()
        }
        json.dump(report_serializable, f, indent=2)
    
    # Phase 1: Detection
    if not args.skip_detection:
        # 모델 로드
        print("\n--- Loading YOLO-E Model ---")
        model = YOLOE(config.checkpoint)
        model.to(config.device)
        model.eval()
        
        logger = run_detection_phase(
            config, model, class_names, confounder_indices,
            max_images=args.max_images,
            verbose=args.verbose,
        )
        
        # 캐시 저장
        logger.save(output_dir / "detection_cache.pkl")
    else:
        # 캐시 로드
        cache_path = args.detection_cache or output_dir / "detection_cache.pkl"
        print(f"\nLoading detection cache from {cache_path}")
        logger = DetectionLogger.load(cache_path)
    
    # Phase 2: Embedding
    if not args.skip_embedding:
        attribute_cache = run_embedding_phase(
            config, class_names, verbose=args.verbose
        )
        
        # 캐시 저장
        generator = AttributeEmbeddingGenerator(
            text_model_name=config.text_model,
            device=config.device
        )
        generator.save_cache(attribute_cache, output_dir / "attribute_cache.json")
    else:
        # 캐시 로드
        cache_path = args.embedding_cache or output_dir / "attribute_cache.json"
        print(f"\nLoading embedding cache from {cache_path}")
        attribute_cache = AttributeEmbeddingGenerator.load_cache(cache_path)
    
    # Phase 3: Evaluation
    h1_result = run_evaluation_phase(
        config, logger, attribute_cache, class_names, output_dir,
        verbose=args.verbose,
    )
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  u_sem AUROC: {h1_result.auroc_u_sem:.4f}")
    print(f"  Confidence AUROC (baseline): {h1_result.auroc_confidence:.4f}")
    print(f"  Cohen's d: {h1_result.cohens_d_u_sem:.4f}")


if __name__ == "__main__":
    main()
