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
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP")
    parser.add_argument("--top-m", type=int, default=10,
                        help="Top-M classes for gating")
    parser.add_argument("--num-views", type=int, default=5,
                        help="Number of attribute views (K)")
    parser.add_argument("--text-model", type=str, default="mobileclip:blt",
                        help="Text model for embeddings (default: mobileclip:blt)")
    
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
    
    # 클래스 파일 (converter가 저장한 것 사용)
    parser.add_argument("--classes-file", type=str, default=None,
                        help="Path to classes.txt (from convert_lvis_to_yolo.py)")
    
    return parser.parse_args()


def run_detection_phase(config: ExperimentConfig,
                        model: YOLOE,
                        class_names: dict,
                        confounder_indices: set,
                        gt_to_model_idx: dict = None,
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
    
    # 모델 설정 - 라벨 임베딩 로드 (tools/mobileclip_blt/lvis_label_embeddings.pt)
    names = [class_names[i] for i in range(len(class_names))]
    names = [name.split("/")[0] for name in names]  # 슬래시 앞부분만 사용
    
    label_emb_path = Path("tools/mobileclip_blt/lvis_label_embeddings.pt")
    if label_emb_path.exists():
        print(f"  Loading label embeddings from {label_emb_path}...")
        label_embeddings = torch.load(label_emb_path, map_location=config.device)
        
        # names 순서대로 임베딩 추출
        txt_feats_list = []
        for name in names:
            if name in label_embeddings:
                txt_feats_list.append(label_embeddings[name])
            else:
                print(f"  WARNING: '{name}' not found in label embeddings")
                txt_feats_list.append(torch.zeros_like(next(iter(label_embeddings.values()))))
        
        txt_feats = torch.stack(txt_feats_list).unsqueeze(0).to(config.device)
        tpe = model.model.model[-1].get_tpe(txt_feats)  # head.get_tpe() 적용
    else:
        print(f"  Label embeddings not found at {label_emb_path}")
        print(f"  Run: python tools/generate_label_embedding.py --lvis")
        print(f"  Building with model.get_text_pe() (requires mobileclip_blt.pt)...")
        tpe = model.get_text_pe(names)
    
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
    
    # split 선택: minival (5000) vs val (19809)
    split_key = config.split if config.split in data else 'val'
    split_path = data.get(split_key)
    print(f"  Using split: {split_key} -> {split_path}")
    
    dataset = build_yolo_dataset(
        dataset_cfg,
        split_path,
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
    
    # 진단용 카운터
    tp_any_count = 0  # IoU만으로 TP (클래스 무시)
    tp_class_count = 0  # IoU + 클래스 일치 TP
    total_preds = 0
    total_gts = 0
    debug_printed = False  # 디버깅 출력 여부
    
    # 클래스 분포 분석 (IoU > 0.5인 쌍)
    pred_class_counts = {}  # 모델이 예측한 클래스 분포
    gt_class_counts = {}    # GT 클래스 분포
    mismatch_examples = []  # 불일치 예시 수집
    
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
            logger.stats["total_images"] += 1
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
            gt_classes_orig = gt_cls_raw.squeeze(-1).int() if gt_cls_raw.dim() > 1 else gt_cls_raw.int()
            
            # GT 인덱스를 모델 인덱스로 변환 (라벨 파일 ↔ data.yaml 순서 불일치 해결)
            if gt_to_model_idx is not None and len(gt_classes_orig) > 0:
                gt_classes = torch.tensor(
                    [gt_to_model_idx.get(c.item(), c.item()) for c in gt_classes_orig],
                    dtype=torch.int
                )
            else:
                gt_classes = gt_classes_orig
            
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
            
            # 진단: TP_any (IoU만) vs TP_class (IoU + 클래스)
            total_preds += len(pred_boxes)
            total_gts += len(gt_classes)
            
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                from ultralytics.utils.metrics import box_iou
                ious = box_iou(pred_boxes, gt_boxes)  # [Np, Ng]
                
                # TP_any: IoU > 0.5인 pred가 있으면 카운트 (클래스 무시)
                tp_any_count += (ious.max(dim=1).values > 0.5).sum().item()
                
                # TP_class: IoU > 0.5 AND 클래스 일치
                for p_idx in range(len(pred_boxes)):
                    max_iou, max_gt_idx = ious[p_idx].max(dim=0)
                    if max_iou > 0.5:
                        p_cls = pred_classes[p_idx].item()
                        g_cls = gt_classes[max_gt_idx].item()
                        
                        # 클래스 분포 수집
                        pred_class_counts[p_cls] = pred_class_counts.get(p_cls, 0) + 1
                        gt_class_counts[g_cls] = gt_class_counts.get(g_cls, 0) + 1
                        
                        if p_cls == g_cls:
                            tp_class_count += 1
                        else:
                            # 불일치 예시 수집 (최대 100개)
                            if len(mismatch_examples) < 100:
                                p_name = class_names.get(p_cls, f"?{p_cls}")
                                g_name = class_names.get(g_cls, f"?{g_cls}")
                                mismatch_examples.append(f"pred[{p_cls}]={p_name} vs GT[{g_cls}]={g_name}")
            
            # 첫 번째로 GT와 pred가 모두 있는 이미지에서 디버깅 출력
            if not debug_printed and len(gt_boxes) > 0 and len(pred_boxes) > 0:
                debug_printed = True
                print(f"\n[DEBUG] First image with detections (batch={batch_idx}, img={b}):")
                print(f"  Image path: {img_path}")
                print(f"  Pred boxes: {len(pred_boxes)}, GT boxes: {len(gt_boxes)}")
                
                if gt_to_model_idx is not None:
                    print(f"  GT index remapping: ENABLED")
                else:
                    print(f"  GT index remapping: DISABLED (indices assumed to match)")
                
                # IoU가 높은 pred-GT 쌍 찾아서 클래스 비교 (핵심 디버깅)
                ious_debug = box_iou(pred_boxes, gt_boxes)
                print(f"\n  [CLASS MAPPING DEBUG] High-IoU pairs:")
                
                # IoU > 0.3인 쌍 찾기
                high_iou_pairs = []
                for p_idx in range(min(len(pred_boxes), 50)):
                    max_iou, max_gt_idx = ious_debug[p_idx].max(dim=0)
                    if max_iou > 0.3:
                        p_cls = pred_classes[p_idx].item()
                        g_cls_orig = gt_classes_orig[max_gt_idx].item()
                        g_cls = gt_classes[max_gt_idx].item()
                        p_name = class_names.get(p_cls, f"?{p_cls}")
                        g_name = class_names.get(g_cls, f"?{g_cls}")
                        match = "MATCH" if p_cls == g_cls else "MISMATCH"
                        # 원본 GT 인덱스와 변환된 인덱스 모두 표시
                        if gt_to_model_idx is not None:
                            high_iou_pairs.append(f"    IoU={max_iou:.2f}: pred[{p_cls}]={p_name} vs GT[{g_cls_orig}→{g_cls}]={g_name} [{match}]")
                        else:
                            high_iou_pairs.append(f"    IoU={max_iou:.2f}: pred[{p_cls}]={p_name} vs GT[{g_cls}]={g_name} [{match}]")
                
                if high_iou_pairs:
                    print(f"  Found {len(high_iou_pairs)} pairs with IoU>0.3:")
                    for pair in high_iou_pairs[:15]:
                        print(pair)
                else:
                    print("    No pairs with IoU>0.3 found!")
                
                print(f"\n  IoU matrix stats: max={ious_debug.max().item():.3f}, mean={ious_debug.mean().item():.3f}")
            
            # Logger에 전달 (image_path 포함 - Phase 3에서 attribute inference용)
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
    
    # 진단 결과 출력
    print(f"\n[DIAGNOSIS] TP Analysis:")
    print(f"  Total predictions: {total_preds}")
    print(f"  Total GT boxes: {total_gts}")
    print(f"  TP_any (IoU>0.5, class ignored): {tp_any_count}")
    print(f"  TP_class (IoU>0.5 + class match): {tp_class_count}")
    print(f"  -> If TP_any >> TP_class: Class mapping is broken (A/B)")
    print(f"  -> If TP_any is also low: IoU/coordinate issue (C)")
    
    # 클래스 분포 분석 출력
    print(f"\n[DIAGNOSIS] Class Distribution (IoU>0.5 pairs):")
    print(f"  Unique pred classes: {len(pred_class_counts)}")
    print(f"  Unique GT classes: {len(gt_class_counts)}")
    
    # 가장 많이 예측된 클래스 Top 10
    if pred_class_counts:
        top_pred = sorted(pred_class_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10 predicted classes:")
        for cls_id, cnt in top_pred:
            cls_name = class_names.get(cls_id, f"?{cls_id}")
            print(f"    [{cls_id}] {cls_name}: {cnt}")
    
    # 가장 많은 GT 클래스 Top 10
    if gt_class_counts:
        top_gt = sorted(gt_class_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  Top 10 GT classes:")
        for cls_id, cnt in top_gt:
            cls_name = class_names.get(cls_id, f"?{cls_id}")
            print(f"    [{cls_id}] {cls_name}: {cnt}")
    
    # 불일치 예시 출력
    if mismatch_examples:
        print(f"\n[DIAGNOSIS] Mismatch examples (first 20 of {len(mismatch_examples)}):")
        for ex in mismatch_examples[:20]:
            print(f"    {ex}")
    
    return logger


def run_embedding_phase(config: ExperimentConfig,
                        class_names: dict,
                        verbose: bool = False) -> AttributeEmbeddingCache:
    """
    Phase 2: Attribute view 임베딩 생성
    Note: Phase 2와 3에서는 CLIP 사용 (MobileCLIP 체크포인트 로드 문제 회피)
    """
    print("\n" + "="*60)
    print("Phase 2: Attribute Embedding Generation")
    print("="*60)
    
    # MobileCLIP 사용 (캐시 있으면 캐시 사용, 없으면 모델 로드)
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
                         model,
                         output_dir: Path,
                         verbose: bool = False):
    """
    Phase 3: Confidence matching, u_sem 계산, H1 검증
    
    Args:
        model: YOLOE 모델 (cv4 접근용)
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
    
    # 3.3 u_sem 계산 (YOLOE attribute inference)
    print("\n--- Semantic Uncertainty Calculation (YOLOE attribute inference) ---")
    u_sem_calculator = SemanticUncertaintyCalculator(
        attribute_cache=attribute_cache,
        class_names=class_names,
        model=model,  # attribute inference용
        top_m=config.top_m_classes,
        device=config.device,
    )
    print(f"  Will run YOLOE inference with attribute embeddings for each detection")
    
    # YOLOE feature로 u_sem 계산
    u_sem_by_group = {}
    for group_name, detections in matched_data.items():
        if not isinstance(detections, list):
            continue
        print(f"  Computing u_sem for {group_name} ({len(detections)} samples)...")
        u_sems = u_sem_calculator.compute_for_detections(detections, use_gating=False)
        u_sem_by_group[group_name] = u_sems
    u_sem_stats = analyze_u_sem_statistics(u_sem_by_group)
    
    print(f"  u_sem statistics:")
    for group, stats in u_sem_stats.items():
        if isinstance(stats, dict):
            def _fmt(v):
                return f"{v:.8e}" if isinstance(v, float) else v
            summary = {k: _fmt(v) for k, v in stats.items()}
            print(f"    {group}: {summary}")
    
    # 3.4 Artifactness Score (Track B)
    print("\n--- Artifactness Score Calculation ---")
    # MobileCLIP 사용 (캐시 있으면 캐시 사용)
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
    
    # 1. data.yaml에서 모델 클래스 이름 로드 (모델이 사용할 순서)
    if isinstance(data["names"], dict):
        sorted_keys = sorted(data["names"].keys())
        model_names = [data["names"][k] for k in sorted_keys]
    else:
        model_names = list(data["names"])
    
    class_names = {i: name for i, name in enumerate(model_names)}
    names = model_names
    
    # 2. classes_file이 있으면 GT 라벨 인덱스 -> 모델 인덱스 매핑 생성
    # (convert_lvis_to_yolo.py가 생성한 파일 = 라벨 파일의 인덱스 순서)
    classes_file = args.classes_file if hasattr(args, 'classes_file') and args.classes_file else None
    gt_to_model_idx = None
    
    if classes_file and Path(classes_file).exists():
        print(f"Loading GT label order from: {classes_file}")
        with open(classes_file, 'r') as f:
            label_names = [line.strip() for line in f if line.strip()]
        
        # 모델 클래스 이름 -> 모델 인덱스 (슬래시 앞 부분만 비교용으로 추가)
        name_to_model_idx = {}
        for i, name in enumerate(model_names):
            name_to_model_idx[name] = i
            # 슬래시가 있으면 첫 부분도 등록
            if "/" in name:
                name_to_model_idx[name.split("/")[0]] = i
        
        # GT 라벨 인덱스 -> 모델 인덱스 매핑
        gt_to_model_idx = {}
        matched = 0
        for label_idx, label_name in enumerate(label_names):
            if label_name in name_to_model_idx:
                gt_to_model_idx[label_idx] = name_to_model_idx[label_name]
                matched += 1
            elif label_name.split("/")[0] in name_to_model_idx:
                gt_to_model_idx[label_idx] = name_to_model_idx[label_name.split("/")[0]]
                matched += 1
            else:
                # 매핑 실패 - 그대로 유지 (경고)
                gt_to_model_idx[label_idx] = label_idx
        
        print(f"  Label->Model index mapping: {matched}/{len(label_names)} classes matched")
        
        # 매핑 샘플 출력 (디버깅)
        print(f"  Mapping samples:")
        for i in [0, 1, 76, 430, len(label_names)-1]:
            if i < len(label_names):
                label_name = label_names[i]
                model_idx = gt_to_model_idx.get(i, i)
                model_name = model_names[model_idx] if model_idx < len(model_names) else "?"
                print(f"    GT[{i}]={label_name} -> Model[{model_idx}]={model_name}")
    else:
        print(f"  No classes_file provided or file not found. Assuming GT and model indices match.")
    
    print(f"train: {data.get('train')}")
    print(f"val: {data.get('val')}")
    print(f"nc: {data.get('nc')}, names_len: {len(names)}")
    
    # 클래스 순서 확인용 출력
    print(f"First 5 classes: {names[:5]}")
    print(f"Last 5 classes: {names[-5:]}")
    
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
    
    # 모델 로드 (Phase 1 & Phase 3에서 필요)
    print("\n--- Loading YOLO-E Model ---")
    model = YOLOE(config.checkpoint)
    model.to(config.device)
    model.eval()
    
    # Phase 1: Detection
    if not args.skip_detection:
        logger = run_detection_phase(
            config, model, class_names, confounder_indices,
            gt_to_model_idx=gt_to_model_idx,
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
        config, logger, attribute_cache, class_names, model, output_dir,
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
