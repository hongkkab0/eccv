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
    EnhancedTriadSplit,
    js_divergence,
)
from scipy.special import softmax
from .artifactness_score import ArtifactnessScorer
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
    
    # fuse 상태 확인 (경고만, 실패 아님 - cls_logits는 fused에서도 사용 가능)
    head = model.model.model[-1]
    cv3_out_dim = head.cv3[0][-1].weight.shape[0] if hasattr(head, 'cv3') else -1
    embed_dim = head.embed if hasattr(head, 'embed') else 512
    is_fused = cv3_out_dim != embed_dim
    
    print(f"  Model fuse status:")
    print(f"    cv3 output dim: {cv3_out_dim}")
    print(f"    embed_dim: {embed_dim}")
    print(f"    is_fused: {is_fused}")
    
    if is_fused:
        print(f"  NOTE: Model is fused - will use cls_logits for u_sem (no 512-dim features)")
    else:
        print(f"  OK: Model is NOT fused - 512-dim features available")
    
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
    
    # YOLOE head 참조 (region feature 추출용)
    head = model.model.model[-1]
    
    # fuse 상태 확인 및 경고
    if hasattr(head, 'is_fused') and head.is_fused:
        print("  WARNING: Model is already fused. Region features will not be available.")
        print("  To capture region features, use a non-fused model.")
    
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
        
        # Region features 캡처 (fuse 전에만 유효)
        region_features = None
        if hasattr(head, '_region_features') and head._region_features is not None:
            region_features = head._region_features[0].cpu()  # [embed, num_anchors]
        
        # cls_logits 캡처 (항상 유효 - fused 모델에서도)
        cls_logits = None
        if hasattr(head, '_last_cls') and head._last_cls is not None:
            cls_logits = head._last_cls[0].cpu()  # [nc, num_anchors]
        
        # anchor 정보 캡처
        anchor_xy = None
        anchor_strides = None
        if hasattr(head, '_last_anchor_xy') and head._last_anchor_xy is not None:
            anchor_xy = head._last_anchor_xy.cpu()  # [num_anchors, 2]
            anchor_strides = head._last_strides.cpu()  # [num_anchors]
        
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
            
            # Detection별 feature/logits 추출 (bbox 중심에 가장 가까운 anchor)
            det_region_features = None
            det_cls_logits = None
            
            if anchor_xy is not None and anchor_strides is not None:
                # ===== anchor shape 정규화 (브로드캐스팅 버그 수정) =====
                # anchor_xy: [A, 2] or [2, A] -> [A, 2]
                _anchor_xy = anchor_xy.float()
                if _anchor_xy.ndim == 2 and _anchor_xy.shape[0] == 2 and _anchor_xy.shape[1] != 2:
                    _anchor_xy = _anchor_xy.t().contiguous()
                _anchor_xy = _anchor_xy.reshape(-1, 2)  # [A, 2]
                
                # strides: [1, A] or [A] -> [A]
                _strides = anchor_strides.float()
                if _strides.ndim == 2:
                    _strides = _strides.squeeze(0)
                _strides = _strides.reshape(-1)  # [A]
                
                # anchor를 픽셀 좌표로 변환: [A, 2] * [A, 1] = [A, 2]
                anchor_px = _anchor_xy * _strides[:, None]
                # =========================================================
                
                det_feats = []
                det_logits = []
                
                for p_idx in range(len(pred_boxes)):
                    cx = (pred_boxes[p_idx, 0] + pred_boxes[p_idx, 2]) / 2
                    cy = (pred_boxes[p_idx, 1] + pred_boxes[p_idx, 3]) / 2
                    
                    # nearest anchor 찾기
                    dist = ((anchor_px[:, 0] - cx) ** 2 + (anchor_px[:, 1] - cy) ** 2)
                    nearest_idx = dist.argmin().item()
                    
                    # region feature (있으면)
                    if region_features is not None:
                        det_feats.append(region_features[:, nearest_idx])
                    
                    # cls logits (항상)
                    if cls_logits is not None:
                        det_logits.append(cls_logits[:, nearest_idx])
                
                if det_feats:
                    det_region_features = torch.stack(det_feats).unsqueeze(0)  # [1, N, embed]
                if det_logits:
                    det_cls_logits = torch.stack(det_logits).unsqueeze(0)  # [1, N, nc]
            
            # Logger에 전달 (image_path + region_feature + cls_logits 포함)
            logger.process_batch(
                preds=torch.cat([pred_boxes, pred_confs.unsqueeze(1), pred_classes.unsqueeze(1).float()], dim=1).unsqueeze(0),
                gt_bboxes=[gt_boxes],
                gt_classes=[gt_classes],
                image_ids=[img_id],
                image_paths=[img_path],
                region_features=det_region_features,
                cls_logits=det_cls_logits,
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
    Phase 3: Confidence matching, u_sem/s_art 계산, H1 검증
    
    핵심 변경:
    1. MobileCLIP 일관성 확보
    2. Depiction_FP를 artifactness score로 슬라이싱 (클래스 리스트 X)
    3. 샘플 drop 원인 로깅
    4. u_sem 유닛테스트
    """
    print("\n" + "="*60)
    print("Phase 3: Evaluation")
    print("="*60)
    
    # 3.1 기존 Triad Split
    print("\n--- Original Triad Split ---")
    triad_split = logger.get_triad_split()
    
    for group, dets in triad_split.items():
        print(f"  {group}: {len(dets)} detections")
    
    # 3.2 Confidence Matching (기존 triad split 기반)
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
    
    # 3.3 u_sem 계산 (캐시된 MobileCLIP embeddings + region feature)
    print("\n--- Semantic Uncertainty Calculation ---")
    u_sem_calculator = SemanticUncertaintyCalculator(
        class_names=class_names,
        device=config.device,
        temperature=10.0,
        top_m=config.top_m_classes,
        cache_dir="tools/mobileclip_blt",
    )
    
    # Region feature 유무 확인
    has_region_features = False
    sample_feat = None
    for group_name, detections in matched_data.items():
        if isinstance(detections, list):
            for det in detections[:10]:
                if det.region_feature is not None:
                    has_region_features = True
                    sample_feat = det.region_feature
                    break
            if has_region_features:
                break
    
    # cls_logits 유무 확인 (fused 모델에서도 사용 가능)
    has_cls_logits = False
    sample_logits = None
    for group_name, detections in matched_data.items():
        if isinstance(detections, list):
            for det in detections[:10]:
                if det.cls_logits is not None:
                    has_cls_logits = True
                    sample_logits = det.cls_logits
                    break
            if has_cls_logits:
                break
    
    if has_region_features:
        print(f"  Region features available (unfused model)")
        print(f"  Sample feature: shape={sample_feat.shape}, norm={np.linalg.norm(sample_feat):.4f}")
    elif has_cls_logits:
        print(f"  Region features NOT available (fused model)")
        print(f"  Using cls_logits instead: shape={sample_logits.shape}")
        print(f"  NOTE: u_sem will use entropy-based method (not JS divergence)")
    else:
        raise RuntimeError(
            "FATAL: No region features AND no cls_logits found.\n"
            "This means detection cache is corrupted or incomplete.\n"
            "Solution: Delete cache and re-run detection phase."
        )
    
    # Sanity check (첫 번째 샘플)
    if has_region_features:
        print("\n--- Sanity Check ---")
        for group_name, detections in matched_data.items():
            if isinstance(detections, list):
                for det in detections:
                    if det.region_feature is not None:
                        u_sem_calculator.sanity_check(det)
                        break
                break
    
    # Global alpha 추정 (안정적인 스케일)
    print("\n--- Estimating Global Alpha ---")
    all_dets_for_alpha = []
    for group_name, detections in matched_data.items():
        if isinstance(detections, list):
            all_dets_for_alpha.extend(detections[:500])
    global_alpha = u_sem_calculator.estimate_global_alpha(all_dets_for_alpha, n_samples=1000)
    
    # u_sem, u_sem_cond, artifactness 계산
    u_sem_by_group = {}
    u_sem_cond_by_group = {}
    s_art_by_group = {}
    drop_stats_by_group = {}
    
    for group_name, detections in matched_data.items():
        if not isinstance(detections, list) or len(detections) == 0:
            continue
        
        print(f"\n  Computing u_sem for {group_name} ({len(detections)} samples)...")
        u_sems, u_sem_conds, s_arts, drop_stats = u_sem_calculator.compute_for_detections(detections, verbose=True)
        
        # CRITICAL: valid==0이면 실패
        if drop_stats["valid"] == 0:
            raise RuntimeError(
                f"FATAL: {group_name} has 0 valid samples (all {drop_stats['total']} dropped).\n"
                f"Drop stats: {drop_stats}\n"
                "This invalidates all metrics. Fix region feature extraction first."
            )
        
        u_sem_by_group[group_name] = u_sems
        u_sem_cond_by_group[group_name] = u_sem_conds
        s_art_by_group[group_name] = s_arts
        drop_stats_by_group[group_name] = drop_stats
    
    # u_sem 통계 (기존 + 조건부)
    print(f"\n  u_sem statistics:")
    for group in u_sem_by_group.keys():
        u_sems = u_sem_by_group[group]
        u_sem_conds = u_sem_cond_by_group[group]
        valid_mask = ~np.isnan(u_sems) & (u_sems > 0)
        valid_u = u_sems[valid_mask]
        valid_u_cond = u_sem_conds[valid_mask]
        if len(valid_u) > 0:
            print(f"    {group}:")
            print(f"      u_sem:      mean={valid_u.mean():.4f}, std={valid_u.std():.4f}")
            print(f"      u_sem_cond: mean={valid_u_cond.mean():.4f}, std={valid_u_cond.std():.4f}")
        else:
            print(f"    {group}: no valid samples (total={len(u_sems)})")
    
    # 3.4 Enhanced Triad Split (artifactness score 기반)
    print("\n--- Enhanced Triad Split (artifactness-based) ---")
    enhanced_splitter = EnhancedTriadSplit(depiction_threshold=0.0)
    
    # 모든 detection과 u_sem/u_sem_cond/s_art를 합침
    all_detections = []
    all_s_arts = []
    all_u_sems = []
    all_u_sem_conds = []
    
    for group_name, detections in matched_data.items():
        if not isinstance(detections, list):
            continue
        s_arts = s_art_by_group.get(group_name, np.zeros(len(detections)))
        u_sems = u_sem_by_group.get(group_name, np.zeros(len(detections)))
        u_sem_conds = u_sem_cond_by_group.get(group_name, np.zeros(len(detections)))
        
        for det, s, u, uc in zip(detections, s_arts, u_sems, u_sem_conds):
            all_detections.append(det)
            all_s_arts.append(s)
            all_u_sems.append(u)
            all_u_sem_conds.append(uc)
    
    enhanced_groups = enhanced_splitter.split_detections(all_detections, np.array(all_s_arts))
    enhanced_splitter.print_split_stats(enhanced_groups)
    
    # 3.5 Near-miss vs Hard-negative Split (핵심!)
    print("\n--- Semantic FP Split (Near-miss vs Hard-negative) ---")
    from .semantic_uncertainty import SemanticFPSplitter
    
    semantic_splitter = SemanticFPSplitter(
        class_embeddings=u_sem_calculator.emb.base_embeddings,
        class_names=class_names,
        near_miss_threshold=0.5,   # cosine sim >= 0.5 → near-miss
        hard_negative_threshold=0.2  # cosine sim <= 0.2 → hard-negative
    )
    
    # Semantic_FP만 분리
    semantic_fp_dets = matched_data.get("Semantic_FP", [])
    semantic_fp_u_sems = u_sem_by_group.get("Semantic_FP", np.array([]))
    semantic_fp_u_sem_conds = u_sem_cond_by_group.get("Semantic_FP", np.array([]))
    
    if len(semantic_fp_dets) > 0:
        split_groups = semantic_splitter.split_semantic_fps(semantic_fp_dets, semantic_fp_u_sems)
        semantic_splitter.print_split_stats(split_groups)
        
        # Top 혼동 클래스 쌍 출력
        print("\n  Top confusion pairs:")
        confusion_pairs = semantic_splitter.get_confusion_matrix_subset(semantic_fp_dets, top_k=10)
        for pair in confusion_pairs[:5]:
            print(f"    {pair['names'][0]} vs {pair['names'][1]}: "
                  f"count={pair['count']}, sim={pair['similarity']:.3f}")
    else:
        split_groups = {}
        print("  No Semantic_FP detections to split")
    
    # Enhanced groups에서 u_sem 재구성
    enhanced_u_sem = {}
    enhanced_s_art = {}
    enhanced_detections = {}
    
    for group_name, items in enhanced_groups.items():
        if items:
            enhanced_detections[group_name] = [det for det, _ in items]
            # 해당 detection들의 u_sem, s_art 찾기
            det_set = set(id(det) for det, _ in items)
            u_sems_for_group = [u for det, u in zip(all_detections, all_u_sems) if id(det) in det_set]
            s_arts_for_group = [s for det, s in zip(all_detections, all_s_arts) if id(det) in det_set]
            enhanced_u_sem[group_name] = np.array(u_sems_for_group)
            enhanced_s_art[group_name] = np.array(s_arts_for_group)
    
    # 3.6 H1 검증 (여러 버전)
    print("\n--- H1 Verification ---")
    evaluator = H1Evaluator()
    h1_result = None
    h1_result_cond = None
    h1_result_hard_neg = None
    h1_result_depiction = None
    
    # A) 기존: TP vs Semantic_FP (u_sem)
    if "TP" in matched_data and "Semantic_FP" in matched_data:
        print("\n  [A] TP vs Semantic_FP (u_sem, 기존 정의):")
        conf_tp = np.array([d.confidence for d in matched_data["TP"]])
        conf_sem_fp = np.array([d.confidence for d in matched_data["Semantic_FP"]])
        
        h1_result = evaluator.evaluate(
            u_sem_tp=u_sem_by_group.get("TP", np.array([])),
            u_sem_semantic_fp=u_sem_by_group.get("Semantic_FP", np.array([])),
            confidence_tp=conf_tp,
            confidence_semantic_fp=conf_sem_fp,
        )
        print(format_h1_results(h1_result))
    
    # A-2) 조건부 u_sem: TP vs Semantic_FP (u_sem_cond)
    if "TP" in matched_data and "Semantic_FP" in matched_data:
        print("\n  [A-2] TP vs Semantic_FP (u_sem_cond, 조건부):")
        h1_result_cond = evaluator.evaluate(
            u_sem_tp=u_sem_cond_by_group.get("TP", np.array([])),
            u_sem_semantic_fp=u_sem_cond_by_group.get("Semantic_FP", np.array([])),
            confidence_tp=conf_tp,
            confidence_semantic_fp=conf_sem_fp,
        )
        print(format_h1_results(h1_result_cond))
    
    # A-3) Hard-negative FP만: TP vs HardNegative_FP (핵심 가설 검증)
    if len(split_groups.get("HardNegative_FP", [])) >= 10:
        print("\n  [A-3] TP vs HardNegative_FP (진짜 semantic error):")
        hard_neg_dets = [det for det, _ in split_groups["HardNegative_FP"]]
        hard_neg_u_sems = np.array([u for _, u in split_groups["HardNegative_FP"]])
        
        # TP와 비교
        tp_u = u_sem_by_group.get("TP", np.array([]))
        
        if len(tp_u) >= 10:
            h1_result_hard_neg = evaluator.evaluate(
                u_sem_tp=tp_u,
                u_sem_semantic_fp=hard_neg_u_sems,
                confidence_tp=conf_tp,
                confidence_semantic_fp=np.array([d.confidence for d in hard_neg_dets]),
            )
            print(format_h1_results(h1_result_hard_neg))
        else:
            print("      TP 샘플 부족")
    else:
        print(f"\n  [A-3] HardNegative_FP 샘플 부족 ({len(split_groups.get('HardNegative_FP', []))}개)")
    
    # B) 새로운: TP vs Depiction_FP (H1 핵심 대상)
    if "TP" in enhanced_u_sem and "Depiction_FP" in enhanced_u_sem:
        tp_u = enhanced_u_sem["TP"]
        dep_u = enhanced_u_sem["Depiction_FP"]
        
        # 유효한 샘플만
        tp_valid = tp_u[tp_u > 0]
        dep_valid = dep_u[dep_u > 0]
        
        print(f"\n  [B] TP vs Depiction_FP (H1 핵심 대상):")
        print(f"      TP: n={len(tp_valid)}, mean_u_sem={tp_valid.mean():.4f}" if len(tp_valid) > 0 else "      TP: n=0")
        print(f"      Depiction_FP: n={len(dep_valid)}, mean_u_sem={dep_valid.mean():.4f}" if len(dep_valid) > 0 else "      Depiction_FP: n=0")
        
        if len(tp_valid) >= 10 and len(dep_valid) >= 10:
            conf_tp_dep = np.array([d.confidence for d in enhanced_detections.get("TP", [])])
            conf_dep_fp = np.array([d.confidence for d in enhanced_detections.get("Depiction_FP", [])])
            
            # 유효한 것만
            conf_tp_dep = conf_tp_dep[:len(tp_valid)]
            conf_dep_fp = conf_dep_fp[:len(dep_valid)]
            
            h1_result_depiction = evaluator.evaluate(
                u_sem_tp=tp_valid,
                u_sem_semantic_fp=dep_valid,
                confidence_tp=conf_tp_dep,
                confidence_semantic_fp=conf_dep_fp,
            )
            print(format_h1_results(h1_result_depiction))
        else:
            print("      → 샘플 부족으로 AUROC 계산 불가 (최소 10개 필요)")
    else:
        print("\n  [B] Depiction_FP가 없어서 H1 핵심 검증 불가")
    
    # 3.7 결과 저장
    print("\n--- Saving Results ---")
    
    # u_sem/s_art 통계
    u_sem_stats = {}
    for group, u_sems in enhanced_u_sem.items():
        valid = u_sems[u_sems > 0]
        if len(valid) > 0:
            u_sem_stats[group] = {"mean": float(valid.mean()), "std": float(valid.std()), "n": int(len(valid))}
    
    s_art_stats = {}
    for group, s_arts in enhanced_s_art.items():
        if len(s_arts) > 0:
            s_art_stats[group] = {"mean": float(s_arts.mean()), "std": float(s_arts.std()), "n": int(len(s_arts))}
    
    # JSON 결과
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "checkpoint": config.checkpoint,
            "iou_threshold": config.iou_threshold,
            "top_m": config.top_m_classes,
            "num_views": config.num_attribute_views,
            "temperature": 10.0,
            "method": "YOLOE_region_feature + MobileCLIP_embeddings",
        },
        "detection_stats": logger.get_stats(),
        "enhanced_split": {k: len(v) for k, v in enhanced_groups.items()},
        "drop_stats": drop_stats_by_group,
        "matching_verification": verification,
        "u_sem_stats": u_sem_stats,
        "s_art_stats": s_art_stats,
    }
    
    if h1_result:
        results["h1_result_semantic_fp"] = {
            "auroc_u_sem": h1_result.auroc_u_sem,
            "aupr_u_sem": h1_result.aupr_u_sem,
            "auroc_confidence": h1_result.auroc_confidence,
            "cohens_d_u_sem": h1_result.cohens_d_u_sem,
            "n_tp": h1_result.n_tp,
            "n_semantic_fp": h1_result.n_semantic_fp,
        }
    
    if h1_result_depiction:
        results["h1_result_depiction_fp"] = {
            "auroc_u_sem": h1_result_depiction.auroc_u_sem,
            "aupr_u_sem": h1_result_depiction.aupr_u_sem,
            "auroc_confidence": h1_result_depiction.auroc_confidence,
            "cohens_d_u_sem": h1_result_depiction.cohens_d_u_sem,
            "n_tp": h1_result_depiction.n_tp,
            "n_depiction_fp": h1_result_depiction.n_semantic_fp,
        }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 시각화
    print("\n--- Generating Visualizations ---")
    try:
        # Enhanced split 기반 시각화
        confidence_data = {}
        for group_name, items in enhanced_groups.items():
            if items:
                confidence_data[group_name] = np.array([det.confidence for det, _ in items])
        
        auroc = h1_result_depiction.auroc_u_sem if h1_result_depiction else (h1_result.auroc_u_sem if h1_result else 0.5)
        
        # ROC data (TP vs Depiction_FP 우선)
        if "TP" in enhanced_u_sem and "Depiction_FP" in enhanced_u_sem:
            tp_u = enhanced_u_sem["TP"][enhanced_u_sem["TP"] > 0]
            dep_u = enhanced_u_sem["Depiction_FP"][enhanced_u_sem["Depiction_FP"] > 0]
            if len(tp_u) > 0 and len(dep_u) > 0:
                roc_data = evaluator.get_roc_curve_data(tp_u, dep_u)
            else:
                roc_data = None
        else:
            roc_data = None
        
        save_all_figures(
            output_dir / "figures",
            confidence_data,
            enhanced_u_sem,
            roc_data,
            auroc,
            logger.get_stats(),
        )
    except Exception as e:
        print(f"  WARNING: Visualization failed: {e}")
    
    print(f"\nAll results saved to {output_dir}")
    
    return h1_result_depiction if h1_result_depiction else h1_result


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
    
    # ===== fuse 자동 적용 방지 =====
    # Ultralytics predictor가 자동으로 fuse를 호출하는 것을 막음
    if hasattr(model, "overrides"):
        model.overrides["fuse"] = False
        print(f"  Disabled auto-fuse in model.overrides")
    
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "args"):
            model.predictor.args.fuse = False
            print(f"  Disabled auto-fuse in predictor.args")
    
    # 최후의 수단: fuse() 메서드 자체를 무력화
    original_fuse = getattr(model, "fuse", None)
    def noop_fuse(*args, **kwargs):
        print(f"  [BLOCKED] model.fuse() call intercepted and skipped")
        return model
    model.fuse = noop_fuse
    print(f"  Blocked model.fuse() method")
    # ================================
    
    # fuse 상태 확인 (CRITICAL)
    head = model.model.model[-1]
    
    # 진단: cv3의 마지막 레이어 weight shape으로 fused 여부 확인
    cv3_out_dim = head.cv3[0][-1].weight.shape[0] if hasattr(head, 'cv3') else -1
    embed_dim = head.embed if hasattr(head, 'embed') else 512
    
    print(f"  cv3 output dim: {cv3_out_dim}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  is_fused flag: {head.is_fused if hasattr(head, 'is_fused') else 'N/A'}")
    
    if cv3_out_dim != embed_dim:
        print(f"\n  ERROR: Model is FUSED (cv3 outputs {cv3_out_dim} dims instead of {embed_dim})")
        print(f"  512-dim region features are NOT available.")
        print(f"  u_sem calculation requires unfused model.")
        print(f"\n  SOLUTION: Use unfused checkpoint or disable fuse.")
        raise RuntimeError(
            f"Cannot run u_sem experiment: model is fused (cv3_out={cv3_out_dim}, need={embed_dim}). "
            f"Use unfused checkpoint."
        )
    else:
        print(f"  Model is NOT fused - 512-dim region features available.")
    
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
    if h1_result:
        print(f"\nKey Results:")
        print(f"  u_sem AUROC: {h1_result.auroc_u_sem:.4f}")
        print(f"  Confidence AUROC (baseline): {h1_result.auroc_confidence:.4f}")
        print(f"  Cohen's d: {h1_result.cohens_d_u_sem:.4f}")
    else:
        print("\n  H1 evaluation skipped (missing data)")
    
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
