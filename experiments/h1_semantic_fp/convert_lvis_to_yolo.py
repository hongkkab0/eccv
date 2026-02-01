"""
LVIS JSON annotation -> YOLO format 변환 스크립트
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse


def convert_lvis_to_yolo(lvis_json_path: str, output_dir: str, split: str = "val"):
    """
    LVIS JSON annotation을 YOLO 포맷 라벨로 변환
    
    Args:
        lvis_json_path: LVIS annotation JSON 경로 (예: lvis_v1_val.json)
        output_dir: 출력 디렉토리 (예: /datasets/lvis/labels/val2017)
        split: 'train' or 'val'
    """
    print(f"Loading LVIS annotation: {lvis_json_path}")
    with open(lvis_json_path, 'r') as f:
        lvis_data = json.load(f)
    
    # 이미지 정보 (id -> info)
    images = {img['id']: img for img in lvis_data['images']}
    print(f"Total images: {len(images)}")
    
    # 카테고리 정보 (LVIS는 1-indexed, 1203 classes)
    # LVIS category_id는 연속적이지 않을 수 있음 -> 연속 인덱스로 매핑 필요
    categories = {cat['id']: cat for cat in lvis_data['categories']}
    
    # category_id -> 0-indexed class_id 매핑
    sorted_cat_ids = sorted(categories.keys())
    cat_id_to_class_id = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    
    print(f"Total categories: {len(categories)}")
    print(f"Category ID range: {min(sorted_cat_ids)} - {max(sorted_cat_ids)}")
    
    # 이미지별 annotation 그룹화
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    print(f"Images with annotations: {len(img_to_anns)}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 각 이미지에 대해 YOLO 포맷 라벨 생성
    label_count = 0
    for img_id, img_info in tqdm(images.items(), desc="Converting"):
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 파일명에서 확장자 제거하고 .txt로
        # LVIS/COCO 이미지 파일명: 000000000139.jpg -> 000000000139.txt
        if 'file_name' in img_info:
            # COCO 형식: val2017/000000000139.jpg or 000000000139.jpg
            file_name = Path(img_info['file_name']).stem
        else:
            # fallback
            file_name = str(img_id).zfill(12)
        
        label_file = output_path / f"{file_name}.txt"
        
        anns = img_to_anns.get(img_id, [])
        
        lines = []
        for ann in anns:
            # bbox: [x, y, width, height] in absolute pixels (COCO format)
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # YOLO format: class_id x_center y_center width height (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # 경계 체크
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            
            # class_id (0-indexed)
            class_id = cat_id_to_class_id[ann['category_id']]
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 라벨 파일 작성 (annotation이 없어도 빈 파일 생성)
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))
        
        if lines:
            label_count += 1
    
    print(f"\nConversion complete!")
    print(f"Total label files created: {len(images)}")
    print(f"Label files with annotations: {label_count}")
    print(f"Output directory: {output_path}")
    
    # 클래스 이름 파일도 생성 (확인용)
    names_file = output_path.parent / f"classes_{split}.txt"
    with open(names_file, 'w') as f:
        for cat_id in sorted_cat_ids:
            f.write(f"{categories[cat_id]['name']}\n")
    print(f"Class names saved to: {names_file}")
    
    return cat_id_to_class_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LVIS JSON to YOLO format")
    parser.add_argument("--lvis-json", type=str, required=True,
                        help="Path to LVIS annotation JSON (e.g., lvis_v1_val.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for YOLO labels (e.g., /datasets/lvis/labels/val2017)")
    parser.add_argument("--split", type=str, default="val",
                        help="Split name (train/val)")
    
    args = parser.parse_args()
    
    convert_lvis_to_yolo(args.lvis_json, args.output_dir, args.split)
