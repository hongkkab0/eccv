import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from tqdm import tqdm
import os
from ultralytics.nn.text_model import build_text_model

@smart_inference_mode()
def generate_label_embedding(model, texts, batch=512):
    model = build_text_model(model, device='cuda')
    assert(not model.training)
    
    text_tokens = model.tokenize(texts)
    txt_feats = []
    for text_token in tqdm(text_tokens.split(batch)):
        txt_feats.append(model.encode_text(text_token))
    txt_feats = torch.cat(txt_feats, dim=0)
    return txt_feats.cpu()


def collect_grounding_labels(cache_path):
    labels = np.load(cache_path, allow_pickle=True)
    cat_names = set()
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_names.add(t)
    
    return cat_names

def collect_detection_labels(yaml_path):
    cat_names = set()
    
    data = yaml_load(yaml_path, append_filename=True)
    names = [name.split("/") for name in data["names"].values()]
    for name in names:
        for n in name:
            n = n.strip()
            assert(n)
            cat_names.add(n)
    
    return cat_names

ATTRIBUTE_TEMPLATES = {
    "material": [
        "a {cls} made of metal",
        "a {cls} made of plastic",
        "a {cls} made of glass",
        "a {cls} made of wood",
        "a {cls} made of fabric",
        "a {cls} made of organic material",
    ],
    "texture": [
        "a smooth {cls}",
        "a rough {cls}",
        "a shiny {cls}",
        "a fuzzy {cls}",
        "a patterned {cls}",
    ],
    "shape": [
        "a round {cls}",
        "a rectangular {cls}",
        "a elongated {cls}",
        "a compact {cls}",
        "an irregularly shaped {cls}",
    ],
    "context": [
        "a {cls} in indoor setting",
        "a {cls} in outdoor setting",
        "a {cls} in natural environment",
        "a {cls} in urban environment",
        "a {cls} in domestic setting",
    ],
    "state": [
        "a new {cls}",
        "a well-used {cls}",
        "a clean {cls}",
        "a weathered {cls}",
        "a partially visible {cls}",
    ],
}

PARAPHRASE_TEMPLATES = [
    "a photo of a {cls}",
    "an image of a {cls}",
    "a picture of a {cls}",
    "a {cls}",
    "one {cls}",
]

# Artifactness 판별용 프롬프트
DEPICTION_PROMPTS = [
    "a toy",
    "a toy version",
    "a statue",
    "a sculpture", 
    "a poster",
    "a painting",
    "a drawing",
    "a figurine",
    "a doll",
    "a puppet",
    "a cartoon character",
    "a stuffed animal",
    "a plush toy",
    "a mannequin",
    "a model replica",
    "a printed image",
    "a 2D depiction",
    "an artificial representation",
]

REAL_PROMPTS = [
    "a real photo of the object",
    "a real object",
    "a living thing",
    "an actual object",
    "a genuine item",
    "a real animal",
    "a real person",
    "a real vehicle",
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvis', action='store_true', help='Generate LVIS label embeddings only')
    parser.add_argument('--lvis-all', action='store_true', help='Generate LVIS + attribute template embeddings')
    args = parser.parse_args()
    
    os.environ["PYTHONHASHSEED"] = "0"
    model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    save_dir = f'tools/{model.replace(":", "_")}'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.lvis or args.lvis_all:
        # LVIS 기본 임베딩 생성
        lvis_yaml = 'ultralytics/cfg/datasets/lvis.yaml'
        all_cat_names = list(collect_detection_labels(lvis_yaml))
        print(f"Generating embeddings for {len(all_cat_names)} LVIS classes...")
        
        all_cat_feats = generate_label_embedding(model, all_cat_names)
        
        cat_name_feat_map = {}
        for name, feat in zip(all_cat_names, all_cat_feats):
            cat_name_feat_map[name] = feat
        
        save_path = f'{save_dir}/lvis_label_embeddings.pt'
        torch.save(cat_name_feat_map, save_path)
        print(f"Saved to {save_path}")
    
    if args.lvis_all:
        # Attribute 템플릿 임베딩도 생성
        lvis_yaml = 'ultralytics/cfg/datasets/lvis.yaml'
        all_cat_names = list(collect_detection_labels(lvis_yaml))
        
        print(f"\nGenerating attribute template embeddings...")
        
        # 모든 템플릿 텍스트 수집
        all_template_texts = []
        template_keys = []  # (template_type, template, class_name)
        
        # Attribute templates
        for attr_type, templates in ATTRIBUTE_TEMPLATES.items():
            for template in templates:
                for cls_name in all_cat_names:
                    text = template.format(cls=cls_name)
                    all_template_texts.append(text)
                    template_keys.append((attr_type, template, cls_name))
        
        # Paraphrase templates
        for template in PARAPHRASE_TEMPLATES:
            for cls_name in all_cat_names:
                text = template.format(cls=cls_name)
                all_template_texts.append(text)
                template_keys.append(("paraphrase", template, cls_name))
        
        print(f"Total template texts: {len(all_template_texts)}")
        
        # 임베딩 생성
        all_template_feats = generate_label_embedding(model, all_template_texts)
        
        # 저장 형식: {text: embedding}
        template_emb_map = {}
        for text, feat in zip(all_template_texts, all_template_feats):
            template_emb_map[text] = feat
        
        save_path = f'{save_dir}/lvis_attribute_embeddings.pt'
        torch.save(template_emb_map, save_path)
        print(f"Saved {len(template_emb_map)} embeddings to {save_path}")
        
        # Artifactness 프롬프트 임베딩도 생성
        print(f"\nGenerating artifactness prompt embeddings...")
        artifactness_texts = DEPICTION_PROMPTS + REAL_PROMPTS
        artifactness_feats = generate_label_embedding(model, artifactness_texts)
        
        artifactness_emb_map = {}
        for text, feat in zip(artifactness_texts, artifactness_feats):
            artifactness_emb_map[text] = feat
        
        save_path = f'{save_dir}/artifactness_embeddings.pt'
        torch.save(artifactness_emb_map, save_path)
        print(f"Saved {len(artifactness_emb_map)} embeddings to {save_path}")
    
    if not args.lvis and not args.lvis_all:
        # 기존 로직 (train용)
        flickr_cache = '../datasets/flickr/annotations/final_flickr_separateGT_train_segm.cache'
        mixed_grounding_cache = '../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.cache'
        objects365v1_yaml = 'ultralytics/cfg/datasets/Objects365v1.yaml'
        
        all_cat_names = set()
        all_cat_names |= collect_detection_labels(objects365v1_yaml)
        all_cat_names |= collect_grounding_labels(flickr_cache)
        all_cat_names |= collect_grounding_labels(mixed_grounding_cache)
        
        all_cat_names = list(all_cat_names)
        
        all_cat_feats = generate_label_embedding(model, all_cat_names)
        
        cat_name_feat_map = {}
        for name, feat in zip(all_cat_names, all_cat_feats):
            cat_name_feat_map[name] = feat
        
        os.makedirs(f'tools/{model}', exist_ok=True)
        torch.save(cat_name_feat_map, f'tools/{model}/train_label_embeddings.pt')
