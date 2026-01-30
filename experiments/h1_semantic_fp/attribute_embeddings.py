"""
Attribute View Embeddings Generation (Track A)
================================================
Hierarchical Attributes → u_sem (JS divergence) 계산을 위한 임베딩 생성

클래스별 K개의 attribute view 임베딩을 생성하여
view posterior 계산에 사용
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from tqdm import tqdm


@dataclass
class AttributeViewSet:
    """클래스별 attribute view 임베딩 세트"""
    class_idx: int
    class_name: str
    
    # View embeddings: [K, embed_dim]
    view_embeddings: np.ndarray
    
    # View 설명
    view_names: List[str]  # e.g., ["material", "texture", "shape", ...]
    view_prompts: List[str]  # 실제 사용된 프롬프트


@dataclass
class AttributeEmbeddingCache:
    """전체 클래스의 attribute embedding 캐시"""
    # {class_idx: AttributeViewSet}
    class_views: Dict[int, AttributeViewSet] = field(default_factory=dict)
    
    # 메타데이터
    num_views: int = 5
    embed_dim: int = 512
    text_model: str = "clip:ViT-B/32"


class AttributeEmbeddingGenerator:
    """
    Attribute view 임베딩 생성기
    
    Track A: Hierarchical Attributes
    - 슈퍼클래스 기반으로 하위 클래스에 속성 상속
    - K개의 view (material, texture, shape, context, state)
    """
    
    # View 종류별 프롬프트 템플릿
    VIEW_TEMPLATES = {
        "material": [
            "a {cls} made of metal",
            "a {cls} made of plastic",
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
    
    # 대조군용 paraphrase 템플릿
    PARAPHRASE_TEMPLATES = [
        "a photo of a {cls}",
        "an image of a {cls}",
        "a picture of a {cls}",
        "a {cls}",
        "one {cls}",
    ]
    
    def __init__(self, 
                 text_model_name: str = "clip:ViT-B/32",
                 device: str = "cuda",
                 num_views: int = 5):
        """
        Args:
            text_model_name: 텍스트 모델 이름 (e.g., "clip:ViT-B/32")
            device: 디바이스
            num_views: view 수 (K)
        """
        self.text_model_name = text_model_name
        self.device = device
        self.num_views = num_views
        
        # 텍스트 모델 로드
        self.text_model = None
        self._load_text_model()
    
    def _load_text_model(self):
        """텍스트 모델 로드"""
        from ultralytics.nn.text_model import build_text_model
        self.text_model = build_text_model(self.text_model_name, device=self.device)
        self.text_model.eval()
    
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """텍스트를 임베딩으로 인코딩"""
        tokens = self.text_model.tokenize(texts)
        embeddings = self.text_model.encode_text(tokens)
        return embeddings.cpu().numpy()
    
    def generate_class_views(self, 
                             class_name: str,
                             class_idx: int,
                             use_paraphrase: bool = False) -> AttributeViewSet:
        """
        단일 클래스의 attribute view 임베딩 생성
        
        Args:
            class_name: 클래스 이름
            class_idx: 클래스 인덱스
            use_paraphrase: True면 paraphrase 사용 (대조군)
        
        Returns:
            AttributeViewSet
        """
        # 클래스 이름 정리 (첫 번째 이름만 사용)
        clean_name = class_name.split("/")[0].strip()
        
        if use_paraphrase:
            # 대조군: 동의어 paraphrase
            templates = self.PARAPHRASE_TEMPLATES[:self.num_views]
            view_names = [f"paraphrase_{i}" for i in range(len(templates))]
        else:
            # 실제: attribute views
            view_types = list(self.VIEW_TEMPLATES.keys())[:self.num_views]
            templates = []
            view_names = []
            for vt in view_types:
                # 각 view type에서 첫 번째 템플릿 사용
                templates.append(self.VIEW_TEMPLATES[vt][0])
                view_names.append(vt)
        
        # 프롬프트 생성
        prompts = [t.format(cls=clean_name) for t in templates]
        
        # 임베딩 생성
        embeddings = self._encode_texts(prompts)  # [K, embed_dim]
        
        return AttributeViewSet(
            class_idx=class_idx,
            class_name=class_name,
            view_embeddings=embeddings,
            view_names=view_names,
            view_prompts=prompts,
        )
    
    def generate_all_class_views(self,
                                 class_names: Dict[int, str],
                                 use_paraphrase: bool = False,
                                 show_progress: bool = True) -> AttributeEmbeddingCache:
        """
        모든 클래스의 attribute view 임베딩 생성
        
        Args:
            class_names: {class_idx: class_name}
            use_paraphrase: 대조군 생성 여부
            show_progress: 진행률 표시
        
        Returns:
            AttributeEmbeddingCache
        """
        cache = AttributeEmbeddingCache(
            num_views=self.num_views,
            embed_dim=512,  # CLIP default
            text_model=self.text_model_name,
        )
        
        items = list(class_names.items())
        if show_progress:
            items = tqdm(items, desc="Generating attribute embeddings")
        
        for class_idx, class_name in items:
            view_set = self.generate_class_views(
                class_name=class_name,
                class_idx=class_idx,
                use_paraphrase=use_paraphrase,
            )
            cache.class_views[class_idx] = view_set
        
        return cache
    
    def save_cache(self, cache: AttributeEmbeddingCache, output_path: str):
        """캐시 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # NumPy 배열을 리스트로 변환하여 저장
        data = {
            "metadata": {
                "num_views": cache.num_views,
                "embed_dim": cache.embed_dim,
                "text_model": cache.text_model,
            },
            "class_views": {}
        }
        
        for idx, view_set in cache.class_views.items():
            data["class_views"][str(idx)] = {
                "class_idx": view_set.class_idx,
                "class_name": view_set.class_name,
                "view_embeddings": view_set.view_embeddings.tolist(),
                "view_names": view_set.view_names,
                "view_prompts": view_set.view_prompts,
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved attribute embeddings to {output_path}")
    
    @staticmethod
    def load_cache(path: str) -> AttributeEmbeddingCache:
        """캐시 로드"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        cache = AttributeEmbeddingCache(
            num_views=data["metadata"]["num_views"],
            embed_dim=data["metadata"]["embed_dim"],
            text_model=data["metadata"]["text_model"],
        )
        
        for idx_str, view_data in data["class_views"].items():
            cache.class_views[int(idx_str)] = AttributeViewSet(
                class_idx=view_data["class_idx"],
                class_name=view_data["class_name"],
                view_embeddings=np.array(view_data["view_embeddings"]),
                view_names=view_data["view_names"],
                view_prompts=view_data["view_prompts"],
            )
        
        return cache


def get_view_embeddings_for_class(cache: AttributeEmbeddingCache, 
                                  class_idx: int) -> Optional[np.ndarray]:
    """클래스의 view 임베딩 반환"""
    if class_idx in cache.class_views:
        return cache.class_views[class_idx].view_embeddings
    return None


def get_view_embeddings_for_classes(cache: AttributeEmbeddingCache,
                                    class_indices: List[int]) -> np.ndarray:
    """
    여러 클래스의 view 임베딩 반환
    
    Returns:
        [len(class_indices), K, embed_dim]
    """
    embeddings = []
    for idx in class_indices:
        if idx in cache.class_views:
            embeddings.append(cache.class_views[idx].view_embeddings)
        else:
            # 없는 클래스는 zero로 채움
            embeddings.append(np.zeros((cache.num_views, cache.embed_dim)))
    
    return np.stack(embeddings, axis=0)
