"""
Artifactness Score Calculation (Track B)
=========================================
MobileCLIP 기반 Visual Primitives → Artifactness score

"is it a toy?", "is it a statue?" 등의 판별 질문을 사용한
depiction/replica 탐지 점수

SemanticUncertaintyCalculator와 같은 MobileCLIP 모델 사용
→ 동일한 image embedding 재활용 가능
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass
from PIL import Image

from .detection_logger import Detection


@dataclass
class ArtifactnessPrompts:
    """Artifactness 판별용 프롬프트"""
    
    # Depiction 관련 프롬프트
    depiction_prompts: List[str] = None
    
    # Real object 프롬프트
    real_prompts: List[str] = None
    
    def __post_init__(self):
        if self.depiction_prompts is None:
            self.depiction_prompts = [
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
        
        if self.real_prompts is None:
            self.real_prompts = [
                "a real photo of the object",
                "a real object",
                "a living thing",
                "an actual object",
                "a genuine item",
                "a real animal",
                "a real person",
                "a real vehicle",
            ]


class ArtifactnessScorer:
    """
    Artifactness score 계산기 (MobileCLIP 기반)
    
    SemanticUncertaintyCalculator와 같은 MobileCLIP 사용
    → 동일한 image embedding을 재활용하여 효율적
    
    Score 계산:
    - s_art = max(<f, depiction>) - max(<f, real>)
    - 높을수록 depiction/replica일 가능성 높음
    """
    
    def __init__(self,
                 device: str = "cuda",
                 method: str = "margin"):
        """
        Args:
            device: 디바이스
            method: "max" 또는 "margin"
        """
        self.device = device
        self.method = method
        
        self.prompts = ArtifactnessPrompts()
        
        # MobileCLIP 모델
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        # 임베딩 캐시
        self.depiction_embeddings: Optional[np.ndarray] = None
        self.real_embeddings: Optional[np.ndarray] = None
        
        self._initialize_mobileclip()
    
    def _initialize_mobileclip(self):
        """MobileCLIP 모델 및 임베딩 초기화"""
        try:
            import mobileclip
            
            # MobileCLIP-B (blt) - YOLOE default
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                'mobileclip_blt', 
                pretrained='checkpoints/mobileclip_blt.pt'
            )
            self.tokenizer = mobileclip.get_tokenizer('mobileclip_blt')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"  Loaded MobileCLIP-BLT for artifactness scoring")
            
        except Exception as e:
            print(f"  WARNING: MobileCLIP load failed ({e}), falling back to CLIP")
            self._load_clip_fallback()
        
        # Text embeddings 생성
        self._build_text_embeddings()
    
    def _load_clip_fallback(self):
        """CLIP fallback"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.tokenizer = clip.tokenize
            self.model.eval()
            print(f"  Loaded CLIP ViT-B/32 (fallback)")
        except ImportError:
            print("  WARNING: Neither MobileCLIP nor CLIP available")
    
    def _build_text_embeddings(self):
        """Text embeddings 생성"""
        if self.model is None:
            return
        
        with torch.no_grad():
            # Depiction 임베딩
            dep_tokens = self.tokenizer(self.prompts.depiction_prompts).to(self.device)
            dep_feats = self.model.encode_text(dep_tokens)
            dep_feats = F.normalize(dep_feats, dim=-1)
            self.depiction_embeddings = dep_feats.cpu().numpy()
            
            # Real 임베딩
            real_tokens = self.tokenizer(self.prompts.real_prompts).to(self.device)
            real_feats = self.model.encode_text(real_tokens)
            real_feats = F.normalize(real_feats, dim=-1)
            self.real_embeddings = real_feats.cpu().numpy()
        
        print(f"  Built {len(self.prompts.depiction_prompts)} depiction + {len(self.prompts.real_prompts)} real embeddings")
    
    def compute_score(self, image_embedding: np.ndarray) -> float:
        """
        Image embedding의 artifactness score 계산
        
        Args:
            image_embedding: [embed_dim] - 이미 정규화된 embedding
        
        Returns:
            Artifactness score
        """
        if self.depiction_embeddings is None or self.real_embeddings is None:
            return 0.0
        
        # 정규화 확인
        f_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-10)
        
        # Depiction 유사도
        dep_sims = np.dot(self.depiction_embeddings, f_norm)
        max_dep_sim = np.max(dep_sims)
        
        if self.method == "max":
            # s_art = max_j <f, T(depiction_j)>
            return float(max_dep_sim)
        
        elif self.method == "margin":
            # s_art = max_j <f, T(depiction_j)> - max_k <f, T(real_k)>
            # 높을수록 depiction일 가능성 높음
            real_sims = np.dot(self.real_embeddings, f_norm)
            max_real_sim = np.max(real_sims)
            return float(max_dep_sim - max_real_sim)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_detailed_scores(self, image_embedding: np.ndarray) -> Dict:
        """
        상세 점수 계산 (분석용)
        
        Returns:
            각 프롬프트별 유사도 딕셔너리
        """
        f_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-10)
        
        dep_sims = np.dot(self.depiction_embeddings, f_norm)
        real_sims = np.dot(self.real_embeddings, f_norm)
        
        return {
            "depiction_scores": {
                prompt: float(sim) 
                for prompt, sim in zip(self.prompts.depiction_prompts, dep_sims)
            },
            "real_scores": {
                prompt: float(sim)
                for prompt, sim in zip(self.prompts.real_prompts, real_sims)
            },
            "max_depiction": float(np.max(dep_sims)),
            "max_real": float(np.max(real_sims)),
            "margin": float(np.max(dep_sims) - np.max(real_sims)),
        }
    
    def compute_for_detection(self, detection: Detection, 
                               image_emb: Optional[np.ndarray] = None) -> float:
        """
        Detection의 artifactness score 계산
        
        Args:
            detection: Detection 객체
            image_emb: 미리 계산된 image embedding (u_sem에서 재활용)
        """
        # image_emb가 제공되면 사용 (효율적)
        if image_emb is not None:
            return self.compute_score(image_emb)
        
        # 없으면 직접 계산
        if detection.image_path is None or self.model is None:
            return 0.0
        
        try:
            img = Image.open(detection.image_path).convert('RGB')
            
            # Crop
            x1, y1, x2, y2 = map(int, detection.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            crop = img.crop((x1, y1, x2, y2))
            
            # MobileCLIP encoding
            with torch.no_grad():
                crop_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
                features = self.model.encode_image(crop_tensor)
                features = F.normalize(features, dim=-1)
                image_emb = features.cpu().numpy().squeeze()
            
            return self.compute_score(image_emb)
            
        except Exception as e:
            return 0.0
    
    def compute_for_detections(self, detections: List[Detection], 
                                image_embs: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        여러 detection의 artifactness score 일괄 계산
        
        Args:
            detections: Detection 리스트
            image_embs: 미리 계산된 image embeddings (u_sem에서 재활용)
        """
        scores = []
        for i, det in enumerate(detections):
            emb = image_embs[i] if image_embs is not None else None
            scores.append(self.compute_for_detection(det, emb))
        return np.array(scores)
    
    def compute_for_groups(self,
                           groups: Dict[str, List[Detection]],
                           group_embeddings: Optional[Dict[str, List[np.ndarray]]] = None
                           ) -> Dict[str, np.ndarray]:
        """
        그룹별 artifactness score 계산
        
        Args:
            groups: {group_name: [Detection, ...]}
            group_embeddings: {group_name: [image_emb, ...]}
        
        Returns:
            {group_name: np.ndarray of scores}
        """
        results = {}
        for group_name, dets in groups.items():
            embs = group_embeddings.get(group_name) if group_embeddings else None
            results[group_name] = self.compute_for_detections(dets, embs)
            print(f"    {group_name}: {len(dets)} detections, "
                  f"mean s_art={results[group_name].mean():.4f}")
        return results
