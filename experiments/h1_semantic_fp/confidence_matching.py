"""
Confidence-Matched Sampling
============================
세 그룹(TP, Semantic FP, Background FP)의 confidence 분포를 동일하게 매칭

"confidence로는 구분 불가"를 증명하기 위한 핵심 모듈
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

from .detection_logger import Detection


@dataclass
class ConfidenceMatchedSample:
    """Confidence-matched 샘플"""
    bin_start: float
    bin_end: float
    
    tp_samples: List[Detection]
    semantic_fp_samples: List[Detection]
    background_fp_samples: List[Detection]
    
    # 샘플 수
    samples_per_group: int


class ConfidenceMatchedSampler:
    """
    Confidence-matched 샘플링 수행
    
    각 confidence bin에서 TP/Semantic FP/Background FP를 동일 개수로 샘플링하여
    세 그룹의 confidence 분포가 동일하게 만듦
    """
    
    def __init__(self,
                 bin_start: float = 0.80,
                 bin_end: float = 0.95,
                 bin_step: float = 0.02,
                 samples_per_bin: int = 50,
                 seed: int = 42):
        """
        Args:
            bin_start: confidence bin 시작값
            bin_end: confidence bin 끝값
            bin_step: bin 간격
            samples_per_bin: 각 bin에서 그룹당 샘플링 수
            seed: 랜덤 시드
        """
        self.bin_start = bin_start
        self.bin_end = bin_end
        self.bin_step = bin_step
        self.samples_per_bin = samples_per_bin
        self.seed = seed
        
        # Bins 생성
        self.bins = self._create_bins()
        
        random.seed(seed)
        np.random.seed(seed)
    
    def _create_bins(self) -> List[Tuple[float, float]]:
        """Confidence bins 생성"""
        bins = []
        current = self.bin_start
        while current < self.bin_end:
            bins.append((current, min(current + self.bin_step, self.bin_end)))
            current += self.bin_step
        return bins
    
    def _assign_to_bins(self, detections: List[Detection]) -> Dict[Tuple[float, float], List[Detection]]:
        """Detection들을 confidence bin에 할당"""
        binned = defaultdict(list)
        
        for det in detections:
            for bin_start, bin_end in self.bins:
                if bin_start <= det.confidence < bin_end:
                    binned[(bin_start, bin_end)].append(det)
                    break
        
        return binned
    
    def sample(self, 
               triad_split: Dict[str, List[Detection]],
               min_samples_per_bin: int = 10) -> List[ConfidenceMatchedSample]:
        """
        Confidence-matched 샘플링 수행
        
        Args:
            triad_split: {"TP": [...], "Semantic_FP": [...], "Background_FP": [...]}
            min_samples_per_bin: bin당 최소 샘플 수 (이보다 적으면 해당 bin 제외)
        
        Returns:
            ConfidenceMatchedSample 리스트
        """
        # 각 그룹을 bin에 할당
        tp_binned = self._assign_to_bins(triad_split["TP"])
        sem_fp_binned = self._assign_to_bins(triad_split["Semantic_FP"])
        bg_fp_binned = self._assign_to_bins(triad_split["Background_FP"])
        
        matched_samples = []
        
        for bin_range in self.bins:
            tp_in_bin = tp_binned.get(bin_range, [])
            sem_fp_in_bin = sem_fp_binned.get(bin_range, [])
            bg_fp_in_bin = bg_fp_binned.get(bin_range, [])
            
            # 모든 그룹에서 최소 samples 이상 있어야 함
            min_available = min(len(tp_in_bin), len(sem_fp_in_bin), len(bg_fp_in_bin))
            
            if min_available < min_samples_per_bin:
                continue
            
            # 샘플링 수 결정 (요청 수와 가용 수 중 작은 값)
            n_samples = min(self.samples_per_bin, min_available)
            
            # 각 그룹에서 랜덤 샘플링
            tp_sampled = random.sample(tp_in_bin, n_samples)
            sem_fp_sampled = random.sample(sem_fp_in_bin, n_samples)
            bg_fp_sampled = random.sample(bg_fp_in_bin, n_samples)
            
            matched_samples.append(ConfidenceMatchedSample(
                bin_start=bin_range[0],
                bin_end=bin_range[1],
                tp_samples=tp_sampled,
                semantic_fp_samples=sem_fp_sampled,
                background_fp_samples=bg_fp_sampled,
                samples_per_group=n_samples,
            ))
        
        return matched_samples
    
    def get_pooled_samples(self, 
                           matched_samples: List[ConfidenceMatchedSample]
                           ) -> Dict[str, List[Detection]]:
        """
        모든 bin의 샘플을 합쳐서 반환
        
        Returns:
            {"TP": [...], "Semantic_FP": [...], "Background_FP": [...]}
        """
        pooled = {
            "TP": [],
            "Semantic_FP": [],
            "Background_FP": [],
        }
        
        for sample in matched_samples:
            pooled["TP"].extend(sample.tp_samples)
            pooled["Semantic_FP"].extend(sample.semantic_fp_samples)
            pooled["Background_FP"].extend(sample.background_fp_samples)
        
        return pooled
    
    def verify_matching(self, 
                        matched_samples: List[ConfidenceMatchedSample]) -> Dict:
        """
        Confidence matching이 잘 되었는지 검증
        
        Returns:
            검증 결과 딕셔너리
        """
        pooled = self.get_pooled_samples(matched_samples)
        
        # 각 그룹의 confidence 분포 통계
        stats = {}
        for group_name, detections in pooled.items():
            confs = np.array([d.confidence for d in detections])
            stats[group_name] = {
                "count": len(detections),
                "mean": float(np.mean(confs)) if len(confs) > 0 else 0,
                "std": float(np.std(confs)) if len(confs) > 0 else 0,
                "min": float(np.min(confs)) if len(confs) > 0 else 0,
                "max": float(np.max(confs)) if len(confs) > 0 else 0,
                "median": float(np.median(confs)) if len(confs) > 0 else 0,
            }
        
        # 그룹 간 분포 유사도 (mean 차이)
        means = [stats[g]["mean"] for g in ["TP", "Semantic_FP", "Background_FP"]]
        max_mean_diff = max(means) - min(means) if means else 0
        
        return {
            "per_group_stats": stats,
            "max_mean_difference": max_mean_diff,
            "is_well_matched": max_mean_diff < 0.02,  # mean 차이가 0.02 미만이면 OK
            "total_samples": sum(s["count"] for s in stats.values()),
            "num_bins_used": len(matched_samples),
        }
    
    def get_confidence_histogram_data(self,
                                      matched_samples: List[ConfidenceMatchedSample]
                                      ) -> Dict[str, np.ndarray]:
        """
        시각화용 히스토그램 데이터 반환
        
        Returns:
            {"TP": confidence array, "Semantic_FP": ..., "Background_FP": ...}
        """
        pooled = self.get_pooled_samples(matched_samples)
        
        return {
            group: np.array([d.confidence for d in dets])
            for group, dets in pooled.items()
        }


def create_confidence_matched_dataset(
    triad_split: Dict[str, List[Detection]],
    config: Optional[Dict] = None
) -> Tuple[Dict[str, List[Detection]], Dict]:
    """
    Convenience function: confidence-matched 데이터셋 생성
    
    Args:
        triad_split: Triad split 결과
        config: 설정 (없으면 기본값 사용)
    
    Returns:
        (matched_data, verification_stats)
    """
    if config is None:
        config = {
            "bin_start": 0.80,
            "bin_end": 0.95,
            "bin_step": 0.02,
            "samples_per_bin": 50,
            "seed": 42,
        }
    
    sampler = ConfidenceMatchedSampler(**config)
    matched_samples = sampler.sample(triad_split)
    pooled = sampler.get_pooled_samples(matched_samples)
    verification = sampler.verify_matching(matched_samples)
    
    return pooled, verification
