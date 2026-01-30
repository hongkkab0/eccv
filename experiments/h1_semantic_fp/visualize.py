"""
Visualization Tools for H1 Experiment
======================================
실험 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_plot_style():
    """논문용 플롯 스타일 설정"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_confidence_histogram(confidence_data: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> Figure:
    """
    Confidence 분포 히스토그램 (3그룹 비교)
    
    Fig.1 핵심: confidence-matched 상태를 보여주는 그림
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "TP": "#2ecc71",  # green
        "Semantic_FP": "#e74c3c",  # red
        "Background_FP": "#3498db",  # blue
    }
    
    labels = {
        "TP": "True Positive",
        "Semantic_FP": "Semantic False Positive",
        "Background_FP": "Background False Positive",
    }
    
    bins = np.linspace(0.75, 1.0, 26)
    
    for group, data in confidence_data.items():
        if len(data) > 0:
            ax.hist(data, bins=bins, alpha=0.5, label=labels.get(group, group),
                   color=colors.get(group, 'gray'), density=True)
    
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution (Matched)")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 통계 텍스트
    stats_text = []
    for group, data in confidence_data.items():
        if len(data) > 0:
            stats_text.append(f"{group}: μ={np.mean(data):.3f}, σ={np.std(data):.3f}")
    ax.text(0.95, 0.95, "\n".join(stats_text), transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved confidence histogram to {save_path}")
    
    return fig


def plot_u_sem_distribution(u_sem_data: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> Figure:
    """
    u_sem 분포 비교 (TP vs Semantic FP)
    
    핵심 그림: Semantic FP가 더 높은 u_sem을 가짐을 보여줌
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "TP": "#2ecc71",
        "Semantic_FP": "#e74c3c",
        "Background_FP": "#3498db",
    }
    
    # 히스토그램
    ax1 = axes[0]
    for group in ["TP", "Semantic_FP"]:
        if group in u_sem_data and len(u_sem_data[group]) > 0:
            ax1.hist(u_sem_data[group], bins=30, alpha=0.6, 
                    label=group.replace("_", " "), color=colors[group], density=True)
    
    ax1.set_xlabel("$u_{sem}$ (JS Divergence)")
    ax1.set_ylabel("Density")
    ax1.set_title("Semantic Uncertainty Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    box_data = []
    box_labels = []
    box_colors = []
    
    for group in ["TP", "Semantic_FP", "Background_FP"]:
        if group in u_sem_data and len(u_sem_data[group]) > 0:
            box_data.append(u_sem_data[group])
            box_labels.append(group.replace("_", " "))
            box_colors.append(colors[group])
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel("$u_{sem}$")
    ax2.set_title("$u_{sem}$ by Detection Type")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved u_sem distribution to {save_path}")
    
    return fig


def plot_roc_curve(roc_data: Dict,
                   auroc: float,
                   save_path: Optional[str] = None) -> Figure:
    """
    ROC Curve 시각화
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(roc_data["fpr"], roc_data["tpr"], 
            color='#e74c3c', lw=2, label=f'$u_{{sem}}$ (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: TP vs Semantic FP Classification')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved ROC curve to {save_path}")
    
    return fig


def plot_correlation_scatter(u_sem: np.ndarray,
                             violations: np.ndarray,
                             spearman_rho: float,
                             p_value: float,
                             save_path: Optional[str] = None) -> Figure:
    """
    u_sem vs Human Violation Count 산점도
    
    H1 직접 검증: u_sem이 높을수록 violation이 많다
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(u_sem, violations, alpha=0.5, c='#e74c3c', s=50)
    
    # 추세선
    z = np.polyfit(u_sem, violations, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(u_sem), max(u_sem), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', lw=2, alpha=0.8)
    
    ax.set_xlabel("$u_{sem}$ (JS Divergence)")
    ax.set_ylabel("Human Violation Count")
    ax.set_title("Correlation: $u_{sem}$ vs Attribute Violations")
    
    # 통계 표시
    ax.text(0.05, 0.95, f"Spearman ρ = {spearman_rho:.3f}\np = {p_value:.2e}",
            transform=ax.transAxes, verticalalignment='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved correlation scatter to {save_path}")
    
    return fig


def plot_comparison_bar(metrics: Dict[str, float],
                        baseline_metrics: Dict[str, float],
                        save_path: Optional[str] = None) -> Figure:
    """
    u_sem vs Baseline 메트릭 비교 막대 그래프
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [metrics[m] for m in metric_names], 
                   width, label='$u_{sem}$', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, [baseline_metrics.get(m, 0) for m in metric_names],
                   width, label='Confidence (baseline)', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: $u_{sem}$ vs Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved comparison bar to {save_path}")
    
    return fig


def plot_triad_split_pie(stats: Dict[str, int],
                         save_path: Optional[str] = None) -> Figure:
    """
    Triad Split 분포 파이 차트
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['True Positive', 'Semantic FP', 'Background FP']
    sizes = [stats.get('tp_count', 0), 
             stats.get('semantic_fp_count', 0),
             stats.get('background_fp_count', 0)]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    explode = (0, 0.05, 0)  # Semantic FP 강조
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('Detection Triad Split Distribution')
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved triad split pie to {save_path}")
    
    return fig


def create_summary_figure(confidence_data: Dict[str, np.ndarray],
                          u_sem_data: Dict[str, np.ndarray],
                          roc_data: Dict,
                          auroc: float,
                          stats: Dict,
                          save_path: Optional[str] = None) -> Figure:
    """
    논문용 종합 Figure 생성 (2x2 레이아웃)
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {
        "TP": "#2ecc71",
        "Semantic_FP": "#e74c3c",
        "Background_FP": "#3498db",
    }
    
    # (a) Confidence histogram - confidence matching 증명
    ax = axes[0, 0]
    bins = np.linspace(0.75, 1.0, 26)
    for group in ["TP", "Semantic_FP"]:
        if group in confidence_data and len(confidence_data[group]) > 0:
            ax.hist(confidence_data[group], bins=bins, alpha=0.5, 
                   label=group.replace("_", " "), color=colors[group], density=True)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("(a) Confidence Distribution (Matched)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) u_sem distribution - 분리 가능성
    ax = axes[0, 1]
    for group in ["TP", "Semantic_FP"]:
        if group in u_sem_data and len(u_sem_data[group]) > 0:
            ax.hist(u_sem_data[group], bins=30, alpha=0.5, 
                   label=group.replace("_", " "), color=colors[group], density=True)
    ax.set_xlabel("$u_{sem}$")
    ax.set_ylabel("Density")
    ax.set_title("(b) Semantic Uncertainty Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) ROC curve
    ax = axes[1, 0]
    ax.plot(roc_data["fpr"], roc_data["tpr"], 
            color='#e74c3c', lw=2, label=f'$u_{{sem}}$ (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('(c) ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # (d) Box plot comparison
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    box_colors_list = []
    for group in ["TP", "Semantic_FP", "Background_FP"]:
        if group in u_sem_data and len(u_sem_data[group]) > 0:
            box_data.append(u_sem_data[group])
            box_labels.append(group.replace("_", "\n"))
            box_colors_list.append(colors[group])
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("$u_{sem}$")
    ax.set_title("(d) $u_{sem}$ by Detection Type")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved summary figure to {save_path}")
    
    return fig


def save_all_figures(output_dir: str,
                     confidence_data: Dict[str, np.ndarray],
                     u_sem_data: Dict[str, np.ndarray],
                     roc_data: Dict,
                     auroc: float,
                     stats: Dict):
    """모든 Figure 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confidence_histogram(confidence_data, output_dir / "confidence_histogram.png")
    plot_u_sem_distribution(u_sem_data, output_dir / "u_sem_distribution.png")
    plot_roc_curve(roc_data, auroc, output_dir / "roc_curve.png")
    plot_triad_split_pie(stats, output_dir / "triad_split.png")
    create_summary_figure(confidence_data, u_sem_data, roc_data, auroc, stats,
                         output_dir / "summary_figure.png")
    
    print(f"\nAll figures saved to {output_dir}")
