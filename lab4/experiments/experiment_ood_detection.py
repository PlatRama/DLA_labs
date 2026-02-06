"""
Exercise 1: OOD Detection Pipeline
Detect out-of-distribution samples using different scoring methods.
"""
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from lab4.configs.config import OODDetectionConfig
from lab4.models.ood_models import SimpleCNN
from lab4.utils.data_utils import create_id_dataloaders, create_ood_dataloader
from lab4.utils.ood_metrics import (
    compute_ood_scores,
    compute_ood_metrics,
    find_optimal_threshold
)
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device
from utils_for_all.checkpoint import load_checkpoint


def plot_score_distributions(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    score_method: str,
    save_path: str = None
):
    """
    Plot ID vs OOD score distributions.
    
    Args:
        id_scores: ID scores
        ood_scores: OOD scores
        score_method: Scoring method name
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(id_scores.cpu().numpy(), bins=50, alpha=0.6, label='ID', density=True)
    axes[0].hist(ood_scores.cpu().numpy(), bins=50, alpha=0.6, label='OOD', density=True)
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Score Distribution - {score_method}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sorted scores
    axes[1].plot(sorted(id_scores.cpu().numpy()), label='ID', alpha=0.7)
    axes[1].plot(sorted(ood_scores.cpu().numpy()), label='OOD', alpha=0.7)
    axes[1].set_xlabel('Sample Index (sorted)')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Sorted Scores - {score_method}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_roc_pr_curves(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    save_path: str = None
):
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        id_scores: ID scores
        ood_scores: OOD scores
        save_path: Path to save plot
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    # Prepare data (ID=1, OOD=0)
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    
    # Compute curves
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    axes[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    axes[1].plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})', linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def run_ood_detection(
    classifier_checkpoint: str = './checkpoints/lab4/classifier_baseline/best.pt',
    ood_dataset: str = 'cifar100_subset',
    score_method: str = 'max_softmax',
    temperature: float = 1.0,
    plot: bool = True
):
    """
    Run OOD detection experiment.
    
    Args:
        classifier_checkpoint: Path to classifier checkpoint
        ood_dataset: OOD dataset type
        score_method: Scoring method
        temperature: Temperature for softmax
        plot: Whether to plot results
    """
    # Create configuration
    config = OODDetectionConfig(
        classifier_checkpoint=classifier_checkpoint,
        ood_dataset=ood_dataset,
        score_method=score_method,
        temperature=temperature,
        batch_size=128,
        experiment_name=f'ood_{ood_dataset}_{score_method}',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('ood_detection', log_file=f'./logs/lab4/ood_detection.log')
    
    logger.info("=" * 80)
    logger.info(f"OOD Detection Experiment")
    logger.info(f"OOD Dataset: {ood_dataset}")
    logger.info(f"Score Method: {score_method}")
    logger.info("=" * 80)
    
    # Load classifier
    logger.info(f"Loading classifier from {classifier_checkpoint}")
    model = SimpleCNN(num_classes=10)
    load_checkpoint(
        classifier_checkpoint,
        model=model,
        device=device,
        strict=True
    )
    model = model.to(device)
    model.eval()
    
    # Create dataloaders
    _, _, id_test_loader = create_id_dataloaders(
        dataset='cifar10',
        batch_size=config.batch_size,
        num_workers=4,
        augment=False,
        root='./data'
    )
    
    ood_loader = create_ood_dataloader(
        ood_type=ood_dataset,
        id_dataset='cifar10',
        batch_size=config.batch_size,
        num_workers=4,
        num_samples=10000,
        root='./data'
    )
    
    # Compute scores
    logger.info(f"Computing ID scores...")
    id_scores = compute_ood_scores(
        model=model,
        dataloader=id_test_loader,
        score_method=score_method,
        temperature=temperature,
        device=device
    )
    
    logger.info(f"Computing OOD scores...")
    ood_scores = compute_ood_scores(
        model=model,
        dataloader=ood_loader,
        score_method=score_method,
        temperature=temperature,
        device=device
    )
    
    logger.info(f"\nScore Statistics:")
    logger.info(f"  ID scores: mean={id_scores.mean():.4f}, std={id_scores.std():.4f}")
    logger.info(f"  OOD scores: mean={ood_scores.mean():.4f}, std={ood_scores.std():.4f}")
    
    # Compute metrics
    metrics = compute_ood_metrics(
        id_scores=id_scores,
        ood_scores=ood_scores,
        fpr_threshold=0.95
    )
    
    # Find optimal threshold
    threshold, accuracy = find_optimal_threshold(id_scores, ood_scores)
    logger.info(f"\nOptimal Threshold: {threshold:.4f}")
    logger.info(f"Classification Accuracy: {accuracy * 100:.2f}%")
    
    # Plot results
    if plot:
        plot_score_distributions(
            id_scores, ood_scores, score_method,
            save_path=f'./plots/score_dist_{ood_dataset}_{score_method}.png'
        )
        
        plot_roc_pr_curves(
            id_scores, ood_scores,
            save_path=f'./plots/roc_pr_{ood_dataset}_{score_method}.png'
        )
    
    logger.info("=" * 80)
    
    return metrics


def compare_ood_methods():
    """
    Compare different OOD detection methods.
    """
    methods = ['max_softmax', 'max_logit', 'energy']
    ood_datasets = ['cifar100_subset', 'fakedata']
    
    results = {}
    
    for ood_dataset in ood_datasets:
        results[ood_dataset] = {}
        
        print(f"\n{'='*80}")
        print(f"OOD Dataset: {ood_dataset}")
        print(f"{'='*80}\n")
        
        for method in methods:
            print(f"Testing {method}...")
            metrics = run_ood_detection(
                ood_dataset=ood_dataset,
                score_method=method,
                plot=False
            )
            results[ood_dataset][method] = metrics
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("OOD Detection Results Comparison")
    print(f"{'='*80}\n")
    
    for ood_dataset in ood_datasets:
        print(f"\nOOD Dataset: {ood_dataset}")
        print(f"{'-'*80}")
        print(f"{'Method':<20} {'AUROC':<12} {'FPR@95':<12} {'AUPR':<12}")
        print(f"{'-'*80}")
        
        for method in methods:
            m = results[ood_dataset][method]
            print(f"{method:<20} {m['auroc']:<12.4f} {m['fpr@95']:<12.4f} {m['aupr']:<12.4f}")
        
        print(f"{'-'*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OOD Detection Experiments')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/lab4/classifier_baseline/best.pt',
                       help='Classifier checkpoint path')
    parser.add_argument('--ood-dataset', type=str, default='cifar100_subset',
                       choices=['cifar100_subset', 'fakedata', 'svhn'],
                       help='OOD dataset')
    parser.add_argument('--score-method', type=str, default='max_softmax',
                       choices=['max_softmax', 'max_logit', 'energy'],
                       help='Scoring method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softmax')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all methods')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Create plots directory
    Path('./plots/lab4').mkdir(exist_ok=True)
    
    if args.compare:
        compare_ood_methods()
    else:
        metrics = run_ood_detection(
            classifier_checkpoint=args.checkpoint,
            ood_dataset=args.ood_dataset,
            score_method=args.score_method,
            temperature=args.temperature,
            plot=not args.no_plot
        )
        
        print(f"\nOOD Detection Results:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  FPR@95: {metrics['fpr@95']:.4f}")
        print(f"  AUPR: {metrics['aupr']:.4f}")
