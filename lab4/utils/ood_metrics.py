import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def max_softmax_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Maximum softmax probability as OOD score.
    Higher score = more likely ID.
    
    Args:
        logits: Model logits [batch, num_classes]
        temperature: Temperature for softmax
    
    Returns:
        Scores [batch]
    """
    probs = F.softmax(logits / temperature, dim=1)
    scores = probs.max(dim=1)[0]
    return scores


def max_logit_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum logit as OOD score.
    Higher score = more likely ID.
    
    Args:
        logits: Model logits [batch, num_classes]
    
    Returns:
        Scores [batch]
    """
    scores = logits.max(dim=1)[0]
    return scores


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy score for OOD detection.
    Higher score = more likely ID.
    
    Args:
        logits: Model logits [batch, num_classes]
        temperature: Temperature parameter
    
    Returns:
        Scores [batch]
    """
    scores = temperature * torch.logsumexp(logits / temperature, dim=1)
    return scores


def reconstruction_score(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Reconstruction error as OOD score.
    Lower score = more likely ID (better reconstruction).
    
    Args:
        images: Original images [batch, C, H, W]
        reconstructions: Reconstructed images [batch, C, H, W]
        reduction: How to reduce over spatial dimensions
    
    Returns:
        Scores [batch]
    """
    mse = F.mse_loss(reconstructions, images, reduction='none')
    
    if reduction == 'mean':
        # Mean over C, H, W
        scores = mse.mean(dim=[1, 2, 3])
    elif reduction == 'sum':
        scores = mse.sum(dim=[1, 2, 3])
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return scores


@torch.no_grad()
def compute_ood_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    score_method: str = 'max_softmax',
    temperature: float = 1.0,
    device: str = 'cuda',
    autoencoder: nn.Module = None
) -> torch.Tensor:
    """
    Compute OOD scores for a dataset.
    
    Args:
        model: Classifier model
        dataloader: Data loader
        score_method: Scoring method
        temperature: Temperature for softmax/energy
        device: Device
        autoencoder: Autoencoder model (for reconstruction scores)
    
    Returns:
        Scores tensor
    """
    model.eval()
    all_scores = []
    
    for images, _ in dataloader:
        images = images.to(device)
        
        if score_method == 'mse' or score_method == 'reconstruction':
            # Use autoencoder reconstruction error
            assert autoencoder is not None
            autoencoder.eval()
            reconstructions = autoencoder(images)
            scores = reconstruction_score(images, reconstructions)
            # Negate so higher = more ID (for consistency)
            scores = -scores
        else:
            # Use classifier-based scores
            logits = model(images)
            
            if score_method == 'max_softmax':
                scores = max_softmax_score(logits, temperature)
            elif score_method == 'max_logit':
                scores = max_logit_score(logits)
            elif score_method == 'energy':
                scores = energy_score(logits, temperature)
            else:
                raise ValueError(f"Unknown score method: {score_method}")
        
        all_scores.append(scores.cpu())
    
    return torch.cat(all_scores)


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    fpr_threshold: float = 0.95
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Assumes higher score = more likely ID.
    
    Args:
        id_scores: Scores for ID data
        ood_scores: Scores for OOD data
        fpr_threshold: FPR threshold for FPR@X metric
    
    Returns:
        Dictionary of metrics
    """
    # Combine scores and create labels
    # ID = 1 (positive), OOD = 0 (negative)
    scores = torch.cat([id_scores, ood_scores]).numpy()
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # FPR@95 (false positive rate at 95% true positive rate)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find FPR at specified TPR threshold
    idx = np.argmax(tpr >= fpr_threshold)
    fpr_at_threshold = fpr[idx]
    
    # AUPR (Area Under Precision-Recall curve)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    
    # Detection error
    # Minimum of (FPR + FNR) / 2
    fnr = 1 - tpr
    detection_error = np.minimum(fpr + fnr, 1.0).min() / 2
    
    metrics = {
        'auroc': auroc,
        f'fpr@{int(fpr_threshold*100)}': fpr_at_threshold,
        'aupr': aupr,
        'detection_error': detection_error
    }
    
    logger.info(f"OOD Detection Metrics:")
    logger.info(f"  AUROC: {auroc:.4f}")
    logger.info(f"  FPR@{int(fpr_threshold*100)}: {fpr_at_threshold:.4f}")
    logger.info(f"  AUPR: {aupr:.4f}")
    logger.info(f"  Detection Error: {detection_error:.4f}")
    
    return metrics


def find_optimal_threshold(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        id_scores: ID scores
        ood_scores: OOD scores
    
    Returns:
        Tuple of (threshold, accuracy)
    """
    scores = torch.cat([id_scores, ood_scores]).numpy()
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    
    # Try different thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    accuracies = []
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = (predictions == labels).mean()
        accuracies.append(accuracy)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return best_threshold, best_accuracy
