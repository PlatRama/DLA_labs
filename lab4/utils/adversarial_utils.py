import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: Target model
        images: Clean images [batch, C, H, W]
        labels: True labels
        epsilon: Perturbation magnitude
        targeted: Whether to perform targeted attack
        target_labels: Target labels for targeted attack
    
    Returns:
        Adversarial images
    """
    images = images.clone().detach()
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Compute loss
    if targeted:
        # Targeted: minimize loss for target class
        assert target_labels is not None
        loss = F.cross_entropy(outputs, target_labels)
        loss = -loss  # Gradient descent on -loss = gradient ascent on loss
    else:
        # Untargeted: maximize loss for true class
        loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples
    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign
    
    # Clamp to valid range
    adv_images = torch.clamp(adv_images, images.min(), images.max())
    
    return adv_images.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    alpha: float = 0.007,
    num_steps: int = 40,
    random_start: bool = True,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: Target model
        images: Clean images
        labels: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_steps: Number of iterations
        random_start: Whether to start from random point
        targeted: Whether to perform targeted attack
        target_labels: Target labels for targeted attack
    
    Returns:
        Adversarial images
    """
    images = images.clone().detach()
    
    # Random initialization
    if random_start:
        adv_images = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, images.min(), images.max())
    else:
        adv_images = images.clone()
    
    # PGD iterations
    for step in range(num_steps):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Compute loss
        if targeted:
            assert target_labels is not None
            loss = F.cross_entropy(outputs, target_labels)
            loss = -loss
        else:
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial images
        grad_sign = adv_images.grad.data.sign()
        adv_images = adv_images.detach() + alpha * grad_sign
        
        # Project back to epsilon ball
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = images + perturbation
        
        # Clamp to valid range
        adv_images = torch.clamp(adv_images, images.min(), images.max())
    
    return adv_images.detach()


def bim_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    alpha: float = 0.007,
    num_steps: int = 10,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Basic Iterative Method (BIM) attack.
    Same as PGD but without random start.
    
    Args:
        model: Target model
        images: Clean images
        labels: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_steps: Number of iterations
        targeted: Whether to perform targeted attack
        target_labels: Target labels for targeted attack
    
    Returns:
        Adversarial images
    """
    return pgd_attack(
        model, images, labels,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        random_start=False,
        targeted=targeted,
        target_labels=target_labels
    )


def evaluate_attack(
    model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate attack success.
    
    Args:
        model: Target model
        clean_images: Clean images
        adv_images: Adversarial images
        labels: True labels
        device: Device
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Clean accuracy
        clean_outputs = model(clean_images.to(device))
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = (clean_preds == labels.to(device)).float().mean().item()
        
        # Adversarial accuracy
        adv_outputs = model(adv_images.to(device))
        adv_preds = adv_outputs.argmax(dim=1)
        adv_acc = (adv_preds == labels.to(device)).float().mean().item()
        
        # Attack success rate
        attack_success = (clean_preds != adv_preds).float().mean().item()
        
        # Average perturbation
        perturbation = (adv_images - clean_images).abs().mean().item()
    
    return {
        'clean_acc': clean_acc * 100,
        'adv_acc': adv_acc * 100,
        'attack_success_rate': attack_success * 100,
        'avg_perturbation': perturbation
    }


def adversarial_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    alpha: float = 0.007,
    num_steps: int = 7,
    criterion: nn.Module = None
) -> torch.Tensor:
    """
    Compute loss for adversarial training.
    
    Args:
        model: Model being trained
        images: Clean images
        labels: Labels
        epsilon: Attack epsilon
        alpha: Attack step size
        num_steps: Attack steps
        criterion: Loss criterion
    
    Returns:
        Loss tensor
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Generate adversarial examples
    model.eval()  # Set to eval for attack generation
    adv_images = pgd_attack(
        model, images, labels,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        random_start=True
    )
    model.train()  # Back to train mode
    
    # Compute loss on adversarial examples
    outputs = model(adv_images)
    loss = criterion(outputs, labels)
    
    return loss
