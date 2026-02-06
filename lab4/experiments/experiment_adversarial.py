from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lab4.configs.config import AdversarialConfig
from lab4.models.ood_models import SimpleCNN
from lab4.utils.adversarial_utils import (
    fgsm_attack,
    pgd_attack,
    bim_attack,
    evaluate_attack
)
from lab4.utils.data_utils import create_id_dataloaders, get_inverse_transform
from utils_for_all.checkpoint import load_checkpoint
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device


def visualize_adversarial_examples(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    clean_labels: torch.Tensor,
    adv_labels: torch.Tensor,
    class_names: list,
    num_samples: int = 5,
    save_path: str = None
):
    """
    Visualize clean vs adversarial images.
    
    Args:
        clean_images: Clean images
        adv_images: Adversarial images
        clean_labels: Clean predictions
        adv_labels: Adversarial predictions
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save plot
    """
    inv_transform = get_inverse_transform('cifar10')
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    
    for i in range(num_samples):
        # Clean image
        clean_img = inv_transform(clean_images[i]).permute(1, 2, 0).cpu().numpy()
        clean_img = np.clip(clean_img, 0, 1)
        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f'Clean: {class_names[clean_labels[i]]}')
        axes[i, 0].axis('off')
        
        # Adversarial image
        adv_img = inv_transform(adv_images[i]).permute(1, 2, 0).cpu().numpy()
        adv_img = np.clip(adv_img, 0, 1)
        axes[i, 1].imshow(adv_img)
        axes[i, 1].set_title(f'Adversarial: {class_names[adv_labels[i]]}')
        axes[i, 1].axis('off')
        
        # Difference (amplified for visibility)
        diff = (adv_images[i] - clean_images[i]).abs()
        diff_img = inv_transform(diff * 10).permute(1, 2, 0).cpu().numpy()
        diff_img = np.clip(diff_img, 0, 1)
        axes[i, 2].imshow(diff_img)
        axes[i, 2].set_title('Difference (10x)')
        axes[i, 2].axis('off')
        
        # Perturbation histogram
        perturbation = (adv_images[i] - clean_images[i]).flatten().cpu().numpy()
        axes[i, 3].hist(perturbation, bins=50, alpha=0.7)
        axes[i, 3].set_title('Perturbation Dist')
        axes[i, 3].set_xlabel('Value')
        axes[i, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def run_adversarial_attack(
    classifier_checkpoint: str = './checkpoints/lab4/classifier_baseline/best.pt',
    attack_method: str = 'fgsm',
    epsilon: float = 0.031,
    targeted: bool = False,
    num_samples: int = 1000,
    visualize: bool = True
):
    """
    Run adversarial attack experiment.
    
    Args:
        classifier_checkpoint: Path to classifier checkpoint
        attack_method: Attack method (fgsm, pgd, bim)
        epsilon: Attack epsilon
        targeted: Whether to use targeted attack
        num_samples: Number of samples to attack
        visualize: Whether to visualize results
    """
    # Create configuration
    config = AdversarialConfig(
        classifier_checkpoint=classifier_checkpoint,
        attack_method=attack_method,
        epsilon=epsilon,
        targeted=targeted,
        num_test_samples=num_samples,
        experiment_name=f'adversarial_{attack_method}',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('adversarial', log_file=f'./logs/lab4/adversarial_{attack_method}.log')
    
    logger.info("=" * 80)
    logger.info(f"Adversarial Attack Experiment")
    logger.info(f"Attack Method: {attack_method.upper()}")
    logger.info(f"Epsilon: {epsilon}")
    logger.info(f"Targeted: {targeted}")
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
    
    # Create dataloader
    _, _, test_loader = create_id_dataloaders(
        dataset='cifar10',
        batch_size=32,
        num_workers=4,
        augment=False,
        root='./data'
    )
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Collect samples
    all_clean_images = []
    all_adv_images = []
    all_labels = []
    all_clean_preds = []
    all_adv_preds = []
    
    total_samples = 0
    
    for images, labels in test_loader:
        if total_samples >= num_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Generate adversarial examples
        if attack_method == 'fgsm':
            adv_images = fgsm_attack(
                model, images, labels,
                epsilon=epsilon,
                targeted=targeted,
                target_labels=None  # Random targets for targeted attack
            )
        elif attack_method == 'pgd':
            adv_images = pgd_attack(
                model, images, labels,
                epsilon=epsilon,
                alpha=config.pgd_alpha,
                num_steps=config.pgd_steps,
                random_start=config.pgd_random_start,
                targeted=targeted,
                target_labels=None
            )
        elif attack_method == 'bim':
            adv_images = bim_attack(
                model, images, labels,
                epsilon=epsilon,
                alpha=config.pgd_alpha,
                num_steps=10,
                targeted=targeted,
                target_labels=None
            )
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Get predictions
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            adv_preds = model(adv_images).argmax(dim=1)
        
        all_clean_images.append(images.cpu())
        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())
        all_clean_preds.append(clean_preds.cpu())
        all_adv_preds.append(adv_preds.cpu())
        
        total_samples += images.size(0)
    
    # Concatenate results
    clean_images = torch.cat(all_clean_images)[:num_samples]
    adv_images = torch.cat(all_adv_images)[:num_samples]
    labels = torch.cat(all_labels)[:num_samples]
    clean_preds = torch.cat(all_clean_preds)[:num_samples]
    adv_preds = torch.cat(all_adv_preds)[:num_samples]
    
    # Evaluate attack
    metrics = evaluate_attack(
        model, clean_images, adv_images, labels, device
    )
    
    logger.info(f"\nAttack Results:")
    logger.info(f"  Clean Accuracy: {metrics['clean_acc']:.2f}%")
    logger.info(f"  Adversarial Accuracy: {metrics['adv_acc']:.2f}%")
    logger.info(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    logger.info(f"  Average Perturbation: {metrics['avg_perturbation']:.6f}")
    
    # Visualize
    if visualize:
        visualize_adversarial_examples(
            clean_images[:5],
            adv_images[:5],
            clean_preds[:5],
            adv_preds[:5],
            class_names,
            num_samples=5,
            save_path=f'./plots/adversarial_{attack_method}_eps{epsilon:.3f}.png'
        )
    
    logger.info("=" * 80)
    
    return metrics


def compare_attack_methods():
    """
    Compare different adversarial attack methods.
    """
    methods = ['fgsm', 'bim', 'pgd']
    epsilons = [0.007, 0.015, 0.031]  # ~2/255, ~4/255, ~8/255
    
    results = {}
    
    for method in methods:
        results[method] = {}
        
        for eps in epsilons:
            print(f"\nTesting {method.upper()} with epsilon={eps:.3f}...")
            metrics = run_adversarial_attack(
                attack_method=method,
                epsilon=eps,
                num_samples=1000,
                visualize=False
            )
            results[method][eps] = metrics
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("Adversarial Attack Results Comparison")
    print(f"{'='*80}\n")
    
    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print(f"{'-'*80}")
        print(f"{'Epsilon':<15} {'Clean Acc':<15} {'Adv Acc':<15} {'Attack Success':<15}")
        print(f"{'-'*80}")
        
        for eps in epsilons:
            m = results[method][eps]
            print(f"{eps:<15.3f} {m['clean_acc']:<15.2f} {m['adv_acc']:<15.2f} {m['attack_success_rate']:<15.2f}")
        
        print(f"{'-'*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Adversarial Attack Experiments')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/lab4/classifier_baseline/best.pt',
                       help='Classifier checkpoint path')
    parser.add_argument('--attack', type=str, default='fgsm',
                       choices=['fgsm', 'pgd', 'bim'],
                       help='Attack method')
    parser.add_argument('--epsilon', type=float, default=0.031,
                       help='Attack epsilon (8/255 ≈ 0.031)')
    parser.add_argument('--targeted', action='store_true',
                       help='Use targeted attack')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to attack')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all attack methods')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create plots directory
    Path('./plots').mkdir(exist_ok=True)
    
    if args.compare:
        compare_attack_methods()
    else:
        metrics = run_adversarial_attack(
            classifier_checkpoint=args.checkpoint,
            attack_method=args.attack,
            epsilon=args.epsilon,
            targeted=args.targeted,
            num_samples=args.num_samples,
            visualize=not args.no_visualize
        )
        
        print(f"\nAdversarial Attack Results:")
        print(f"  Clean Accuracy: {metrics['clean_acc']:.2f}%")
        print(f"  Adversarial Accuracy: {metrics['adv_acc']:.2f}%")
        print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
