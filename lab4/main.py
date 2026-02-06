import argparse
from pathlib import Path

from experiments.experiment_adversarial import run_adversarial_attack, compare_attack_methods
from experiments.experiment_ood_detection import run_ood_detection, compare_ood_methods
from experiments.train_autoencoder import train_autoencoder
from experiments.train_classifier import train_classifier
from experiments.train_robust import train_robust_classifier


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Lab4 - OOD Detection & Adversarial Learning')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['classifier', 'ood', 'adversarial', 'robust', 'autoencoder', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer epochs for testing')
    
    # Classifier args
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (auto-selected if not specified)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    
    # OOD args
    parser.add_argument('--ood-dataset', type=str, default='cifar100_subset',
                       choices=['cifar100_subset', 'fakedata', 'svhn'],
                       help='OOD dataset')
    parser.add_argument('--score-method', type=str, default='energy',
                       choices=['max_softmax', 'max_logit', 'energy', 'mse'],
                       help='OOD scoring method')
    
    # Adversarial args
    parser.add_argument('--attack', type=str, default='pgd',
                       choices=['fgsm', 'pgd', 'bim'],
                       help='Adversarial attack method')
    parser.add_argument('--epsilon', type=float, default=0.031,
                       help='Attack epsilon (8/255 ≈ 0.031)')
    
    # Robust training args
    parser.add_argument('--adv-ratio', type=float, default=0.5,
                       help='Adversarial ratio for robust training')
    
    args = parser.parse_args()
    
    # Auto-select epochs based on mode
    if args.epochs is None:
        if args.quick:
            classifier_epochs = 20
            robust_epochs = 20
            autoencoder_epochs = 10
        else:
            classifier_epochs = 100
            robust_epochs = 100
            autoencoder_epochs = 50
    else:
        classifier_epochs = args.epochs
        robust_epochs = args.epochs
        autoencoder_epochs = args.epochs
    
    # Create necessary directories
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./logs').mkdir(exist_ok=True)
    Path('./plots').mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Lab4 - Out-of-Distribution Detection & Adversarial Learning")
    print("=" * 80 + "\n")
    
    if args.quick:
        print("⚡ QUICK MODE: Using fewer epochs for faster testing\n")
    
    if args.experiment == 'classifier':
        print("Training Baseline Classifier\n")
        metrics = train_classifier(
            dataset='cifar10',
            num_classes=10,
            num_epochs=classifier_epochs,
            batch_size=args.batch_size,
            lr=0.1,
            experiment_name='classifier_baseline'
        )
        print(f"\nClassifier trained!")
        print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
    
    elif args.experiment == 'ood':
        print("Exercise 1: OOD Detection\n")
        
        # Check if classifier exists
        classifier_path = './checkpoints/classifier_baseline/best.pt'
        if not Path(classifier_path).exists():
            print("⚠️  Baseline classifier not found!")
            print("Training classifier first...\n")
            train_classifier(num_epochs=classifier_epochs, batch_size=args.batch_size)
            print()
        
        print(f"Running OOD detection with {args.score_method} on {args.ood_dataset}\n")
        metrics = run_ood_detection(
            classifier_checkpoint=classifier_path,
            ood_dataset=args.ood_dataset,
            score_method=args.score_method,
            plot=True
        )
        
        print(f"\nOOD Detection Results:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  FPR@95: {metrics['fpr@95']:.4f}")
        print(f"  AUPR: {metrics['aupr']:.4f}")
    
    elif args.experiment == 'adversarial':
        print("Exercise 2: Adversarial Attacks\n")
        
        # Check if classifier exists
        classifier_path = './checkpoints/classifier_baseline/best.pt'
        if not Path(classifier_path).exists():
            print("⚠️  Baseline classifier not found!")
            print("Training classifier first...\n")
            train_classifier(num_epochs=classifier_epochs, batch_size=args.batch_size)
            print()
        
        print(f"Running {args.attack.upper()} attack with epsilon={args.epsilon}\n")
        metrics = run_adversarial_attack(
            classifier_checkpoint=classifier_path,
            attack_method=args.attack,
            epsilon=args.epsilon,
            num_samples=1000,
            visualize=True
        )
        
        print(f"\nAdversarial Attack Results:")
        print(f"  Clean Accuracy: {metrics['clean_acc']:.2f}%")
        print(f"  Adversarial Accuracy: {metrics['adv_acc']:.2f}%")
        print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    
    elif args.experiment == 'robust':
        print("Exercise 3: Robust Training\n")
        print(f"Training robust classifier with {args.adv_ratio*100:.0f}% adversarial examples\n")
        
        metrics = train_robust_classifier(
            num_epochs=robust_epochs,
            adv_ratio=args.adv_ratio,
            train_epsilon=args.epsilon,
            experiment_name='robust_classifier'
        )
        
        print(f"\nRobust Classifier trained!")
        print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
        
        # Test robustness
        print("\n" + "-" * 80)
        print("Testing robustness against PGD attack...\n")
        robust_metrics = run_adversarial_attack(
            classifier_checkpoint='./checkpoints/robust_classifier/best.pt',
            attack_method='pgd',
            epsilon=args.epsilon,
            num_samples=1000,
            visualize=False
        )
        
        print(f"\nRobust Model vs PGD:")
        print(f"  Clean Accuracy: {robust_metrics['clean_acc']:.2f}%")
        print(f"  Adversarial Accuracy: {robust_metrics['adv_acc']:.2f}%")
    
    elif args.experiment == 'autoencoder':
        print("Training Autoencoder for OOD Detection\n")
        
        # Check if classifier exists
        classifier_path = './checkpoints/classifier_baseline/best.pt'
        if not Path(classifier_path).exists():
            print("⚠️  Baseline classifier not found!")
            print("Training classifier first...\n")
            train_classifier(num_epochs=classifier_epochs, batch_size=args.batch_size)
            print()
        
        results = train_autoencoder(
            classifier_checkpoint=classifier_path,
            freeze_encoder=True,
            num_epochs=autoencoder_epochs,
            experiment_name='autoencoder'
        )
        
        print(f"\nAutoencoder trained!")
        print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    
    elif args.experiment == 'all':
        print("Running All Experiments\n")
        
        # 1. Train baseline classifier
        print("\n" + "=" * 80)
        print("Step 1/5: Training Baseline Classifier")
        print("=" * 80)
        train_classifier(
            num_epochs=classifier_epochs,
            batch_size=args.batch_size,
            experiment_name='classifier_baseline'
        )
        
        # 2. OOD Detection
        print("\n" + "=" * 80)
        print("Step 2/5: OOD Detection (comparing methods)")
        print("=" * 80)
        compare_ood_methods()
        
        # 3. Adversarial Attacks
        print("\n" + "=" * 80)
        print("Step 3/5: Adversarial Attacks (comparing methods)")
        print("=" * 80)
        compare_attack_methods()
        
        # 4. Robust Training
        print("\n" + "=" * 80)
        print("Step 4/5: Robust Training")
        print("=" * 80)
        train_robust_classifier(
            num_epochs=robust_epochs,
            adv_ratio=0.5,
            experiment_name='robust_classifier'
        )
        
        # Test robustness
        print("\nTesting robust classifier against PGD...")
        run_adversarial_attack(
            classifier_checkpoint='./checkpoints/robust_classifier/best.pt',
            attack_method='pgd',
            epsilon=0.031,
            num_samples=1000,
            visualize=False
        )
        
        # 5. Autoencoder
        print("\n" + "=" * 80)
        print("Step 5/5: Autoencoder for OOD Detection")
        print("=" * 80)
        train_autoencoder(
            classifier_checkpoint='./checkpoints/classifier_baseline/best.pt',
            num_epochs=autoencoder_epochs,
            experiment_name='autoencoder'
        )
        
        print("\n" + "=" * 80)
        print("🎉 All Lab4 experiments completed!")
        print("=" * 80)
        print("\nCheckpoints saved in: ./checkpoints/")
        print("Logs saved in: ./logs/")
        print("Plots saved in: ./plots/")
    
    print("\n✅ Lab4 completed!\n")


if __name__ == "__main__":
    main()
