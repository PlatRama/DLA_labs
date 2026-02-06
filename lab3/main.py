import argparse

from experiments.experiment_baseline import compare_frozen_vs_trainable
from experiments.experiment_baseline_svm import train_baseline_svm
from experiments.experiment_finetuning import (
    run_full_finetuning,
    run_lora_finetuning,
    compare_finetuning_methods
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Lab3 - Transformers Experiments')
    parser.add_argument('--experiment', type=str, default='lora',
                       choices=['baseline_svm', 'baseline', 'finetuning', 'lora', 'compare_all', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--model-name', type=str,
                       default='distilbert/distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (auto-selected if not specified)')
    
    # LoRA specific
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Lab3 - Transformers: HuggingFace Ecosystem")
    print("=" * 80 + "\n")
    
    if args.experiment == 'baseline_svm' or args.experiment == 'all':
        print("Running Exercise 1.3: Baseline with SVM\n")
        metrics = train_baseline_svm(
            model_name=args.model_name,
            C=1.0,
            kernel='rbf',
            device='cuda',
            seed=42
        )
        print(f"\nFinal Results:")
        print(f"  Validation Accuracy: {metrics['val_acc'] * 100:.2f}%")
        print(f"  Test Accuracy: {metrics['test_acc'] * 100:.2f}%")
    
    elif args.experiment == 'baseline' or args.experiment == 'all':
        print("Running Exercise 2: Baseline with PyTorch Trainer\n")
        print("Comparing frozen vs trainable backbone...\n")
        compare_frozen_vs_trainable()
    
    elif args.experiment == 'finetuning' or args.experiment == 'all':
        print("Running Exercise 2.3: Full Fine-tuning\n")
        lr = args.lr if args.lr is not None else 2e-5
        results = run_full_finetuning(
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr
        )
        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {results['eval_accuracy'] * 100:.2f}%")
        print(f"  Test F1: {results['eval_f1']:.4f}")
    
    elif args.experiment == 'lora' or args.experiment == 'all':
        print("Running Exercise 3.1: LoRA Fine-tuning\n")
        lr = args.lr if args.lr is not None else 1e-4
        results = run_lora_finetuning(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr
        )
        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {results['eval_accuracy'] * 100:.2f}%")
        print(f"  Test F1: {results['eval_f1']:.4f}")
    
    elif args.experiment == 'compare_all' or args.experiment == 'all':
        print("Comparing All Fine-tuning Methods\n")
        compare_finetuning_methods()
    
    """elif args.experiment == 'all':
        print("Running All Experiments\n")
        
        # 1. Baseline SVM
        print("\n" + "=" * 80)
        print("1. Exercise 1.3: Baseline SVM")
        print("=" * 80)
        train_baseline_svm()
        
        # 2. Baseline PyTorch
        print("\n" + "=" * 80)
        print("2. Exercise 2: Baseline PyTorch")
        print("=" * 80)
        compare_frozen_vs_trainable()
        
        # 3. LoRA (best method)
        print("\n" + "=" * 80)
        print("3. Exercise 3.1: LoRA Fine-tuning")
        print("=" * 80)
        run_lora_finetuning(num_epochs=10)
        
        print("\n" + "=" * 80)
        print("All experiments completed!")
        print("=" * 80)"""
    
    print("\n✅ Lab3 completed!\n")


if __name__ == "__main__":
    main()
