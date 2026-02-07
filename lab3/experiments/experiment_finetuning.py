import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from lab3.utils.data_utils import create_sentiment_dataloaders
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed


def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='binary'),
        "precision": precision_score(labels, predictions, average='binary'),
        "recall": recall_score(labels, predictions, average='binary'),
    }


def run_full_finetuning(
    model_name: str = "distilbert/distilbert-base-uncased",
    experiment_name: str = "finetuning_full",
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-5,
    seed: int = 42
):
    """
    Full fine-tuning with HuggingFace Trainer.
    
    Args:
        model_name: Pretrained model name
        experiment_name: Experiment name
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
    """
    set_seed(seed)
    logger = setup_logger('finetuning', log_file=f'./logs/lab3/{experiment_name}.log')
    
    logger.info("=" * 80)
    logger.info(f"Full Fine-tuning: {experiment_name}")
    logger.info("=" * 80)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = create_sentiment_dataloaders(
        model_name=model_name,
        batch_size=batch_size,
        use_hf_trainer=True  # Return HF datasets
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load model config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        classifier_dropout=0.3
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\nModel: {model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/lab3/{experiment_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"./logs/lab3/{experiment_name}",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        seed=seed,
        dataloader_pin_memory=True,
        report_to="tensorboard"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    logger.info("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info("\nTest Results:")
    logger.info(f"  Accuracy: {test_results['eval_accuracy'] * 100:.2f}%")
    logger.info(f"  F1 Score: {test_results['eval_f1']:.4f}")
    logger.info(f"  Precision: {test_results['eval_precision']:.4f}")
    logger.info(f"  Recall: {test_results['eval_recall']:.4f}")
    logger.info("=" * 80)
    
    return test_results


def run_lora_finetuning(
    model_name: str = "distilbert/distilbert-base-uncased",
    experiment_name: str = "finetuning_lora",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list = None,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42
):
    """
    LoRA fine-tuning (parameter-efficient).
    
    Args:
        model_name: Pretrained model name
        experiment_name: Experiment name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling)
        lora_dropout: LoRA dropout
        target_modules: Modules to apply LoRA to
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
    """
    if target_modules is None:
        # Default for DistilBERT
        target_modules = ["q_lin", "k_lin", "v_lin"]
    
    set_seed(seed)
    logger = setup_logger('lora', log_file=f'./logs/lab3/{experiment_name}.log')
    
    logger.info("=" * 80)
    logger.info(f"LoRA Fine-tuning: {experiment_name}")
    logger.info(f"LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    logger.info("=" * 80)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = create_sentiment_dataloaders(
        model_name=model_name,
        batch_size=batch_size,
        use_hf_trainer=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load model config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        classifier_dropout=0.3
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    logger.info("\nModel with LoRA:")
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/lab3/{experiment_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"./logs/lab3/{experiment_name}",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        seed=seed,
        dataloader_pin_memory=True,
        report_to="tensorboard"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    logger.info("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    logger.info("\nTest Results:")
    logger.info(f"  Accuracy: {test_results['eval_accuracy'] * 100:.2f}%")
    logger.info(f"  F1 Score: {test_results['eval_f1']:.4f}")
    logger.info(f"  Precision: {test_results['eval_precision']:.4f}")
    logger.info(f"  Recall: {test_results['eval_recall']:.4f}")
    logger.info("=" * 80)
    
    return test_results


def compare_finetuning_methods():
    """
    Compare full fine-tuning vs LoRA.
    """
    print("\n" + "=" * 80)
    print("Comparing Fine-tuning Methods")
    print("=" * 80 + "\n")
    
    # Full fine-tuning
    print("1. Full Fine-tuning...")
    full_results = run_full_finetuning(
        experiment_name='finetuning_full',
        num_epochs=5,  # Fewer epochs for comparison
        lr=2e-5
    )
    
    # LoRA fine-tuning
    print("\n2. LoRA Fine-tuning...")
    lora_results = run_lora_finetuning(
        experiment_name='finetuning_lora',
        lora_r=16,
        lora_alpha=32,
        num_epochs=10,  # More epochs since it's faster
        lr=1e-4
    )
    
    # Compare
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("-" * 80)
    print(f"{'Method':<30} {'Accuracy':<15} {'F1 Score':<15}")
    print("-" * 80)
    print(f"{'Full Fine-tuning':<30} {full_results['eval_accuracy']*100:<15.2f} {full_results['eval_f1']:<15.4f}")
    print(f"{'LoRA Fine-tuning':<30} {lora_results['eval_accuracy']*100:<15.2f} {lora_results['eval_f1']:<15.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tuning Experiments')
    parser.add_argument('--method', type=str, default='lora',
                       choices=['full', 'lora', 'compare'],
                       help='Fine-tuning method')
    parser.add_argument('--model-name', type=str,
                       default='distilbert/distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        compare_finetuning_methods()
    elif args.method == 'full':
        lr = args.lr if args.lr is not None else 2e-5
        results = run_full_finetuning(
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr
        )
        print(f"\nTest Accuracy: {results['eval_accuracy']*100:.2f}%")
    elif args.method == 'lora':
        lr = args.lr if args.lr is not None else 1e-4
        results = run_lora_finetuning(
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
        print(f"\nTest Accuracy: {results['eval_accuracy']*100:.2f}%")
