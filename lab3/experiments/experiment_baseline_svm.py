from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC

from lab3.utils.data_utils import extract_features_for_baseline
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed

logger = setup_logger('baseline_svm', log_file='./logs/lab3/baseline_svm.log')

def train_baseline_svm(
    model_name: str = "distilbert/distilbert-base-uncased",
    C: float = 1.0,
    kernel: str = 'rbf',
    device: str = 'cuda',
    seed: int = 42
):
    """
    Train baseline SVM on extracted features.
    
    Args:
        model_name: Name of pretrained model for feature extraction
        C: SVM regularization parameter
        kernel: SVM kernel
        device: Device for feature extraction
        seed: Random seed
    """
    set_seed(seed)
    
    logger.info("=" * 80)
    logger.info("Exercise 1.3: Baseline with Feature Extraction + SVM")
    logger.info("=" * 80)
    
    # Extract features
    logger.info(f"\nExtracting features using {model_name}...")
    (train_features, train_labels,
     val_features, val_labels,
     test_features, test_labels) = extract_features_for_baseline(
        model_name=model_name,
        save_path='./features/distilbert_features.pt',
        device=device
    )
    
    # Convert to numpy
    X_train = train_features.numpy()
    y_train = train_labels.numpy()
    X_val = val_features.numpy()
    y_val = val_labels.numpy()
    X_test = test_features.numpy()
    y_test = test_labels.numpy()
    
    logger.info(f"\nFeatures shape:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    # Train SVM
    logger.info(f"\nTraining SVM (C={C}, kernel={kernel})...")
    svm = SVC(C=C, kernel=kernel, random_state=seed, verbose=True)
    svm.fit(X_train, y_train)
    
    # Evaluate on validation
    logger.info("\nValidation Results:")
    val_preds = svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='binary')
    
    logger.info(f"  Accuracy: {val_acc * 100:.2f}%")
    logger.info(f"  F1 Score: {val_f1:.4f}")
    
    # Evaluate on test
    logger.info("\nTest Results:")
    test_preds = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='binary')
    
    logger.info(f"  Accuracy: {test_acc * 100:.2f}%")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, test_preds, target_names=['Negative', 'Positive']))
    
    logger.info("=" * 80)
    
    return {
        'val_acc': val_acc,
        'val_f1': val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline SVM Experiment')
    parser.add_argument('--model-name', type=str, default='distilbert/distilbert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--C', type=float, default=1.0,
                       help='SVM regularization parameter')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly'],
                       help='SVM kernel')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for feature extraction')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    metrics = train_baseline_svm(
        model_name=args.model_name,
        C=args.C,
        kernel=args.kernel,
        device=args.device,
        seed=args.seed
    )
    
    print(f"\nFinal Results:")
    print(f"  Validation Accuracy: {metrics['val_acc'] * 100:.2f}%")
    print(f"  Test Accuracy: {metrics['test_acc'] * 100:.2f}%")
