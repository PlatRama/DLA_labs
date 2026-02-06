import logging
from typing import Tuple, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

logger = logging.getLogger(__name__)


def load_rotten_tomatoes_dataset():
    """
    Load the Rotten Tomatoes dataset from HuggingFace.
    
    Returns:
        Dataset splits (train, validation, test)
    """
    dataset_name = "cornell-movie-review-data/rotten_tomatoes"
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    logger.info(f"Dataset loaded:")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Validation: {len(dataset['validation'])} samples") 
    logger.info(f"  Test: {len(dataset['test'])} samples")
    
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    text_column: str = 'text',
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True
) -> Dataset:
    """
    Tokenize a HuggingFace dataset.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use
        text_column: Name of text column
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
    
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
    
    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized


def create_sentiment_dataloaders(
    model_name: str,
    batch_size: int = 64,
    max_length: int = 512,
    num_workers: int = 4,
    use_hf_trainer: bool = False
) -> Union[Tuple[DataLoader, DataLoader, DataLoader], 
           Tuple[Dataset, Dataset, Dataset]]:
    """
    Create dataloaders for sentiment analysis on Rotten Tomatoes.
    
    Args:
        model_name: Name of the model (for tokenizer)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        use_hf_trainer: If True, return HF Datasets; if False, return DataLoaders
    
    Returns:
        Train, validation, test dataloaders or datasets
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_rotten_tomatoes_dataset()
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length' if not use_hf_trainer else False,
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize all splits
    logger.info("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )
    
    train_dataset = tokenized_dataset['train']
    val_dataset = tokenized_dataset['validation']
    test_dataset = tokenized_dataset['test']
    
    # Return HF datasets for Trainer
    if use_hf_trainer:
        return train_dataset, val_dataset, test_dataset
    
    # Create PyTorch dataloaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    logger.info("Dataloaders created successfully")
    
    return train_loader, val_loader, test_loader


def extract_features_for_baseline(
    model_name: str,
    save_path: Optional[str] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features from a pretrained model for baseline experiments.
    Uses the [CLS] token from the last layer.
    
    Args:
        model_name: Name of the pretrained model
        save_path: Path to save extracted features (optional)
        device: Device to use
    
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels, 
                  test_features, test_labels)
    """
    from transformers import AutoModel
    from tqdm import tqdm
    
    logger.info(f"Extracting features using {model_name}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_rotten_tomatoes_dataset()
    
    def extract_split_features(split_name):
        """Extract features for a dataset split."""
        features = []
        labels = []
        
        split = dataset[split_name]
        
        with torch.no_grad():
            for i in tqdm(range(len(split)), desc=f'Extracting {split_name}'):
                text = split[i]['text']
                label = split[i]['label']
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get features from [CLS] token
                outputs = model(**inputs)
                cls_features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                
                features.append(cls_features.cpu())
                labels.append(label)
        
        features = torch.cat(features, dim=0)
        labels = torch.tensor(labels)
        
        return features, labels
    
    # Extract for all splits
    train_features, train_labels = extract_split_features('train')
    val_features, val_labels = extract_split_features('validation')
    test_features, test_labels = extract_split_features('test')
    
    logger.info(f"Features extracted:")
    logger.info(f"  Train: {train_features.shape}")
    logger.info(f"  Val: {val_features.shape}")
    logger.info(f"  Test: {test_features.shape}")
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'train_features': train_features,
            'train_labels': train_labels,
            'val_features': val_features,
            'val_labels': val_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'model_name': model_name
        }, save_path)
        
        logger.info(f"Features saved to {save_path}")
    
    return (train_features, train_labels, 
            val_features, val_labels,
            test_features, test_labels)
