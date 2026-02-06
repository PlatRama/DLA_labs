import logging
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

def get_transforms(dataset: str, augment: bool = True):
    """
    Get data transforms for the dataset.
    
    Args:
        dataset: Dataset name ('mnist', 'cifar10', 'cifar100')
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset == 'mnist':
        # MNIST normalization
        mean = (0.1307,)
        std = (0.3081,)
        
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif dataset in ['cifar10', 'cifar100']:
        # CIFAR normalization
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return train_transform, test_transform


def get_dataset(
    dataset_name: str,
    root: str = './data',
    train: bool = True,
    augment: bool = True
):
    """
    Get dataset.
    
    Args:
        dataset_name: Name of dataset
        root: Root directory for data
        train: Whether to load training set
        augment: Whether to apply data augmentation
    
    Returns:
        Dataset object
    """
    train_transform, test_transform = get_transforms(dataset_name, augment)
    transform = train_transform if train else test_transform
    
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    augment: bool = True,
    root: str = './data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        augment: Whether to apply data augmentation to training set
        root: Root directory for data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_dataset = get_dataset(dataset_name, root=root, train=True, augment=augment)
    test_dataset = get_dataset(dataset_name, root=root, train=False, augment=False)
    
    # Split train into train and validation
    num_train = len(train_dataset)
    num_val = int(num_train * val_split)
    
    indices = np.random.permutation(num_train)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Train samples: {len(train_subset)}")
    logger.info(f"Val samples: {len(val_subset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


class DistillationDataset(Dataset):
    """
    Dataset wrapper for knowledge distillation.
    Includes soft labels from teacher model.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        teacher_outputs: torch.Tensor
    ):
        """
        Args:
            base_dataset: Original dataset
            teacher_outputs: Pre-computed teacher outputs (logits or probabilities)
        """
        self.base_dataset = base_dataset
        self.teacher_outputs = teacher_outputs
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        teacher_output = self.teacher_outputs[idx]
        return image, label, teacher_output


def create_distillation_dataloaders(
    dataset_name: str,
    teacher_outputs_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    root: str = './data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for knowledge distillation.
    
    Args:
        dataset_name: Name of dataset
        teacher_outputs_path: Path to saved teacher outputs
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split fraction
        root: Root directory for data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load teacher outputs
    teacher_data = torch.load(teacher_outputs_path)
    train_teacher_outputs = teacher_data['train_outputs']
    val_teacher_outputs = teacher_data['val_outputs']
    
    # Load base datasets
    train_dataset = get_dataset(dataset_name, root=root, train=True, augment=True)
    test_dataset = get_dataset(dataset_name, root=root, train=False, augment=False)
    
    # Split indices
    num_train = len(train_dataset)
    num_val = int(num_train * val_split)
    
    indices = np.random.permutation(num_train)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    # Create distillation datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_distill = DistillationDataset(train_subset, train_teacher_outputs[train_indices])
    val_distill = DistillationDataset(val_subset, val_teacher_outputs[val_indices])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_distill,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_distill,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
