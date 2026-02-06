import logging
from typing import Tuple, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FakeData

logger = logging.getLogger(__name__)

def get_transforms(dataset: str = 'cifar10', augment: bool = True):
    """
    Get data transforms.
    
    Args:
        dataset: Dataset name
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset in ['cifar10', 'cifar100']:
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
    
    elif dataset == 'svhn':
        # SVHN normalization (similar to CIFAR)
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = train_transform
    
    else:
        # Default transforms
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = train_transform
    
    return train_transform, test_transform


def create_id_dataloaders(
    dataset: str = 'cifar10',
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.0,
    augment: bool = True,
    root: str = './data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create in-distribution (ID) dataloaders.
    
    Args:
        dataset: Dataset name
        batch_size: Batch size
        num_workers: Number of workers
        val_split: Validation split (if 0, use test as val)
        augment: Whether to apply augmentation
        root: Data root directory
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = get_transforms(dataset, augment)
    
    # Load dataset
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'svhn':
        train_dataset = SVHN(root=root, split='train', download=True, transform=train_transform)
        test_dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create validation split if needed
    if val_split > 0:
        num_train = len(train_dataset)
        num_val = int(num_train * val_split)
        indices = np.random.permutation(num_train)
        
        train_subset = Subset(train_dataset, indices[num_val:])
        val_subset = Subset(train_dataset, indices[:num_val])
        
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
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            test_dataset,
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
    
    logger.info(f"ID Dataset: {dataset}")
    logger.info(f"  Train: {len(train_loader.dataset)}")
    logger.info(f"  Val: {len(val_loader.dataset)}")
    logger.info(f"  Test: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def create_ood_dataloader(
    ood_type: str = 'cifar100_subset',
    id_dataset: str = 'cifar10',
    batch_size: int = 128,
    num_workers: int = 4,
    num_samples: int = 10000,
    root: str = './data'
) -> DataLoader:
    """
    Create out-of-distribution (OOD) dataloader.
    
    Args:
        ood_type: Type of OOD dataset
        id_dataset: ID dataset (for getting transforms)
        batch_size: Batch size
        num_workers: Number of workers
        num_samples: Number of samples (for FakeData)
        root: Data root directory
    
    Returns:
        OOD dataloader
    """
    _, test_transform = get_transforms(id_dataset, augment=False)
    
    if ood_type == 'cifar100_subset':
        # Use subset of CIFAR-100 classes not in CIFAR-10
        dataset = CIFAR100(root=root, train=False, download=True, transform=test_transform)
        
        # Select specific classes (e.g., people-related classes)
        # CIFAR-100 has 100 classes, we exclude vehicle and animal classes
        people_classes = ['baby', 'boy', 'girl', 'man', 'woman']
        all_classes = dataset.classes
        
        people_indices = [all_classes.index(c) for c in people_classes if c in all_classes]
        
        # Filter dataset
        indices = [i for i, label in enumerate(dataset.targets) if label in people_indices]
        dataset = Subset(dataset, indices)
        
        logger.info(f"OOD Dataset: CIFAR-100 subset (people classes)")
        logger.info(f"  Classes: {people_classes}")
        logger.info(f"  Samples: {len(dataset)}")
    
    elif ood_type == 'svhn':
        # SVHN as OOD for CIFAR-10
        dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
        logger.info(f"OOD Dataset: SVHN")
        logger.info(f"  Samples: {len(dataset)}")
    
    elif ood_type == 'fakedata':
        # Random noise images
        dataset = FakeData(
            size=num_samples,
            image_size=(3, 32, 32),
            transform=test_transform
        )
        logger.info(f"OOD Dataset: FakeData (random noise)")
        logger.info(f"  Samples: {len(dataset)}")
    
    elif ood_type == 'cifar10':
        # CIFAR-10 as OOD for CIFAR-100 experiments
        dataset = CIFAR10(root=root, train=False, download=True, transform=test_transform)
        logger.info(f"OOD Dataset: CIFAR-10")
        logger.info(f"  Samples: {len(dataset)}")
    
    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")
    
    ood_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return ood_loader


class NormalizeInverse(transforms.Normalize):
    """
    Inverse normalization transform for visualization.
    """
    
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)
    
    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def get_inverse_transform(dataset: str = 'cifar10'):
    """
    Get inverse normalization transform for visualization.
    
    Args:
        dataset: Dataset name
    
    Returns:
        Inverse transform
    """
    if dataset in ['cifar10', 'cifar100']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    return NormalizeInverse(mean, std)
