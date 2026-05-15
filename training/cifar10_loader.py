from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224, padding=4),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# DATA LOADER
# =========================================================

def get_dataloaders(batch_size):

    # -----------------------------------------------------
    # Load base dataset ONLY ONCE
    # -----------------------------------------------------
    base_dataset = datasets.CIFAR10(
        root='data/raw',
        train=True,
        download=True
    )

    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    # -----------------------------------------------------
    # CLEAN TRAIN/VAL SPLIT (reproducible)
    # -----------------------------------------------------
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=targets
    )

    # -----------------------------------------------------
    # TRAIN DATASET (with augmentation)
    # -----------------------------------------------------
    train_dataset = Subset(
        datasets.CIFAR10(
            root='data/raw',
            train=True,
            download=False,
            transform=train_transform
        ),
        train_idx
    )

    # -----------------------------------------------------
    # VAL DATASET (NO augmentation)
    # -----------------------------------------------------
    val_dataset = Subset(
        datasets.CIFAR10(
            root='data/raw',
            train=True,
            download=False,
            transform=val_transform
        ),
        val_idx
    )

    # -----------------------------------------------------
    # TEST DATASET
    # -----------------------------------------------------
    test_dataset = datasets.CIFAR10(
        root='data/raw',
        train=False,
        download=True,
        transform=val_transform
    )

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader