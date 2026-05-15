from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# =========================================================
# TRAIN TRANSFORMS (with augmentation)
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

# =========================================================
# VALIDATION / TEST TRANSFORMS (no augmentation)
# =========================================================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(batch_size):

    # =====================================================
    # 1. SINGLE SOURCE DATASET (NO DUPLICATION)
    # =====================================================
    base_dataset = datasets.CIFAR10(
        root='data/raw',
        train=True,
        download=True
    )

    # =====================================================
    # 2. SPLIT INDICES
    # =====================================================
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    train_indices, val_indices = random_split(
        range(len(base_dataset)),
        [train_size, val_size]
    )

    # =====================================================
    # 3. SUBSETS WITH SEPARATE TRANSFORMS
    # =====================================================

    train_dataset = Subset(
        datasets.CIFAR10(
            root='data/raw',
            train=True,
            download=False,
            transform=train_transform
        ),
        train_indices.indices
    )

    val_dataset = Subset(
        datasets.CIFAR10(
            root='data/raw',
            train=True,
            download=False,
            transform=val_transform
        ),
        val_indices.indices
    )

    # =====================================================
    # 4. TEST DATASET
    # =====================================================
    test_dataset = datasets.CIFAR10(
        root='data/raw',
        train=False,
        download=True,
        transform=val_transform
    )

    # =====================================================
    # 5. DATALOADERS
    # =====================================================
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