from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Resize CIFAR-10 images from 32x32 to 224x224 , ResNet18 expects larger input images
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

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_dataloaders(batch_size):
# dowmnload CIFAR-10 train dataset
    full_train_dataset = datasets.CIFAR10(
        root='data/raw',
        train=True,
        download=True,
        transform=train_transform
    )
# download CIFAR-10 test dataset 
    test_dataset = datasets.CIFAR10(
        root='data/raw',
        train=False,
        download=True,
        transform=val_test_transform
    )
# Split the original training dataset into new training and validation datasets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Validation should not use augmentation
    val_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader
