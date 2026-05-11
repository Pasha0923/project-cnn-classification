import torch

from models.cnn_model import CIFAR10ResNet

from training.cifar10_loader import get_dataloaders

from configuration.config import *


classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# Load test loader
_, _, test_loader = get_dataloaders(
    BATCH_SIZE
)

# Load model
model = CIFAR10ResNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(
    f"Test Accuracy: {accuracy:.2f}%"
)