import torch.nn as nn
from torchvision.models import resnet18

class CIFAR10ResNet(nn.Module):
    def __init__(self):
        super(CIFAR10ResNet, self).__init__()

        self.model = resnet18(weights=None)

        # Fine-Tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # Replace classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, 10)
        )

    def forward(self, x):
        return self.model(x)