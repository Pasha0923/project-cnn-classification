import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.cnn_model import CIFAR10ResNet
from training.cifar10_loader import get_dataloaders
from configuration.config import *
from tqdm import tqdm
# import sys
# import os

# sys.path.append(
#     os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "..")
#     )
# )
# Load datasets
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

# Load model
model = CIFAR10ResNet().to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor=0.5
)

# Metrics storage
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_accuracy = 0


# ================= TRAINING LOOP =================

for epoch in range(EPOCHS):

    # ================= TRAIN =================

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    # for images, labels in train_loader:
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_train_loss = (
        running_loss / len(train_loader)
    )

    epoch_train_acc = (
        100 * correct / total
    )

    train_losses.append(epoch_train_loss)

    train_accuracies.append(epoch_train_acc)

    # ================= VALIDATION =================

    model.eval()

    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        # for images, labels in val_loader:
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(
                outputs.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    epoch_val_loss = (
        val_running_loss / len(val_loader)
    )

    epoch_val_acc = (
        100 * val_correct / val_total
    )

    val_losses.append(epoch_val_loss)

    val_accuracies.append(epoch_val_acc)

    scheduler.step(epoch_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Train Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Acc: {epoch_val_acc:.2f}%"
    )

    # Save best model
    if epoch_val_acc > best_accuracy:

        best_accuracy = epoch_val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved")


print(
    f"Training completed. "
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

# ================= PLOTS =================

# LOSS PLOT
plt.figure(figsize=(10, 5))
plt.plot(train_losses,label='Train Loss')
plt.plot(val_losses,label='Validation Loss')
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/loss_plot.png")

# ACCURACY PLOT
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies,label='Train Accuracy')
plt.plot(val_accuracies,label='Validation Accuracy')
plt.title("Train vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/accuracy_plot.png")