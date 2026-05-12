import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from models.cnn_model import CIFAR10ResNet
from training.cifar10_loader import get_dataloaders
from configuration.config import *
from tqdm import tqdm


train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

# model
model = CIFAR10ResNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor=0.5
)

# metrics
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_accuracy = 0
patience_counter = 0
early_stopping_patience = 5

os.makedirs("outputs", exist_ok=True)

# training
for epoch in range(EPOCHS):

   
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct / total

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # validation
    model.eval()

    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = 100 * val_correct / val_total

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    scheduler.step(epoch_val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"[TRAIN] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% || "
        f"[VAL] Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%"
    )

    #  save best model
    if epoch_val_acc > best_accuracy:
        best_accuracy = epoch_val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
        print("Best model saved")
    else:
        patience_counter += 1

    #  early stopping
    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

# save training history
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

with open("outputs/history.json", "w") as f:
    json.dump(history, f)

print(f"Training completed. Best Val Acc: {best_accuracy:.2f}%")