import torch
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 12
NUM_CLASSES = 10
MODEL_PATH = "models/best_model_2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"