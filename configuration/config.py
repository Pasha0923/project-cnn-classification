import torch
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_CLASSES = 10
MODEL_PATH = "models/best_model_2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"