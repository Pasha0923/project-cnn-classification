# project-cnn-classification

## 📌 Project Description
This project is a deep learning-based image classification system for the CIFAR-10 dataset.
It classifies input images into one of 10 categories using a fine-tuned ResNet18 convolutional neural network.

## Project includes:
- model training and evaluation
- performance visualization
- interactive Streamlit web application for inference
- Dockerized deployment for reproducibility

## 🎯 Project Goal
The main goal of this project is to build a high-accuracy image classifier using deep learning techniques, specifically:
- leverage a pretrained ResNet18 model
- apply transfer learning (fine-tuning)
- achieve strong generalization on CIFAR-10 dataset
- provide a user-friendly interface for predictions

## 🗂️ Project Structure
```bash
project-cnn-classification/
│
├── notebook/
│   └── demo.ipynb
│
├── configuration/
│   └── config.py
│
├── models/
│   ├── cnn_model.py
│   └── best_model_2.pth
│
├── training/
│   ├── cifar10_loader.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   └── history.json
├── .gitignore
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## **Dataset**
The project uses the CIFAR-10 dataset, which contains 60,000 32×32 RGB images across 10 classes.

The dataset is automatically downloaded using PyTorch:
```bash
datasets.CIFAR10(root="data/raw", download=True)
```
Dataset contains 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## **Model Architecture**
Base model: ResNet18 (pretrained on ImageNet)
Transfer Learning strategy:
replaced final fully connected layer
adapted output layer to 10 classes
fine-tuned all layers

## 📦 Libraries
Python 3.11

- torch
- torchvision
- numpy
- pillow
- matplotlib
- streamlit
- tqdm
- scikit-learn

## **Parametres model**
| Parameter | Value |
|--------|-------|
| BATCH_SIZE | 32 |
| LEARNING_RATE | 0.0001 |
| EPOCHS | 10 |
| early_stopping_patience | 5 |

## Notebook (demo.ipynb)
The demo.ipynb notebook contains:

- Model training
- Evaluation model
- Visualization: Loss / Accuracy curves
- Confusion Matrix
- Classification Report
- prediction examples
- ROC-AUC (multiclass OvR)
- ROC curves per class

## 📊 Final Evaluation Summary

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 94.98% |
| Test Accuracy | 94.68% |
| Macro F1-score | 0.95 |
| Macro ROC-AUC (OvR) | 0.9980 |

The model demonstrates strong generalization ability with high classification performance across all 10 CIFAR-10 classes.  
The near-perfect ROC-AUC indicates excellent separability between classes in a one-vs-rest setting.


## ⚡ Local Installation Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Pasha0923/project-cnn-classification.git
cd project-cnn-classification
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Install PyTorch CPU versions:**
```bash
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
```
4. **Run the Streamlit application:**
```bash
streamlit run app.py
```

## Run with Docker

1. **Clone the repository:**
```bash
git clone https://github.com/Pasha0923/project-cnn-classification.git
cd project-cnn-classification
```
2.  **Build and run container**
```bash
docker compose up --build
```
3. **Open in browser**
```bash
http://localhost:8501
```

🐳 Docker Notes
- Uses CPU-only PyTorch build
- No CUDA dependencies required
- Fully reproducible environment
- Streamlit runs inside container

## 📊 Streamlit Application
The app allows:

- image upload (JPG/PNG/JPEG)
- real-time classification
- prediction confidence display
- top-3 predictions
- probability bar chart visualization
- model information sidebar