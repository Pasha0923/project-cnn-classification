import streamlit as st
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.cnn_model import CIFAR10ResNet

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="centered"
)

classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load model

@st.cache_resource
def load_model():
    model = CIFAR10ResNet()
    model.load_state_dict(
        torch.load("models/best_model_2.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# prediction function

def predict_probs(image: Image.Image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    return probs

def get_top3(probs):
    idx = np.argsort(probs)[::-1][:3]
    return [(classes[i], probs[i]) for i in idx]

# UI

st.title("🧠 CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader(
    "Upload image (png, jpg, jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict 🚀"):

        with st.spinner("Model is thinking..."):

            probs = predict_probs(image)

            pred_class = classes[np.argmax(probs)]
            confidence = np.max(probs)

        # result 

        st.success(f"Prediction: **{pred_class}**")
        st.info(f"Confidence: **{confidence:.2%}**")

        # top3 

        st.subheader("🔝 Top-3 Predictions")

        top3 = get_top3(probs)

        for cls, p in top3:
            st.write(f"• {cls}: {p:.2%}")

        # bar-chart

        st.subheader("📊 Class Probabilities")

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(classes, probs)
        plt.xticks(rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Distribution")

        st.pyplot(fig)

# training curves

st.markdown("---")

show_curves = st.checkbox("Show training curves 📈")

if show_curves:

    st.subheader("📈 Training Curves")

    try:
        with open("outputs/history.json", "r") as f:
            history = json.load(f)

        # LOSS
        fig, ax = plt.subplots()
        ax.plot(history["train_losses"], label="Train Loss")
        ax.plot(history["val_losses"], label="Val Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        st.pyplot(fig)

        # ACCURACY
        fig, ax = plt.subplots()
        ax.plot(history["train_accuracies"], label="Train Accuracy")
        ax.plot(history["val_accuracies"], label="Val Accuracy")
        ax.set_title("Accuracy Curve")
        ax.legend()
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("history.json not found. Train model first.")

# sidebar info
st.sidebar.title("ℹ️ Model Info")

st.sidebar.write("ResNet18 (Transfer Learning)")
st.sidebar.write("Dataset: CIFAR-10")
st.sidebar.write("Classes: 10")

icons = {
    "airplane":"✈️",
    "automobile":"🚗",
    "bird":"🐦",
    "cat":"🐱",
    "deer":"🦌",
    "dog":"🐶",
    "frog":"🐸",
    "horse":"🐴",
    "ship":"🚢",
    "truck":"🚚"
}

st.sidebar.markdown("### Classes")

for c in classes:
    st.sidebar.write(f"{icons[c]} {c}")