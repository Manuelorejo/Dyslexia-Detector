import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import DyslexiaCNN
import os
import requests
# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dyslexia Detection", layout="centered")





CLASS_NAMES = ["corrected", "normal", "reversal"]
DEVICE = torch.device("cpu")

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    # Replace with the direct link to the .pth file in Releases > Assets
    URL = "https://github.com/Manuelorejo/Dyslexia-Detector/releases/download/v1.0/dyslexia_cnn.pth"

    # Download if not exists
    if not os.path.exists("dyslexia_cnn.pth"):
        print("Downloading model...")
        r = requests.get(URL, stream=True)
        with open("dyslexia_cnn.pth", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed!")

    # Initialize model and load weights
    model = DyslexiaCNN()
    model.load_state_dict(torch.load("dyslexia_cnn.pth", map_location="cpu"))
    model.eval()
    return model
model = load_model()

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------
# UI
# -----------------------
st.title("📝 Dyslexia Handwriting Detection")
st.write("Upload a handwriting image to classify dyslexic writing patterns.")

uploaded_file = st.file_uploader(
    "Upload handwriting image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    st.markdown("### 📊 Prediction Result")
    st.success(f"**Class:** {CLASS_NAMES[pred_class]}")
    st.write(f"Confidence: **{probs[0][pred_class]*100:.2f}%**")

    st.markdown("---")
    st.caption("""
    **Class meanings**
    - Normal: Typical handwriting
    - Corrected: Reversals corrected during writing
    - Reversal: Persistent letter/number reversals
    """)


