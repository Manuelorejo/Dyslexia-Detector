import os
import sqlite3
import datetime
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import DyslexiaCNN
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Dyslexia Prediction", layout="wide")

# -----------------------
# 🔐 LOGIN PROTECTION
# -----------------------
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login to access the prediction system.")
    st.stop()

user_id = st.session_state.user_id

# -----------------------
# COLOR CONSTANTS
# -----------------------
PRIMARY = "#0E6BA8"
TEAL = "#14B8A6"
RED = "#EF4444"

CLASS_NAMES = ["corrected", "normal", "reversal"]
DEVICE = torch.device("cpu")
UPLOAD_FOLDER = "uploads"
DB_FILE = "patients.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# 🎨 MODERN THEME CSS (UNCHANGED)
# -----------------------
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg, #E0F2FF 0%, #F0F9FF 100%);
    font-family: 'Segoe UI', sans-serif;
}}

h1, h2, h3 {{
    color: {PRIMARY};
    font-weight: 700;
}}

.card {{
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 12px 30px rgba(14,107,168,0.08);
    margin-bottom: 30px;
}}

.upload-card {{
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(14,107,168,0.08);
    text-align: center;
}}

div.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, {TEAL});
    color: white;
    border-radius: 12px;
    height: 45px;
    font-weight: 600;
    border: none;
}}

div.stButton > button:hover {{
    opacity: 0.9;
}}

.stDataFrame {{
    border-radius: 15px;
    overflow: hidden;
}}

.plotly-graph-div {{
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(14,107,168,0.08);
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PRIMARY}, {TEAL});
    color: white;
}}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {{
    color: white;
}}

section[data-testid="stSidebar"] div.stButton > button {{
    background: white;
    color: {PRIMARY};
}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    URL = "https://github.com/Manuelorejo/Dyslexia-Detector/releases/download/v1.0/dyslexia_cnn.pth"

    if not os.path.exists("dyslexia_cnn.pth"):
        r = requests.get(URL, stream=True)
        with open("dyslexia_cnn.pth", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    model = DyslexiaCNN()
    model.load_state_dict(torch.load("dyslexia_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------
# DATABASE UTILITIES (MULTI-USER SAFE)
# -----------------------
def get_patients(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT patient_id, name FROM Patients WHERE user_id = ?",
        (user_id,)
    )
    patients = c.fetchall()
    conn.close()
    return patients


def verify_patient_ownership(patient_id, user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT patient_id FROM Patients WHERE patient_id = ? AND user_id = ?",
        (patient_id, user_id)
    )
    result = c.fetchone()
    conn.close()
    return result is not None


def add_prediction(patient_id, filename, prediction_class, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO Predictions 
        (patient_id, filename, prediction_class, confidence, timestamp) 
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            filename,
            prediction_class,
            confidence,
            datetime.datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()


def get_patient_history(patient_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        SELECT filename, prediction_class, confidence, timestamp 
        FROM Predictions 
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        """,
        (patient_id,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------
# SIDEBAR (FILTERED + SECURE)
# -----------------------
st.sidebar.title("👥 Patients")

patients = get_patients(user_id)

if not patients:
    st.sidebar.warning("No patients available.")
    st.stop()

patient_names = [pname for pid, pname in patients]
patient_dict = {pname: pid for pid, pname in patients}

# If coming from dashboard
if "selected_patient_id" in st.session_state:
    selected_patient_id = st.session_state.selected_patient_id
    selected_patient = next(
        pname for pname, pid in patient_dict.items()
        if pid == selected_patient_id
    )
else:
    selected_patient = st.sidebar.selectbox(
        "Select Patient",
        options=patient_names
    )
    selected_patient_id = patient_dict[selected_patient]

# SECURITY CHECK
if not verify_patient_ownership(selected_patient_id, user_id):
    st.error("Unauthorized patient access.")
    st.stop()

# -----------------------
# MAIN CONTENT
# -----------------------
st.title("📝 Handwriting Upload & Prediction")

uploaded_file = st.file_uploader(
    f"Upload handwriting image for {selected_patient}",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=400)
    st.markdown('</div>', unsafe_allow_html=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = probs[0][pred_class_idx].item()

    st.markdown(f"## Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: {confidence*100:.2f}%")

    fig_ind = go.Figure(go.Bar(
        x=CLASS_NAMES,
        y=probs[0].numpy(),
        marker_color=[TEAL, PRIMARY, RED]
    ))
    fig_ind.update_layout(
        yaxis=dict(range=[0,1]),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    # Save image
    patient_folder = os.path.join(UPLOAD_FOLDER, str(selected_patient_id))
    os.makedirs(patient_folder, exist_ok=True)
    image.save(os.path.join(patient_folder, uploaded_file.name))

    add_prediction(selected_patient_id, uploaded_file.name, pred_class, confidence)

    st.success("✅ Prediction saved to patient history!")

# -----------------------
# HISTORY
# -----------------------
st.subheader("👤 Patient Prediction History")

history = get_patient_history(selected_patient_id)

if history:
    df = pd.DataFrame(history, columns=["Filename", "Prediction", "Confidence", "Timestamp"])
    st.dataframe(df, height=300)

    counts = df["Prediction"].value_counts()

    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map={
            "corrected": TEAL,
            "normal": PRIMARY,
            "reversal": RED
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("No predictions yet for this patient.")