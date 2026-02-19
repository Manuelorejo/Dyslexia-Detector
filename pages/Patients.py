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
# Config
# -----------------------
st.set_page_config(page_title="Dyslexia Prediction", layout="wide")
CLASS_NAMES = ["corrected", "normal", "reversal"]
DEVICE = torch.device("cpu")
UPLOAD_FOLDER = "uploads"
DB_FILE = "patients.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>
.stApp {
    background-color: #F3F4F6;
    font-family: 'Segoe UI', sans-serif;
}

/* Section headings */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
    color: #0E6BA8;
}

/* Uploaded image card */
.upload-card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 20px;
}

/* Prediction charts */
.plotly-graph-div {
    background-color: white !important;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

/* Dataframe styling */
.stDataFrame div[data-testid="stDataFrameContainer"] {
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    overflow: hidden;
}

/* Sidebar */
.stSidebar {
    background-color: #0E6BA8;
    color: white;
    padding-top: 20px;
}

.stSidebar h2, .stSidebar h3 {
    color: white;
}

.stSidebar selectbox, .stSidebar input {
    background-color: white;
    color: #0E6BA8;
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

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
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------
# DB Utilities
# -----------------------
def get_patients():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT patient_id, name FROM Patients")
    patients = c.fetchall()
    conn.close()
    return patients

def add_patient(name, age=None, gender=None, notes=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO Patients (name, age, gender, notes, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, age, gender, notes, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def add_prediction(patient_id, filename, prediction_class, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO Predictions (patient_id, filename, prediction_class, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (patient_id, filename, prediction_class, confidence, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT filename, prediction_class, confidence, timestamp FROM Predictions WHERE patient_id=? ORDER BY timestamp DESC",
        (patient_id,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------
# Session State
# -----------------------
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None
    st.session_state.selected_patient_name = None

# -----------------------
# Sidebar: Patients
# -----------------------
st.sidebar.title("👥 Patients")
patients = get_patients()
default_idx = 0
if st.session_state.selected_patient_id:
    for i, (pid, pname) in enumerate(patients):
        if pid == st.session_state.selected_patient_id:
            default_idx = i
            break

selected_patient = st.sidebar.selectbox(
    "Select Patient",
    options=[pname for pid, pname in patients],
    index=default_idx
)

for pid, pname in patients:
    if pname == selected_patient:
        st.session_state.selected_patient_id = pid
        st.session_state.selected_patient_name = pname
        break

# -----------------------
# Main App
# -----------------------
st.title("📝 Handwriting Upload & Prediction")

if st.session_state.selected_patient_id is None:
    st.info("Please select a patient from the sidebar.")
else:
    st.subheader(f"Patient: {st.session_state.selected_patient_name}")
    uploaded_file = st.file_uploader(
        f"Upload handwriting image for {st.session_state.selected_patient_name}",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=False, width=400)
        st.markdown('</div>', unsafe_allow_html=True)

        # Predict
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = probs[0][pred_class_idx].item()

        st.markdown(f"### Prediction: **{pred_class.upper()}**")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Prediction Bar Chart
        fig_ind = go.Figure(go.Bar(
            x=CLASS_NAMES,
            y=probs[0].numpy(),
            marker_color=["#14B8A6", "#0E6BA8", "#EF4444"]
        ))
        fig_ind.update_layout(
            title="Prediction Confidence",
            yaxis=dict(title="Probability", range=[0,1]),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9'
        )
        st.plotly_chart(fig_ind, use_container_width=True)

        # Save file & store prediction
        patient_folder = os.path.join(UPLOAD_FOLDER, str(st.session_state.selected_patient_id))
        os.makedirs(patient_folder, exist_ok=True)
        file_path = os.path.join(patient_folder, uploaded_file.name)
        image.save(file_path)
        add_prediction(st.session_state.selected_patient_id, uploaded_file.name, pred_class, confidence)
        st.success("✅ Prediction saved to patient history!")

    # -----------------------
    # Patient History
    # -----------------------
    st.subheader("👤 Patient Prediction History")
    history = get_patient_history(st.session_state.selected_patient_id)
    if history:
        df = pd.DataFrame(history, columns=["Filename", "Prediction", "Confidence", "Timestamp"])
        st.dataframe(df, height=300)

        counts = {cls:0 for cls in CLASS_NAMES}
        total_conf = {cls:0.0 for cls in CLASS_NAMES}
        for row in history:
            counts[row[1]] += 1
            total_conf[row[1]] += row[2]

        # Pie chart
        counts_list = [counts[cls] for cls in CLASS_NAMES]
        fig_pie = px.pie(
            names=CLASS_NAMES,
            values=counts_list,
            color=CLASS_NAMES,
            color_discrete_map={"corrected":"#14B8A6", "normal":"#0E6BA8", "reversal":"#EF4444"},
            title="Distribution of Predictions"
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart: avg confidence
        avg_conf = {cls: total_conf[cls]/counts[cls] if counts[cls]>0 else 0 for cls in CLASS_NAMES}
        avg_conf_values = [avg_conf[cls] for cls in CLASS_NAMES]
        fig_bar = go.Figure(data=[go.Bar(
            x=CLASS_NAMES,
            y=avg_conf_values,
            marker_color=["#14B8A6","#0E6BA8","#EF4444"],
            text=[f"{v*100:.1f}%" for v in avg_conf_values],
            textposition='auto'
        )])
        fig_bar.update_layout(
            title="Average Confidence per Class",
            yaxis=dict(range=[0,1], title="Confidence"),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Overall verdict
        overall_class = max(total_conf, key=total_conf.get)
        st.markdown(f"### 🏁 Overall Verdict: **{overall_class.upper()}**")
    else:
        st.info("No predictions yet for this patient.")
