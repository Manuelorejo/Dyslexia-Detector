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

# Load the model
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
# Database Utilities
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
    created_at = datetime.datetime.now().isoformat()
    c.execute(
        "INSERT INTO Patients (name, age, gender, notes, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, age, gender, notes, created_at)
    )
    conn.commit()
    conn.close()

def add_prediction(patient_id, filename, prediction_class, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute(
        "INSERT INTO Predictions (patient_id, filename, prediction_class, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (patient_id, filename, prediction_class, confidence, timestamp)
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
# Initialize session state
# -----------------------
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None
    st.session_state.selected_patient_name = None

# -----------------------
# Sidebar: Patient Management
# -----------------------
st.sidebar.title("Patients")
patients = get_patients()  # list of tuples [(id, name), ...]

# Determine default selected patient
default_idx = 0
if st.session_state.selected_patient_id:
    for i, (pid, pname) in enumerate(patients):
        if pid == st.session_state.selected_patient_id:
            default_idx = i
            break

# Use selectbox to select patient
selected_patient = st.sidebar.selectbox(
    "Select Patient",
    options=[pname for pid, pname in patients],
    index=default_idx
)
# Update session_state when selectbox changes
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
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=False, width=400)

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = probs[0][pred_class_idx].item()

        st.markdown(f"### Prediction: **{pred_class.upper()}**")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # -----------------------
        # Current Prediction Visualization
        # -----------------------
        fig_ind = go.Figure(go.Bar(
            x=CLASS_NAMES,
            y=probs[0].numpy(),
            marker_color=["green","blue","red"],
        ))
        fig_ind.update_layout(
            title="Prediction Confidence",
            yaxis=dict(title="Probability", range=[0,1]),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#f9f9f9'
        )
        st.plotly_chart(fig_ind, use_container_width=True)

        # -----------------------
        # Save file & store prediction
        # -----------------------
        patient_folder = os.path.join(UPLOAD_FOLDER, str(st.session_state.selected_patient_id))
        os.makedirs(patient_folder, exist_ok=True)
        file_path = os.path.join(patient_folder, uploaded_file.name)
        image.save(file_path)
        add_prediction(st.session_state.selected_patient_id, uploaded_file.name, pred_class, confidence)
        st.success("✅ Prediction saved to patient history!")

    # -----------------------
    # Patient History & Visualizations
    # -----------------------
    st.subheader("👤 Patient Prediction History")
    history = get_patient_history(st.session_state.selected_patient_id)
    if history:
        df = pd.DataFrame(history, columns=["Filename", "Prediction", "Confidence", "Timestamp"])
        st.dataframe(df, height=300)

        # Pie chart: distribution of predictions
        counts = {cls:0 for cls in CLASS_NAMES}
        total_conf = {cls:0.0 for cls in CLASS_NAMES}
        for row in history:
            counts[row[1]] += 1
            total_conf[row[1]] += row[2]

        counts_list = [counts[cls] for cls in CLASS_NAMES]
        fig_pie = px.pie(
            names=CLASS_NAMES,
            values=counts_list,
            color=CLASS_NAMES,
            color_discrete_map={"corrected":"green", "normal":"blue", "reversal":"red"},
            title="Distribution of Predictions"
        )
        fig_pie.update_traces(textinfo='none', hovertemplate='%{label}: %{percent}')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart: average confidence per class
        avg_conf = {cls: total_conf[cls]/counts[cls] if counts[cls]>0 else 0 for cls in CLASS_NAMES}
        avg_conf_values = [avg_conf[cls] for cls in CLASS_NAMES]
        fig_bar = go.Figure(data=[
            go.Bar(
                x=CLASS_NAMES,
                y=avg_conf_values,
                marker_color=["green","blue","red"],
                text=[f"{v*100:.1f}%" for v in avg_conf_values],
                textposition='auto'
            )
        ])
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
