import streamlit as st

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Dyslexia Detection", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f5f5f5; padding: 2rem;}
    h1, h2, h3 {color: #1f4e79;}
    .stButton>button {background-color: #1f4e79; color: white; border-radius: 10px; padding: 0.5rem 1rem;}
    .stButton>button:hover {background-color: #145374; color: white;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Hero Section
# -----------------------
st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
st.title("📝 Dyslexia Handwriting Detection")
st.subheader("Empowering doctors to detect dyslexia through handwriting analysis")
st.markdown("""
Detect, analyze, and track handwriting patterns to provide insights into dyslexic tendencies.
Our AI-powered system leverages deep learning to assist with patient assessment and management.
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Features Section
# -----------------------
st.subheader("🌟 Key Features")
cols = st.columns(3)
features = [
    ("Patient Management", "Add and manage patient profiles with detailed info and history."),
    ("Handwriting Analysis", "Upload handwriting images and get accurate predictions with visualizations."),
    ("Analytics Dashboard", "Track overall and individual predictions with interactive charts.")
]

for col, (title, desc) in zip(cols, features):
    col.markdown(f"### {title}")
    col.write(desc)

st.markdown("---")

# -----------------------
# Call to Action Section
# -----------------------
st.subheader("🚀 Get Started")
col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Dashboard"):
        st.switch_page("pages/dashboard.py")
    

with col2:
    if st.button("Go to Prediction System"):
        st.switch_page("pages/patients.py")

st.markdown("---")

# -----------------------
# About Section
# -----------------------
st.subheader("📚 About This Project")
st.markdown("""
This application was developed to support medical professionals in detecting dyslexia through handwriting.  
It uses a **Convolutional Neural Network (CNN)** trained on handwriting datasets to classify patterns into:

- **Normal:** Typical handwriting  
- **Corrected:** Letters/numbers initially reversed but corrected  
- **Reversal:** Persistent reversals during writing  

Every prediction is stored in the patient’s history, contributing to an **overall verdict**.  
Interactive visualizations allow tracking trends across patients and over time.
""")
