import streamlit as st

st.set_page_config(page_title="Dyslexia Detection", layout="wide")

# -----------------------
# Redirect if not logged in
# -----------------------
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first to access this page.")
    st.stop()

# -----------------------
# Page Styles (colorful feature cards)
# -----------------------
st.markdown("""
<style>
    /* Page background */
    .main {
        background: linear-gradient(135deg, #F0F8FF 0%, #E0F2FF 100%);
        padding: 2rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #0E6BA8;  /* primary blue */
        font-weight: 600;
    }

    /* Subtitles / text */
    .stMarkdown p, .stText {
        color: #475569;
        font-size: 16px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0E6BA8, #14B8A6);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        opacity: 0.9;
    }

    /* Columns for features */
    .feature-col {
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        color: white;
        font-weight: 500;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(14,107,168,0.1);
    }

    .feature1 {
        background: linear-gradient(135deg, #0E6BA8, #14B8A6);
    }
    .feature2 {
        background: linear-gradient(135deg, #14B8A6, #0E6BA8);
    }
    .feature3 {
        background: linear-gradient(135deg, #0E6BA8, #0EB8A6);
    }

    .feature-col h3 {
        margin-bottom: 10px;
    }

    .feature-col p {
        font-size: 15px;
        color: #f0f8ff;
    }
    
       
        /* Sidebar background */
    .css-1d391kg {  /* main sidebar container */
        background: linear-gradient(135deg, #0E6BA8, #14B8A6) !important;
        color: white !important;
    }
    
    /* Sidebar title / headers */
    .css-1d391kg h2, 
    .css-1d391kg h3, 
    .css-1d391kg .css-1v0mbdj { 
        color: white !important;
    }
    
    /* Sidebar labels for inputs */
    .css-1d391kg label {
        color: white !important;
    }
    
    /* Sidebar input text color */
    .css-1d391kg input, .css-1d391kg select, .css-1d391kg textarea {
        color: #0E6BA8 !important;
    }
    
    /* Sidebar buttons */
    .css-1d391kg div.stButton > button {
        background: white !important;
        color: #0E6BA8 !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg div.stButton > button:hover {
        background: #e0f2ff !important;
        color: #0E6BA8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Hero Section
# -----------------------
st.markdown("<div style='text-align:center; padding: 20px;'>", unsafe_allow_html=True)
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

for i, (col, (title, desc)) in enumerate(zip(cols, features), start=1):
    col.markdown(f"<div class='feature-col feature{i}'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Call to Action Section
# -----------------------
st.subheader("🚀 Get Started")
col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Dashboard"):
        st.experimental_set_query_params(page="dashboard")
        st.switch_page("pages/Dashboard.py")

with col2:
    if st.button("Go to Prediction System"):
        st.experimental_set_query_params(page="prediction")
        st.switch_page("pages/Patients.py")

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
    