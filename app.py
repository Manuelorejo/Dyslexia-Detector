import streamlit as st

st.set_page_config(page_title="Dire Medical AI", layout="wide")

# -------------------------
# CUSTOM CSS WITH GRADIENT
# -------------------------
st.markdown("""
<style>
/* FULL SCREEN BACKGROUND GRADIENT */
.stApp {
    background: linear-gradient(135deg, #0E6BA8, #14B8A6);
    color: white;
    height: 100vh;
}

/* FLEX CONTAINER ON TOP */
.overlay-container {
    display: flex;
    height: 100vh;
    align-items: center;
    justify-content: center;
    gap: 50px;
    padding: 0 50px;
}

/* LEFT INFO SECTION */
.left-section {
    flex: 1;
}

.left-section h1 {
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 20px;
}

.left-section p {
    font-size: 18px;
    line-height: 1.6;
}

/* LOGIN CARD */
.login-card {
    flex: 1;
    background: white;
    padding: 50px;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
    color: #0E6BA8;
}

.login-card .title {
    font-size: 32px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #0E6BA8;
}

.login-card .subtitle {
    color: #64748B;
    margin-bottom: 30px;
}

/* BUTTONS */
div.stButton > button {
    background-color: #0E6BA8;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #094d78;
}

/* TAB STYLING */
.css-1d391kg {  /* sidebar container if needed later */
    background: linear-gradient(135deg, #0E6BA8, #14B8A6);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE INIT
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------------
# OVERLAY CONTAINER
# -------------------------
st.markdown("""
<div class="overlay-container">
    <div class="left-section">
        <h1>AI-Powered Medical Predictions</h1>
        <p>
            Helping doctors make smarter clinical decisions
            through predictive analytics and intelligent dashboards.
        </p>
    </div>
    <div class="login-card">
        <div class="title">Welcome Back</div>
        <div class="subtitle">Login to access the medical dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN / SIGN UP WIDGETS
# -------------------------
tab1, tab2 = st.tabs(["Login", "Sign Up"])

with tab1:
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if email and password:
            st.session_state.authenticated = True
            st.session_state.user = email
            st.success("Login Successful!")
            st.switch_page("pages/Dashboard.py")
        else:
            st.error("Please enter valid credentials.")

with tab2:
    new_email = st.text_input("Email", key="signup_email")
    new_password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        elif new_email and new_password:
            st.success("Account Created! Please Login.")
        else:
            st.error("Please fill all fields.")
