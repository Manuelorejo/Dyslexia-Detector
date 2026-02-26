import streamlit as st
from database import init_db, create_user, authenticate_user

st.set_page_config(page_title="Dire Medical AI", layout="wide")

# -------------------------
# INITIALIZE DATABASE
# -------------------------
init_db()

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

/* FLEX CONTAINER */
.overlay-container {
    display: flex;
    height: 100vh;
    align-items: center;
    justify-content: center;
    gap: 60px;
    padding: 0 80px;
}

/* LEFT INFO SECTION */
.left-section {
    flex: 1;
}

.left-section h1 {
    font-size: 52px;
    font-weight: bold;
    margin-bottom: 20px;
}

.left-section p {
    font-size: 18px;
    line-height: 1.6;
    opacity: 0.95;
}

/* LOGIN CARD */
.login-wrapper {
    flex: 1;
    display: flex;
    justify-content: center;
}

.login-card {
    background: white;
    padding: 50px;
    border-radius: 20px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.2);
    color: #0E6BA8;
    width: 100%;
    max-width: 450px;
}

.login-card .title {
    font-size: 30px;
    font-weight: 600;
    margin-bottom: 10px;
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
    width: 100%;
}

div.stButton > button:hover {
    background-color: #094d78;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE INIT
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# -------------------------
# LAYOUT
# -------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <div class="left-section">
        <h1>AI-Powered Medical Predictions</h1>
        <p>
            Helping doctors make smarter clinical decisions
            through predictive analytics and intelligent dashboards.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Welcome Back</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Login to access your dashboard</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # ---------------- LOGIN ----------------
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            user_id = authenticate_user(email, password)

            if user_id:
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.success("Login Successful!")
                st.switch_page("pages/Landing.py")
            else:
                st.error("Invalid email or password.")

    # ---------------- SIGN UP ----------------
    with tab2:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_email or not new_password:
                st.error("Please fill all fields.")
            else:
                success = create_user(new_email, new_password)

                if success:
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Email already exists.")


    st.markdown('</div>', unsafe_allow_html=True)
