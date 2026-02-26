import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_FILE = "patients.db"
st.set_page_config(page_title="Dyslexia Dashboard", layout="wide")

# -----------------------
# 🔐 LOGIN PROTECTION
# -----------------------
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please login to access the dashboard.")
    st.stop()

# Get logged-in user
user_id = st.session_state.user_id

# -----------------------
# CUSTOM CSS (UNCHANGED)
# -----------------------
st.markdown("""
<style>
.stApp { background-color: #F3F4F6; font-family: 'Segoe UI', sans-serif; }

.patient-card {
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
    background: linear-gradient(135deg, #0E6BA8, #14B8A6);
    color: white;
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: center;
    cursor: pointer;
}
.patient-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 25px rgba(0,0,0,0.2);
}
.patient-card small { color: rgba(255,255,255,0.8); }

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
    color: #0E6BA8;
}

.plotly-graph-div {
    background-color: white !important;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

.stTabs [role="tablist"] button {
    font-weight: 600;
    color: #0E6BA8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# DATABASE CONNECTION
# -----------------------
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# -----------------------
# 👥 PATIENT CARDS (FILTERED BY USER)
# -----------------------
c.execute("""
    SELECT patient_id, name, age, gender 
    FROM Patients 
    WHERE user_id = ?
""", (user_id,))

all_patients = c.fetchall()

st.subheader("👥 Patients Overview", anchor="patients-overview")

if not all_patients:
    st.markdown("""
    <div style="
        background: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    ">
        <h3 style="color:#0E6BA8; margin-bottom:10px;">
            No Patients Yet
        </h3>
        <p style="color:#64748B; font-size:16px;">
            You haven’t added any patients yet.
            <br><br>
            Once you add a patient, they will appear here.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    cols = st.columns(3)

    for i, (pid, name, age, gender) in enumerate(all_patients):
        with cols[i % 3]:

            c.execute("""
                SELECT COUNT(*), MAX(timestamp) 
                FROM Predictions 
                WHERE patient_id = ?
            """, (pid,))
            total_preds, last_pred = c.fetchone()

            last_pred_str = last_pred if last_pred else "No predictions"

            st.markdown(f"""
            <div class="patient-card">
                <h3>{name}</h3>
                <small>Age: {age}, {gender}</small><br>
                <small>Predictions: {total_preds}</small><br>
                <small>Last Prediction: {last_pred_str}</small>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Select {pid}", key=f"btn_{pid}"):
                st.session_state.selected_patient_id = pid
                st.session_state.selected_patient_name = name
                st.switch_page("pages/Patients.py")

for i, (pid, name, age, gender) in enumerate(all_patients):
    with cols[i % 3]:

        # Only count predictions for this doctor's patient
        c.execute("""
            SELECT COUNT(*), MAX(timestamp) 
            FROM Predictions 
            WHERE patient_id = ?
        """, (pid,))
        total_preds, last_pred = c.fetchone()

        last_pred_str = last_pred if last_pred else "No predictions"

        st.markdown(f"""
        <div class="patient-card">
            <h3>{name}</h3>
            <small>Age: {age}, {gender}</small><br>
            <small>Predictions: {total_preds}</small><br>
            <small>Last Prediction: {last_pred_str}</small>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Select {pid}", key=f"btn_{pid}"):
            st.session_state.selected_patient_id = pid
            st.session_state.selected_patient_name = name
            st.switch_page("pages/Predictions.py")

# -----------------------
# 📊 OVERALL PREDICTIONS PIE (USER ONLY)
# -----------------------
c.execute("""
    SELECT prediction_class, COUNT(*) 
    FROM Predictions 
    WHERE patient_id IN (
        SELECT patient_id FROM Patients WHERE user_id = ?
    )
    GROUP BY prediction_class
""", (user_id,))

overall_counts = c.fetchall()

if overall_counts:
    classes = [row[0] for row in overall_counts]
    counts = [row[1] for row in overall_counts]

    fig_overall = px.pie(
        names=classes,
        values=counts,
        color=classes,
        color_discrete_map={
            "corrected": "#14B8A6",
            "normal": "#0E6BA8",
            "reversal": "#EF4444"
        },
        title="Overall Distribution of Predictions"
    )

    fig_overall.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}: %{value}'
    )

    fig_overall.update_layout(
        title_font_size=22,
        legend_title_text='Prediction Class',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_overall, use_container_width=True)

# -----------------------
# 📈 PREDICTION TRENDS (USER ONLY)
# -----------------------
c.execute("""
    SELECT DATE(timestamp), COUNT(*) 
    FROM Predictions 
    WHERE patient_id IN (
        SELECT patient_id FROM Patients WHERE user_id = ?
    )
    GROUP BY DATE(timestamp)
""", (user_id,))

trend_data = c.fetchall()

if trend_data:
    dates = [row[0] for row in trend_data]
    counts_over_time = [row[1] for row in trend_data]

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=dates,
        y=counts_over_time,
        mode='lines+markers',
        line=dict(color='#0E6BA8', width=3),
        marker=dict(size=8, color='#14B8A6')
    ))

    fig_trend.update_layout(
        title="Predictions Over Time",
        title_font_size=22,
        xaxis_title="Date",
        yaxis_title="Number of Predictions",
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(showgrid=True, gridcolor="#E5E7EB")
    )

    st.plotly_chart(fig_trend, use_container_width=True)


conn.close()
