import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_FILE = "patients.db"
st.set_page_config(page_title="Dyslexia Dashboard", layout="wide")
st.title("📊 Dyslexia Detection Dashboard")

# Initialize session_state for selected patient
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None
    st.session_state.selected_patient_name = None

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# ----------------------
# Patient Cards
# ----------------------
c.execute("SELECT patient_id, name, age, gender FROM Patients")
all_patients = c.fetchall()

st.subheader("👥 Patients Overview")
cols = st.columns(3)
for i, (pid, name, age, gender) in enumerate(all_patients):
    with cols[i % 3]:
        c.execute("SELECT COUNT(*), MAX(timestamp) FROM Predictions WHERE patient_id=?", (pid,))
        total_preds, last_pred = c.fetchone()
        last_pred_str = last_pred if last_pred else "No predictions"
        card_label = f"**{name}**\nAge: {age}, {gender}\nPredictions: {total_preds}\n"
        if st.button(card_label, key=f"btn_{pid}"):
            # Store patient info in session_state
            st.session_state.selected_patient_id = pid
            st.session_state.selected_patient_name = name
            st.rerun()  # reload app to reflect selection

# ----------------------
# Overall Predictions Pie
# ----------------------
c.execute("SELECT prediction_class, COUNT(*) FROM Predictions GROUP BY prediction_class")
overall_counts = c.fetchall()
if overall_counts:
    classes = [row[0] for row in overall_counts]
    counts = [row[1] for row in overall_counts]

    fig_overall = px.pie(
        names=classes,
        values=counts,
        color=classes,
        color_discrete_map={"corrected":"green", "normal":"blue", "reversal":"red"},
        title="Overall Distribution of Predictions"
    )
    fig_overall.update_traces(textinfo='none', hovertemplate='%{label}: %{percent}')
    st.plotly_chart(fig_overall, use_container_width=True)

# ----------------------
# Prediction Trends
# ----------------------
c.execute("SELECT DATE(timestamp), COUNT(*) FROM Predictions GROUP BY DATE(timestamp)")
trend_data = c.fetchall()
if trend_data:
    dates = [row[0] for row in trend_data]
    counts_over_time = [row[1] for row in trend_data]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=dates, y=counts_over_time, mode='lines+markers', line=dict(color='blue')))
    fig_trend.update_layout(title="Predictions Over Time", xaxis_title="Date", yaxis_title="Number of Predictions")
    st.plotly_chart(fig_trend, use_container_width=True)

conn.close()
