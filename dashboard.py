import streamlit as st
import pandas as pd
import joblib
import os
from fairlearn.metrics import demographic_parity_difference

# --- Page Configuration ---
st.set_page_config(
    page_title="Placement Prediction & Fairness Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Vivid Custom CSS Styling ---
st.markdown("""
    <style>
    /* Main Header Banner */
    .banner-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .banner-subtitle {
        font-size: 1.1rem;
        color: #E2E8F0;
        margin-top: 5px;
    }
    /* Metric Card Styling */
    .metric-container {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Colored Header Banner ---
st.markdown("""
    <div class="banner-box">
        <p class="banner-title">🎯 Placement Prediction & Algorithmic Fairness Dashboard</p>
        <p class="banner-subtitle">Enterprise-grade ML evaluation pipeline auditing and mitigating demographic bias using Fairlearn.</p>
    </div>
""", unsafe_allow_html=True)

# --- Safe Artifact Loading ---
if not os.path.exists("dataset/placement_data.csv") or not os.path.exists("models/fair_model.pkl"):
    st.error("⚠️ Model files or dataset not found! Please run `generate_data.py` and `train_and_mitigate.py` first.")
else:
    @st.cache_resource
    def load_artifacts():
        df = pd.read_csv("dataset/placement_data.csv")
        base_model = joblib.load("models/base_model.pkl")
        fair_model = joblib.load("models/fair_model.pkl")
        return df, base_model, fair_model

    df, base_model, fair_model = load_artifacts()

    # --- Sidebar Controls ---
    st.sidebar.header("🔍 Student Profile Simulation")
    st.sidebar.markdown("Adjust parameters to test real-time model behavior:")
    
    cgpa = st.sidebar.slider("CGPA", 4.0, 10.0, 7.8, 0.1)
    internships = st.sidebar.slider("Internships", 0, 5, 2, 1)
    technical_score = st.sidebar.slider("Technical Score", 30.0, 100.0, 80.0, 1.0)
    soft_skills = st.sidebar.slider("Soft Skills Rating", 1, 10, 8, 1)
    gender = st.sidebar.selectbox("Gender Demographic", ["Male", "Female"])

    # --- Prepare Input Data ---
    gender_male = 1 if gender == "Male" else 0
    input_data = pd.DataFrame([[cgpa, internships, technical_score, soft_skills, gender_male]], 
                              columns=["CGPA", "Internships", "TechnicalScore", "SoftSkills", "Gender_Male"])

    # --- Main Dashboard Tabs ---
    tab1, tab2 = st.tabs(["🚀 Live Prediction & Fairness Audit", "📊 Dataset Analytics & Distribution"])

    with tab1:
        st.subheader("📊 Comparative Model Prediction")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Standard Base Model")
            pred_base = base_model.predict(input_data)[0]
            prob_base = base_model.predict_proba(input_data)[0][1]
            if pred_base == 1:
                st.success(f"**Prediction:** Placed 🎯\n\n**Probability:** {prob_base:.2f}")
            else:
                st.error(f"**Prediction:** Not Placed ❌\n\n**Probability:** {prob_base:.2f}")

        with col2:
            st.markdown("### ⚖️ Fairlearn Mitigated Model")
            pred_fair = fair_model.predict(input_data)[0]
            if pred_fair == 1:
                st.success("**Prediction:** Placed 🎯\n\n**Status:** Bias Mitigated & Fair")
            else:
                st.error("**Prediction:** Not Placed ❌\n\n**Status:** Bias Mitigated & Fair")

        st.markdown("---")
        st.subheader("⚖️ Overall Demographic Parity Metrics")

        df_encoded = pd.get_dummies(df, columns=["Gender"], drop_first=True)
        X = df_encoded.drop(columns=["Placed"])
        y = df_encoded["Placed"]
        A = df["Gender"]

        y_pred_base_all = base_model.predict(X)
        y_pred_fair_all = fair_model.predict(X)

        dp_base = demographic_parity_difference(y, y_pred_base_all, sensitive_features=A)
        dp_fair = demographic_parity_difference(y, y_pred_fair_all, sensitive_features=A)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Base Parity Gap", f"{dp_base:.4f}", help="Unmitigated disparity score")
        mcol2.metric("Mitigated Parity Gap", f"{dp_fair:.4f}", delta=f"-{(dp_base - dp_fair):.4f} Bias Reduction", delta_color="normal")
        mcol3.metric("Fairness Status", "Optimized ✅", "Constraint Satisfied")

    with tab2:
        st.subheader("📁 Comprehensive Dataset Explorer")
        st.markdown("A preview of the synthetic training and evaluation dataset records:")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Quick Statistical Summary")
        st.write(df.describe())