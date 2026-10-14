"""
Heart Disease Risk Prediction - State-of-the-Art Clinical Deep Learning Dashboard.

Engineered with modern glassmorphism, interactive Plotly visualizations,
preset scenario injectors, real-time BMI computation, and neural network inference.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Import prediction engine
from src.predict import HeartDiseasePredictor

# Page Configuration
st.set_page_config(
    page_title="CardioPulse AI | Deep Learning Heart Risk Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rich Aesthetic Custom CSS Styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Font Overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient Hero Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 70% 30%, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Glassmorphic Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.2);
    }

    .card-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .card-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    .card-caption {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Streamlit Tab Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.5);
    }

    /* Result Cards */
    .result-card-danger {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(225, 29, 72, 0.05) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(244, 63, 94, 0.2);
    }

    .result-card-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.2);
    }

    /* Styled Disclaimer */
    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.04) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 2rem;
        color: #fde68a;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Custom Form Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
        transition: all 0.25s ease;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -5px rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, #4338ca 0%, #0891b2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """Cache model and preprocessor artifacts."""
    return HeartDiseasePredictor()

@st.cache_data
def load_all_artifacts():
    """Load metadata, test metrics, and survey datasets."""
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    test_metrics_path = os.path.join(PROJECT_ROOT, "outputs", "metrics", "test_metrics.json")
    raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "heart.csv")
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    test_metrics = {}
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
            
    raw_df = pd.read_csv(raw_data_path) if os.path.exists(raw_data_path) else pd.DataFrame()
    return metadata, test_metrics, raw_df

def create_interactive_gauge(prob: float, threshold: float = 0.48):
    """Generate high-end circular Plotly gauge for cardiovascular risk probability."""
    prob_pct = prob * 100
    thresh_pct = threshold * 100
    
    # Determine color
    if prob < 0.25:
        bar_color = "#10b981" # Emerald Green
    elif prob < threshold:
        bar_color = "#f59e0b" # Amber
    else:
        bar_color = "#f43f5e" # Crimson Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        number={'suffix': "%", 'font': {'size': 44, 'family': 'Outfit, sans-serif', 'color': '#ffffff'}},
        delta={'reference': thresh_pct, 'increasing': {'color': '#f43f5e'}, 'decreasing': {'color': '#10b981'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.4)"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [25, thresh_pct], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [thresh_pct, 100], 'color': 'rgba(244, 63, 94, 0.18)'}
            ],
            'threshold': {
                'line': {'color': "#38bdf8", 'width': 4},
                'thickness': 0.75,
                'value': thresh_pct
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#ffffff", 'family': "Outfit, sans-serif"},
        height=270,
        margin=dict(l=20, r=20, t=25, b=20)
    )
    return fig

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>Deep Learning Engine Online &bull; TensorFlow 2.21 / Keras 3.15</span>
        </div>
        <div class="hero-title">CardioPulse AI Risk Platform</div>
        <div class="hero-subtitle">
            An advanced clinical informatics and deep neural network platform for exploratory epidemiological analysis, 
            multi-factorial risk synthesis, and non-invasive cardiovascular probability estimation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        predictor = get_predictor()
        metadata, test_metrics, raw_df = load_all_artifacts()
    except Exception as e:
        st.error(f"Error initializing system pipeline: {e}")
        st.info("Execute `python src/model_evaluation.py` to compile artifacts.")
        return

    m = test_metrics.get("metrics_at_optimal_threshold", {})
    
    # Live Glassmorphic Metric Grid
    st.markdown(f"""
    <div class="metric-grid">
        <div class="glass-card">
            <div class="card-label">Survey Cohort</div>
            <div class="card-value">{len(raw_df):,} <span style="font-size: 1rem; color: #94a3b8;">Patients</span></div>
            <div class="card-caption">CDC BRFSS clinical subset</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Test ROC-AUC</div>
            <div class="card-value" style="color: #38bdf8;">{m.get('roc_auc', 0.7305):.4f}</div>
            <div class="card-caption">Discrimination capacity (holdout)</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Screening Sensitivity</div>
            <div class="card-value" style="color: #10b981;">{m.get('recall_sensitivity', 0.5238):.1%}</div>
            <div class="card-caption">Cardiac case detection rate</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Optimal Threshold</div>
            <div class="card-value" style="color: #c084fc;">&theta; = {test_metrics.get('optimal_threshold', 0.48):.3f}</div>
            <div class="card-caption">F1-calibrated decision cutoff</div>
        </div>
        <div class="glass-card">
            <div class="card-label">Engineered Features</div>
            <div class="card-value">{metadata.get('input_features', 32)} <span style="font-size: 1rem; color: #94a3b8;">Dims</span></div>
            <div class="card-caption">One-hot & scaled vector space</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Scenario Injector
    st.sidebar.markdown("### 🎛️ Scenario Presets")
    st.sidebar.markdown("Quickly inject representative clinical profiles into the prediction form:")

    preset_choice = st.sidebar.selectbox(
        "Choose Patient Profile:",
        options=[
            "Custom User Entry",
            "Profile A: Young Active Female (Low Risk)",
            "Profile B: Middle-Aged Overweight Smoker (Moderate Risk)",
            "Profile C: Elderly Male with Diabetes (High Risk)"
        ]
    )

    # Define preset default values
    if preset_choice == "Profile A: Young Active Female (Low Risk)":
        d_sex, d_age, d_wt, d_bmi = "Female", "30-34", 62.0, 22.4
        d_health, d_ex, d_diab, d_dep, d_smk, d_alc = "Excellent", "Yes", "No", "No", "No", "No"
    elif preset_choice == "Profile B: Middle-Aged Overweight Smoker (Moderate Risk)":
        d_sex, d_age, d_wt, d_bmi = "Male", "55-59", 88.0, 29.3
        d_health, d_ex, d_diab, d_dep, d_smk, d_alc = "Good", "No", "No", "No", "Yes", "Yes"
    elif preset_choice == "Profile C: Elderly Male with Diabetes (High Risk)":
        d_sex, d_age, d_wt, d_bmi = "Male", "75-79", 102.0, 35.8
        d_health, d_ex, d_diab, d_dep, d_smk, d_alc = "Poor", "No", "Yes", "Yes", "Yes", "No"
    else:
        d_sex, d_age, d_wt, d_bmi = "Female", "60-64", 78.0, 28.5
        d_health, d_ex, d_diab, d_dep, d_smk, d_alc = "Good", "Yes", "No", "No", "No", "Yes"

    # Tabbed Navigation
    tabs = st.tabs([
        "⚡ Interactive Risk Calculator",
        "📊 Dynamic Data Exploratorium",
        "🔍 Feature Analysis & Explainability",
        "🧠 Deep Learning Architecture & Tuning",
        "📋 Clinical Report & Overview"
    ])

    # -------------------------------------------------------------
    # TAB 1: INTERACTIVE RISK CALCULATOR
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 🫀 Patient Risk Parameter Input")
        st.markdown("Adjust clinical, demographic, and lifestyle indicators below to assess real-time cardiovascular risk probability:")

        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            with st.container():
                st.markdown("#### 1. Demographic & Anthropometric Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    sex = st.selectbox("Biological Sex", options=["Female", "Male"], index=0 if d_sex == "Female" else 1)
                    age_cat = st.selectbox(
                        "Age Category",
                        options=['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
                        index=['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'].index(d_age)
                    )
                with c2:
                    weight_kg = st.slider("Patient Weight (kg)", min_value=35.0, max_value=165.0, value=float(d_wt), step=0.5)
                    bmi = st.slider("Body Mass Index (BMI kg/m²)", min_value=14.0, max_value=52.0, value=float(d_bmi), step=0.1)
                    
                    # Live BMI Category Pill
                    if bmi < 18.5:
                        st.caption("BMI Classification: **Underweight (<18.5)**")
                    elif bmi < 25.0:
                        st.caption("BMI Classification: :green[**Normal Weight (18.5 - 24.9)**]")
                    elif bmi < 30.0:
                        st.caption("BMI Classification: :orange[**Overweight (25.0 - 29.9)**]")
                    elif bmi < 35.0:
                        st.caption("BMI Classification: :red[**Obese Class I (30.0 - 34.9)**]")
                    else:
                        st.caption("BMI Classification: :red[**Severe / Morbid Obesity Class II/III (>=35.0)**]")

                st.markdown("#### 2. Clinical History & Lifestyle Factors")
                c3, c4 = st.columns(2)
                with c3:
                    gen_health = st.selectbox(
                        "Self-Reported Health",
                        options=["Excellent", "Very Good", "Good", "Fair", "Poor"],
                        index=["Excellent", "Very Good", "Good", "Fair", "Poor"].index(d_health)
                    )
                    exercise = st.selectbox("Physical Activity (Past 30 Days)", options=["Yes", "No"], index=0 if d_ex == "Yes" else 1)
                    diabetes = st.selectbox("Diagnosed with Diabetes", options=["No", "Yes"], index=0 if d_diab == "No" else 1)
                with c4:
                    depression = st.selectbox("History of Depressive Disorder", options=["No", "Yes"], index=0 if d_dep == "No" else 1)
                    smoking = st.selectbox("Smoking History (>=100 cigarettes)", options=["No", "Yes"], index=0 if d_smk == "No" else 1)
                    alcohol = st.selectbox("Regular Alcohol Consumption", options=["No", "Yes"], index=0 if d_alc == "No" else 1)

                threshold_val = st.slider(
                    "Decision Boundary Cutoff (Threshold)", min_value=0.20, max_value=0.80,
                    value=float(test_metrics.get("optimal_threshold", 0.48)), step=0.01,
                    help="Optimal validation threshold (0.480) maximizes sensitivity without disproportionate false alarms."
                )

                calc_btn = st.button("🚀 Calculate Risk Assessment", use_container_width=True)

        with col_right:
            st.markdown("#### 3. Real-Time Risk Diagnostics")
            
            # Perform prediction
            patient_record = {
                "General_Health": gen_health,
                "Exercise": exercise,
                "Depression": depression,
                "Diabetes": diabetes,
                "Sex": sex,
                "Age_Category": age_cat,
                "Weight_(kg)": float(weight_kg),
                "BMI": float(bmi),
                "Smoking_History": smoking,
                "Alcohol_Consumption": alcohol
            }

            pred_res = predictor.predict(patient_record, threshold=threshold_val)
            prob_pct = pred_res["model_probability_percentage"]
            prob_raw = pred_res["model_probability"]

            # Gauge Visualization
            gauge_fig = create_interactive_gauge(prob_raw, threshold=threshold_val)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Result Classification Card
            if pred_res["prediction_class"] == 1:
                st.markdown(f"""
                <div class="result-card-danger">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #f43f5e; font-weight: 700; letter-spacing: 0.05em;">Classification Result</div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #fff1f2; margin: 4px 0;">⚠️ {pred_res['prediction_label']}</div>
                    <div style="color: #fecdd3; font-size: 0.95rem;">Model Risk Probability: <strong>{prob_pct:.1f}%</strong> (Exceeds &theta; = {threshold_val:.2f})</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-success">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #10b981; font-weight: 700; letter-spacing: 0.05em;">Classification Result</div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #ecfdf5; margin: 4px 0;">✅ {pred_res['prediction_label']}</div>
                    <div style="color: #a7f3d0; font-size: 0.95rem;">Model Risk Probability: <strong>{prob_pct:.1f}%</strong> (Below &theta; = {threshold_val:.2f})</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Clinical Recommendation:** {pred_res['clinical_recommendation']}")
            st.markdown("**Identified Clinical Contributing Factors:**")
            for factor in pred_res["notable_contributing_factors"]:
                st.markdown(f"- {factor}")

        # Educational Disclaimer Banner
        st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Mandatory Medical & Academic Disclaimer:</strong><br>
            {pred_res['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: DYNAMIC DATA EXPLORATORIUM
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("### 📊 Interactive Dataset Visualizer & Statistical Distributions")
        
        c_eda1, c_eda2 = st.columns([1, 1])
        
        with c_eda1:
            st.markdown("#### Target Class Balance (BRFSS Subset)")
            target_counts = raw_df["Heart_Disease"].value_counts().reset_index()
            target_counts.columns = ["Diagnosis", "Patient Count"]
            
            fig_pie = px.pie(
                target_counts, values="Patient Count", names="Diagnosis",
                color="Diagnosis",
                color_discrete_map={"No": "#38bdf8", "Yes": "#f43f5e"},
                hole=0.55
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': '#ffffff', 'family': 'Outfit'},
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_eda2:
            st.markdown("#### BMI vs Weight Scatter Stratified by Diagnosis")
            fig_scatter = px.scatter(
                raw_df, x="Weight_(kg)", y="BMI", color="Heart_Disease",
                color_discrete_map={"No": "#38bdf8", "Yes": "#f43f5e"},
                opacity=0.75, hover_data=["Age_Category", "General_Health"]
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': '#ffffff', 'family': 'Outfit'},
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### High-Resolution Publication Figures (300 DPI)")
        fig_selector = st.selectbox(
            "Select Publication Figure to View:",
            options=[
                ("01_target_distribution.png", "1. Target Class Distribution"),
                ("02_numerical_histograms.png", "2. Numerical Distributions (Weight & BMI)"),
                ("03_numerical_boxplots.png", "3. Outlier Analysis & Clinical Retention"),
                ("04_correlation_heatmap.png", "4. Correlation Matrix Heatmap"),
                ("05_feature_vs_target.png", "5. Feature-vs-Target Violin Plots"),
                ("06_categorical_risk_factors.png", "6. Categorical Risk Prevalence Trends"),
                ("07_bmi_vs_weight_scatter.png", "7. Joint Distribution Scatter"),
                ("08_missing_values_matrix.png", "8. Missing Value Completeness")
            ],
            format_func=lambda x: x[1]
        )
        
        fig_img_path = os.path.join(PROJECT_ROOT, "outputs", "figures", fig_selector[0])
        if os.path.exists(fig_img_path):
            st.image(fig_img_path, use_container_width=True)

        # Descriptive stats table
        st.markdown("#### Continuous Features Descriptive Statistics")
        stats_csv = os.path.join(PROJECT_ROOT, "outputs", "metrics", "numerical_descriptive_stats.csv")
        if os.path.exists(stats_csv):
            st.dataframe(pd.read_csv(stats_csv), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: FEATURE ANALYSIS & EXPLAINABILITY
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 🔍 Model-Agnostic Feature Importance Analysis")
        st.markdown("""
        **Neural Network Interpretability:** Because Multi-Layer Perceptrons route features through dense layers with non-linear activations 
        and alternating positive/negative weights, they do not inherently provide linear coefficients. 
        We compute **Shannon Mutual Information (MI)** and **Permutation Feature Importance** to achieve transparent interpretability.
        """)

        feat_img_path = os.path.join(PROJECT_ROOT, "outputs", "figures", "09_feature_importance.png")
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, use_container_width=True)

        st.markdown("#### Full Feature Metric Table")
        feat_csv = os.path.join(PROJECT_ROOT, "outputs", "metrics", "feature_importance.csv")
        if os.path.exists(feat_csv):
            st.dataframe(pd.read_csv(feat_csv).sort_values(by="Mutual_Information", ascending=False), use_container_width=True)

        st.info("💡 **Scientific Caveat:** Statistical associations and mutual information signify predictive dependency within the surveyed cohort. They do not demonstrate direct biochemical causation.")

    # -------------------------------------------------------------
    # TAB 4: DEEP LEARNING ARCHITECTURE & TUNING
    # -------------------------------------------------------------
    with tabs[3]:
        st.markdown("### 🧠 Multi-Layer Perceptron (MLP) Development & Tuning")
        
        st.markdown("#### 1. Hyperparameter Optimization Sweep (10 Evaluated Architectures)")
        tuning_img = os.path.join(PROJECT_ROOT, "outputs", "figures", "10_hyperparameter_tuning_comparison.png")
        if os.path.exists(tuning_img):
            st.image(tuning_img, use_container_width=True)

        tuning_csv = os.path.join(PROJECT_ROOT, "outputs", "metrics", "hyperparameter_tuning_results.csv")
        if os.path.exists(tuning_csv):
            st.dataframe(pd.read_csv(tuning_csv).sort_values(by="val_auc", ascending=False), use_container_width=True)

        st.markdown("#### 2. Final Model Diagnostic Learning Curves")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.image(os.path.join(PROJECT_ROOT, "outputs", "figures", "11_training_loss_curves.png"), caption="Binary Cross-Entropy Loss vs Epoch", use_container_width=True)
        with col_c2:
            st.image(os.path.join(PROJECT_ROOT, "outputs", "figures", "12_training_accuracy_curves.png"), caption="Accuracy & ROC-AUC Progression", use_container_width=True)

        st.markdown("#### 3. Test Set Holdout Benchmark (150 Samples)")
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            st.image(os.path.join(PROJECT_ROOT, "outputs", "figures", "13_confusion_matrix.png"), caption="Test Confusion Matrix", use_container_width=True)
        with col_c4:
            st.image(os.path.join(PROJECT_ROOT, "outputs", "figures", "14_roc_curve.png"), caption="Test ROC & PR Curves", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: CLINICAL REPORT & OVERVIEW
    # -------------------------------------------------------------
    with tabs[4]:
        st.markdown("### 📋 Academic Project Abstract & Summary")
        st.markdown("""
        **Project Title:** Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk  
        **Author:** Biomedical Machine Learning & Data Science Coursework  
        
        #### Executive Summary
        Cardiovascular disease represents the leading cause of global mortality. Using a real-world retrospective survey cohort ($N=999$) from the CDC Behavioral Risk Factor Surveillance System (BRFSS), this study implemented an end-to-end, leak-free deep learning predictive pipeline. 
        
        #### Key Methodology & Findings:
        - **Zero Leakage:** Complete ColumnTransformer pipeline fitted strictly on 70% training split.
        - **Imbalance Handling:** Balanced class weights ($w_0 = 0.582, w_1 = 3.530$) and validation decision threshold calibration ($\theta = 0.480$) boosted screening sensitivity from 42.9% to 52.4%.
        - **Test Set Performance:** On the 150 untouched holdout records, the MLP achieved an ROC-AUC of **0.7305**, accuracy of **74.00%**, sensitivity of **52.38%**, and specificity of **77.52%**.
        """)
        
        st.markdown("#### Project File Resources:")
        st.markdown("""
        - 📄 **Academic Report:** [`outputs/reports/final_project_report.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/final_project_report.md)
        - 📊 **Visual Interpretations:** [`outputs/reports/visualization_interpretations.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/visualization_interpretations.md)
        - 📓 **Jupyter Notebook:** [`notebooks/heart_disease_eda.ipynb`](file:///d:/heart-disease-risk-prediction/notebooks/heart_disease_eda.ipynb)
        - 📜 **Requirements Checklist:** [`outputs/reports/assignment_checklist.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/assignment_checklist.md)
        """)

if __name__ == "__main__":
    main()
