import streamlit as st
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt

# Load models safely
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("label_encoder_sex.pkl")
    le_smoker = joblib.load("label_encoder_smoker.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, le_gender, le_smoker, model

scaler, le_gender, le_smoker, model = load_models()

# Load SVG icon dynamically
svg_path = os.path.join(os.path.dirname(__file__), "utils", "health-svgrepo-com.svg")
svg_icon = ""
if os.path.exists(svg_path):
    with open(svg_path, "r") as f:
        svg_icon = f.read()
    svg_icon = svg_icon.replace('width="800px"', 'width="40px"').replace('height="800px"', 'height="40px"')

st.set_page_config(
    page_title="Health Premium Estimator",
    page_icon=svg_path if svg_icon else "⚕️",
    layout="wide", # Changed to wide to fit the analytics dashboard
    initial_sidebar_state="collapsed"
)

# ----------------- SESSION STATE -----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "user_metrics" not in st.session_state:
    st.session_state.user_metrics = None

def navigate_to(page_name):
    st.session_state.page = page_name

# Global Base CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

/* Move everything a bit upwards */
div.block-container {
    padding-top: 1.5rem !important;
}

/* Floating Sticker Animations */
.stickers-wrapper {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
    -webkit-mask-image: radial-gradient(ellipse at center, rgba(0,0,0,0.05) 30%, rgba(0,0,0,1) 50%);
    mask-image: radial-gradient(ellipse at center, rgba(0,0,0,0.05) 30%, rgba(0,0,0,1) 50%);
}
.floating-sticker {
    position: absolute;
    opacity: 0.9;
    will-change: transform;
}
.s1 { top: 15%; left: -100px; animation: flyRight 20s linear infinite; }
.s2 { top: 65%; right: -100px; animation: flyLeft 25s linear infinite 5s; }
.s3 { bottom: -100px; left: 30%; animation: flyUp 22s linear infinite 2s; }
.s4 { top: -100px; right: 25%; animation: flyDown 18s linear infinite 8s; }
.s5 { top: 40%; left: -100px; animation: flyRight 24s linear infinite 12s; }

@keyframes flyRight {
    0% { transform: translate(0, 0) scale(0.8) rotate(-15deg); }
    50% { transform: translate(60vw, -10vh) scale(1.1) rotate(10deg); }
    100% { transform: translate(120vw, 20vh) scale(0.8) rotate(-15deg); }
}
@keyframes flyLeft {
    0% { transform: translate(0, 0) scale(1) rotate(15deg); }
    50% { transform: translate(-60vw, 15vh) scale(0.9) rotate(-5deg); }
    100% { transform: translate(-120vw, -10vh) scale(1) rotate(15deg); }
}
@keyframes flyUp {
    0% { transform: translate(0, 0) scale(0.9) rotate(0deg); }
    50% { transform: translate(15vw, -60vh) scale(1.2) rotate(-15deg); }
    100% { transform: translate(-10vw, -120vh) scale(0.9) rotate(5deg); }
}
@keyframes flyDown {
    0% { transform: translate(0, 0) scale(1) rotate(5deg); }
    50% { transform: translate(-15vw, 60vh) scale(0.8) rotate(20deg); }
    100% { transform: translate(10vw, 120vh) scale(1.1) rotate(-10deg); }
}
</style>
""", unsafe_allow_html=True)

# Define the HTML block for the stickers
stickers_html = """
<div class="stickers-wrapper">
    <!-- Heart with wings and plus sign -->
    <svg class="floating-sticker s1" viewBox="0 0 100 100" width="60" height="60" xmlns="http://www.w3.org/2000/svg">
      <path d="M40 40 C 20 20, 0 30, 10 60 C 20 70, 35 55, 45 50 Z" fill="#e2e8f0"/>
      <path d="M60 40 C 80 20, 100 30, 90 60 C 80 70, 65 55, 55 50 Z" fill="#e2e8f0"/>
      <path d="M 50 35 A 15 15 0 0 0 20 45 C 20 65 50 85 50 85 C 50 85 80 65 80 45 A 15 15 0 0 0 50 35 Z" fill="#ef4444"/>
      <path d="M 48 50 h 4 v 10 h -4 z" fill="white"/>
      <path d="M 43 53 h 14 v 4 h -14 z" fill="white"/>
    </svg>
    <svg class="floating-sticker s2" viewBox="0 0 100 100" width="50" height="50" xmlns="http://www.w3.org/2000/svg">
      <path d="M40 40 C 20 20, 0 30, 10 60 C 20 70, 35 55, 45 50 Z" fill="#e2e8f0"/>
      <path d="M60 40 C 80 20, 100 30, 90 60 C 80 70, 65 55, 55 50 Z" fill="#e2e8f0"/>
      <path d="M 50 35 A 15 15 0 0 0 20 45 C 20 65 50 85 50 85 C 50 85 80 65 80 45 A 15 15 0 0 0 50 35 Z" fill="#ef4444"/>
      <path d="M 48 50 h 4 v 10 h -4 z" fill="white"/>
      <path d="M 43 53 h 14 v 4 h -14 z" fill="white"/>
    </svg>
    <svg class="floating-sticker s3" viewBox="0 0 100 100" width="70" height="70" xmlns="http://www.w3.org/2000/svg">
      <path d="M40 40 C 20 20, 0 30, 10 60 C 20 70, 35 55, 45 50 Z" fill="#cbd5e1"/>
      <path d="M60 40 C 80 20, 100 30, 90 60 C 80 70, 65 55, 55 50 Z" fill="#cbd5e1"/>
      <path d="M 50 35 A 15 15 0 0 0 20 45 C 20 65 50 85 50 85 C 50 85 80 65 80 45 A 15 15 0 0 0 50 35 Z" fill="#f43f5e"/>
      <path d="M 48 50 h 4 v 10 h -4 z" fill="white"/>
      <path d="M 43 53 h 14 v 4 h -14 z" fill="white"/>
    </svg>
    <svg class="floating-sticker s4" viewBox="0 0 100 100" width="45" height="45" xmlns="http://www.w3.org/2000/svg">
      <path d="M40 40 C 20 20, 0 30, 10 60 C 20 70, 35 55, 45 50 Z" fill="#e2e8f0"/>
      <path d="M60 40 C 80 20, 100 30, 90 60 C 80 70, 65 55, 55 50 Z" fill="#e2e8f0"/>
      <path d="M 50 35 A 15 15 0 0 0 20 45 C 20 65 50 85 50 85 C 50 85 80 65 80 45 A 15 15 0 0 0 50 35 Z" fill="#ef4444"/>
      <path d="M 48 50 h 4 v 10 h -4 z" fill="white"/>
      <path d="M 43 53 h 14 v 4 h -14 z" fill="white"/>
    </svg>
    <svg class="floating-sticker s5" viewBox="0 0 100 100" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
      <path d="M40 40 C 20 20, 0 30, 10 60 C 20 70, 35 55, 45 50 Z" fill="#cbd5e1"/>
      <path d="M60 40 C 80 20, 100 30, 90 60 C 80 70, 65 55, 55 50 Z" fill="#cbd5e1"/>
      <path d="M 50 35 A 15 15 0 0 0 20 45 C 20 65 50 85 50 85 C 50 85 80 65 80 45 A 15 15 0 0 0 50 35 Z" fill="#fb7185"/>
      <path d="M 48 50 h 4 v 10 h -4 z" fill="white"/>
      <path d="M 43 53 h 14 v 4 h -14 z" fill="white"/>
    </svg>
</div>
"""
# We only define the string here, it gets rendered inside the 'home' and 'landing' page blocks.


# ==========================================
# PAGE 0: LANDING PAGE
# ==========================================
if st.session_state.page == "landing":

    # Render floating stickers on the landing page too
    st.markdown(stickers_html, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .lp-icon-box { background-color: #1e293b; width: 72px; height: 72px; border-radius: 18px; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(0,0,0,0.2); margin-bottom: 24px; }
    .lp-title { color: #ffffff; font-size: 46px; font-weight: 800; letter-spacing: -1px; line-height: 1.15; margin-bottom: 16px; }
    .lp-title span { color: #3b82f6; }
    .lp-subtitle { color: #94a3b8; font-size: 18px; font-weight: 400; max-width: 550px; line-height: 1.6; margin-bottom: 50px; }

    /* How It Works Steps */
    .steps-label { color: #64748b; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px; }
    .steps-row { display: flex; gap: 24px; margin-bottom: 50px; }
    .step-card { background-color: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 28px 24px; width: 200px; text-align: center; }
    .step-num { color: #3b82f6; font-size: 14px; font-weight: 700; margin-bottom: 10px; }
    .step-icon { font-size: 28px; margin-bottom: 12px; }
    .step-text { color: #e2e8f0; font-size: 15px; font-weight: 600; line-height: 1.4; }

    /* CTA Button */
    .lp-cta div.stButton > button:first-child {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 18px 60px !important;
        border: none !important;
        font-size: 20px !important;
        box-shadow: 0px 6px 20px rgba(59, 130, 246, 0.35) !important;
    }
    .lp-cta div.stButton > button:first-child:hover {
        background-color: #2563eb !important;
        box-shadow: 0px 10px 28px rgba(59, 130, 246, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
            <div class="lp-icon-box">{}</div>
        """.format(svg_icon.replace('width="40px"', 'width="44px"').replace('height="40px"', 'height="44px"')), unsafe_allow_html=True)

        st.markdown("""
            <div class="lp-title">Know Your Health<br><span>Insurance Premium</span></div>
            <div class="lp-subtitle">Enter a few details about yourself and our AI model will instantly predict your annual health insurance charges with high accuracy.</div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="steps-label">How It Works</div>
            <div class="steps-row">
                <div class="step-card">
                    <div class="step-num">STEP 1</div>
                    <div class="step-icon">📝</div>
                    <div class="step-text">Fill in your health profile</div>
                </div>
                <div class="step-card">
                    <div class="step-num">STEP 2</div>
                    <div class="step-icon">⚙️</div>
                    <div class="step-text">AI calculates your risk score</div>
                </div>
                <div class="step-card">
                    <div class="step-num">STEP 3</div>
                    <div class="step-icon">💰</div>
                    <div class="step-text">Get your premium estimate</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="lp-cta">', unsafe_allow_html=True)
        if st.button("Get Started"):
            navigate_to("home")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
# ==========================================
if st.session_state.page == "home":
    
    # Render floating stickers only on the Home page
    st.markdown(stickers_html, unsafe_allow_html=True)
    
    # Inject Home specific CSS (Light Theme, Clean SaaS)
    st.markdown("""
    <style>
    /* Center the title properly */
    .header-container { text-align: center; padding: 20px 0 30px 0; margin: 0 auto; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .header-title { color: #ffffff; font-size: 38px; font-weight: 700; margin-bottom: 12px; letter-spacing: -0.5px; text-align: center; width: 100%; }
    .header-subtitle { color: #94a3b8; font-size: 18px; font-weight: 400; max-width: 600px; text-align: center; margin: 0 auto; }
    .custom-subheader { color: #e2e8f0; font-size: 20px; font-weight: 600; margin-top: 15px; margin-bottom: -10px; display: flex; align-items: center; }
    .custom-subheader span { background-color: #3b82f6; color: #ffffff; width: 28px; height: 28px; font-size: 15px; display: flex; align-items: center; justify-content: center; border-radius: 50%; margin-right: 12px; }
    
    /* Input Styling - Make Everything Bigger */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { transition: border-color 0.2s ease, box-shadow 0.2s ease !important; padding: 6px !important; font-size: 16px !important; }
    div[data-baseweb="input"] > div:hover, div[data-baseweb="select"] > div:hover { border-color: #3b82f6 !important; }
    label { font-size: 16px !important; font-weight: 500 !important; }

    /* Button Styling overrides */
    div.stButton > button:first-child { background-color: #3b82f6 !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; padding: 16px 24px !important; border: none !important; width: 100% !important; font-size: 18px !important; margin-top: 20px; }
    div.stButton > button:first-child:hover { background-color: #2563eb !important; }
    .btn-secondary div.stButton > button:first-child { background-color: #1e293b !important; color: #3b82f6 !important; border: 1px solid #334155 !important; padding: 14px 24px !important; }
    .btn-secondary div.stButton > button:first-child:hover { background-color: #334155 !important; border-color: #475569 !important; }
    
    /* Result Card */
    .result-card { background-color: #1e293b; border: 1px solid #334155; padding: 35px 25px; border-radius: 12px; margin-top: 20px; text-align: center; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); }
    .result-label { color: #94a3b8; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
    .result-price { color: #ffffff; font-size: 52px; font-weight: 700; line-height: 1.1; margin-bottom: 10px; }
    .result-range { color: #cbd5e1; font-size: 16px; opacity: 0.9; margin-bottom: 18px; }
    .confidence-badge { display: inline-block; background-color: rgba(59, 130, 246, 0.1); color: #60a5fa; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.2); }
    
    .trust-badge-container { text-align: center; margin-top: 24px; margin-bottom: 25px; }
    .trust-badge { display: inline-flex; align-items: center; color: #94A3B8; font-size: 14px; font-weight: 500; }
    .trust-badge svg { margin-right: 8px; color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)
    
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2.5, 1])
    with col_center:
        st.markdown(f"""
        <div class="header-container">
            <div style="margin-bottom: 12px; display: flex; justify-content: center;">
                <div style="background-color: #E0E7FF; width: 64px; height: 64px; border-radius: 16px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(30, 58, 138, 0.1);">
                    {svg_icon}
                </div>
            </div>
            <div class="header-title">Health Insurance Charges Predictor</div>
            <div class="header-subtitle">Welcome! Enter your health profile below to get an instant actuarial premium estimate.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("insurance_form", clear_on_submit=False):
            
            st.markdown("<div class='custom-subheader'><span>1</span> User Details</div>", unsafe_allow_html=True)
            st.write("")
            
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("🎂 Age (Years)", min_value=18, max_value=100, value=30, help="Your current age.")
                with col2:
                    gender = st.selectbox("🚻 Biological Sex", options=le_gender.classes_)

            st.write("")
            
            st.markdown("<div class='custom-subheader'><span>2</span> Medical & Lifestyle Profile</div>", unsafe_allow_html=True)
            st.write("")
            
            with st.container(border=True):
                col3, col4 = st.columns(2)
                with col3:
                    bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, help="Normal range is typically 18.5 - 24.9")
                with col4:
                    smoker = st.selectbox("🚬 Do you smoke?", options=le_smoker.classes_, help="Tobacco use significantly impacts premium rates.")
                
                children = st.number_input("👶 Number of Dependents (Children)", min_value=0, max_value=10, value=0)

            submitted = st.form_submit_button("Calculate Estimated Premium")

        if submitted:
            with st.spinner("Calculating risk profile and estimating premium..."):
                time.sleep(1.2)
                
                input_dict = { "age": [age], "sex": [gender], "bmi": [bmi], "children": [children], "smoker": [smoker] }
                input_data = pd.DataFrame(input_dict)

                input_data["sex"] = le_gender.transform(input_data["sex"])
                input_data["smoker"] = le_smoker.transform(input_data["smoker"])
                num_cols = ["age", "bmi", "children"]
                input_data[num_cols] = scaler.transform(input_data[num_cols])

                expected_cols = model.feature_names_in_
                input_data = input_data.reindex(columns=expected_cols)

                prediction = model.predict(input_data)[0]

                # Save globally for next page
                st.session_state.prediction = prediction
                st.session_state.user_metrics = {
                    "age": age, "gender": gender, "bmi": bmi, "smoker": smoker, "children": children
                }

        # Show result card & analysis button if a prediction exists (persists across reruns)
        if st.session_state.prediction is not None:
            pred = st.session_state.prediction
            lower_bound = pred * 0.92
            upper_bound = pred * 1.08

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Annual Premium</div>
                <div class="result-price">₹{pred:,.0f}</div>
                <div class="result-range">Expected Range: ₹{lower_bound:,.0f} – ₹{upper_bound:,.0f}</div>
                <div class="confidence-badge">✓ 87% Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="trust-badge-container">
                <span class="trust-badge">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                    Premium computed using AI-powered actuarial models
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("See Your Analysis"):
                navigate_to("analysis")
                st.rerun()


# ==========================================
# PAGE 2: ANALYSIS DASHBOARD (DARK THEME)
# ==========================================
elif st.session_state.page == "analysis":

    # Inject Analysis specific CSS (Dark Theme Dashboard matching screenshot)
    st.markdown("""
    <style>
    /* Top Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid #334155;
    }
    .metric-header {
        color: #94a3b8 !important;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff !important;
    }

    /* Risk Indicator Progress Bar */
    .risk-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 20px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .progress-track {
        background-color: #334155;
        border-radius: 8px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .progress-fill {
        background-color: #3b82f6; /* Bright blue */
        height: 100%;
        border-radius: 8px;
    }
    .risk-tag {
        background-color: #3f4531; /* Olive green overlay */
        color: #d9f99d !important;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        display: inline-block;
        border: 1px solid #4d573d;
        width: 100%;
    }
    .risk-tag.warning {
        background-color: #452222; 
        color: #fca5a5 !important;
        border-color: #5c2b2b;
    }

    /* Target Headers */
    .chart-header {
        font-size: 16px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Button Override for Dark Theme */
    div.stButton > button:first-child { 
        background-color: #3b82f6 !important; 
        color: white !important; 
        border-radius: 6px !important; 
        border: none !important; 
        margin-top: 30px;
    }
    div.stButton > button:first-child:hover { background-color: #2563eb !important; }
    
    </style>
    """, unsafe_allow_html=True)

    metrics = st.session_state.user_metrics
    
    # Evaluate Risk
    if metrics["bmi"] < 18.5: bmi_status = "Underweight"
    elif metrics["bmi"] < 25: bmi_status = "Normal"
    elif metrics["bmi"] < 30: bmi_status = "Overweight"
    else: bmi_status = "Obese"

    risk_score = (metrics["age"] * 0.3 + metrics["bmi"] * 0.4 + (1 if metrics["smoker"] == "yes" else 0) * 30)
    risk_percent = min(int(risk_score), 100)

    # 1. TOP METRICS WIDGETS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">👤 AGE</div>
            <div class="metric-value">{metrics['age']} Yrs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">⚖️ BMI CLASS</div>
            <div class="metric-value">{bmi_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">🚬 SMOKER</div>
            <div class="metric-value">{"Yes" if metrics['smoker'] == "yes" else "No"}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">⚠️ RISK SCORE</div>
            <div class="metric-value">{risk_score:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. HEALTH RISK INDICATOR BAR
    st.markdown('<div class="risk-header">🎯 Health Risk Indicator</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="progress-track">
        <div class="progress-fill" style="width: {risk_percent}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    if risk_percent < 35:
        tag_class = ""
        tag_text = "Low Risk Profile ✓"
    elif risk_percent > 70:
        tag_class = "warning"
        tag_text = "High Risk Profile 🚨"
    else:
        tag_class = ""
        tag_text = "Moderate Risk Profile ⚠️"

    st.markdown(f'<div class="risk-tag {tag_class}">{tag_text}</div>', unsafe_allow_html=True)

    # 3. CHARTS SECTION
    colChart1, colChart2 = st.columns(2)

    with colChart1:
        st.markdown('<div class="chart-header">📊 BMI Category Comparison</div>', unsafe_allow_html=True)
        categories = ["Underweight", "Normal", "Overweight", "Obese"]
        ranges = [18.5, 24.9, 29.9, 40] # Matched to reference image ceilings
        colors = ['#38bdf8', '#4ade80', '#facc15', '#fb7185'] # Light Blue, Green, Yellow, Coral
        
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')
        
        bars = ax1.bar(categories, ranges, color=colors, width=0.6)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, ranges):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     str(val), ha='center', color='white', fontsize=9)
        
        # User BMI Dotted Line
        ax1.axhline(y=metrics["bmi"], color='#06b6d4', linestyle=":", linewidth=2)
        ax1.text(3.5, metrics["bmi"] + 1, f'Your BMI: {metrics["bmi"]}', color='#06b6d4', fontsize=9, ha='right')
        
        ax1.set_ylabel("BMI Value", color='white', fontsize=10)
        ax1.tick_params(colors='white', labelsize=9)
        for spine in ax1.spines.values():
            spine.set_color('#334155')
        ax1.yaxis.grid(True, color='#334155', linestyle='-', alpha=0.5)
        
        st.pyplot(fig1)

    with colChart2:
        st.markdown('<div class="chart-header">💰 Insurance Cost Drivers</div>', unsafe_allow_html=True)
        drivers = ["Age Factor", "BMI Factor", "Children Factor"]
        values = [metrics["age"] * 0.2, metrics["bmi"] * 0.3, metrics["children"] * 5]
        colors = ['#818cf8', '#f472b6', '#34d399'] # Periwinkle, Pink, Mint
        
        if metrics["smoker"] == "yes":
            drivers.append("Smoking Penalty")
            values.append(30)
            colors.append('#ef4444') # Red for smoker penalty
            
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#0f172a')
        
        bars2 = ax2.bar(drivers, values, color=colors, width=0.5)
        
        # Add value labels
        for bar, val in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f'{val:.1f}', ha='center', color='white', fontsize=9)
                     
        ax2.set_ylabel("Impact Score", color='white', fontsize=10)
        ax2.tick_params(colors='white', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_color('#334155')
        ax2.yaxis.grid(True, color='#334155', linestyle='-', alpha=0.5)
        
        st.pyplot(fig2)

    st.write("---")
    if st.button("← Back to Form"):
        navigate_to("home")
        st.rerun()