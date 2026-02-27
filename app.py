import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- LOAD MODELS ----------------
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_sex.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ---------- Animated Medical Background ---------- */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f2027, #203a43);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp::before {
    content: "✚  🏥  🩺  💊  ✚  🏥  🩺  💊";
    position: fixed;
    top: -10%; left: -10%; width: 120vw; height: 120vh;
    font-size: 8rem;
    color: rgba(255, 255, 255, 0.02);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-around;
    align-items: center;
    line-height: 2.5;
    z-index: 0;
    pointer-events: none;
    animation: floatingText 40s linear infinite;
    letter-spacing: 50px;
}

@keyframes floatingText {
    0% { transform: translateY(0) rotate(0deg); opacity: 0.5; }
    50% { transform: translateY(-100px) rotate(5deg); opacity: 1; }
    100% { transform: translateY(0) rotate(0deg); opacity: 0.5; }
}

/* ---------- Premium Glass Header ---------- */
.header-box {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 35px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
}

.header-box::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    animation: shine 6s infinite;
}

@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}

.header-box h1 {
    background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: textGradient 3s linear infinite;
    font-weight: 800;
}

@keyframes textGradient {
    to { background-position: 200% center; }
}

/* ---------- Glass Card ---------- */
.card {
    background: rgba(20, 30, 45, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 35px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
    position: relative;
    z-index: 1;
    overflow: hidden;
}

/* ---------- Healthcare Team Walking Animation ---------- */
.medic-parade {
    position: absolute;
    bottom: 10px;
    left: 0; right: 0;
    width: 100%;
    height: 50px;
    pointer-events: none;
    z-index: 2;
    overflow: visible;
}

.medic-char {
    position: absolute;
    bottom: 0;
    font-size: 36px;
    animation: walkAcross var(--walk-speed, 14s) linear infinite;
    animation-delay: var(--walk-delay, 0s);
    left: -60px;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));
}

.medic-char span {
    display: inline-block;
    animation: stepping 0.4s ease-in-out infinite alternate;
}

@keyframes walkAcross {
    0%   { left: -60px; }
    100% { left: 110%; }
}

@keyframes stepping {
    0%   { transform: translateY(0) rotate(-2deg); }
    100% { transform: translateY(-5px) rotate(2deg); }
}

/* ---------- Premium Inputs ---------- */
label {
    font-weight: 600 !important;
    letter-spacing: 0.5px;
}

/* Input Fields and Selectboxes */
div[data-baseweb="select"] > div,
.stNumberInput input {
    border-radius: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: #1e293b !important;
}

div[data-baseweb="select"] span {
    color: #1e293b !important;
}

div[data-baseweb="select"] > div:hover,
.stNumberInput input:hover {
    border: 1px solid rgba(0, 242, 254, 0.5) !important;
}

div[data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus {
    border: 1px solid #00f2fe !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.4), inset 0 2px 4px rgba(0,0,0,0.1) !important;
    transform: translateY(-2px);
}

/* Slider Customization */
div[data-baseweb="slider"] {
    padding-top: 10px;
}

div[data-baseweb="slider"] div[role="slider"] {
    background-color: #00f2fe !important;
    border: 3px solid #1e293b !important;
    box-shadow: 0 0 15px #00f2fe !important;
    width: 24px !important;
    height: 24px !important;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-baseweb="slider"] div[role="slider"]:hover {
    transform: scale(1.2);
}

div[data-baseweb="slider"] div > div {
    height: 8px !important;
    border-radius: 4px !important;
}

/* ---------- Submit Button ---------- */
.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 16px;
    height: 55px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 1px;
    border: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.4);
    width: 100%;
    margin-top: 10px;
    position: relative;
    overflow: hidden;
}

.stButton>button::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, #0072ff, #00c6ff);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 114, 255, 0.6);
}

.stButton>button:active {
    transform: translateY(2px);
}

.stButton>button p {
    position: relative;
    z-index: 1;
}

/* ---------- Result Box ---------- */
.result-box {
    background: linear-gradient(135deg, rgba(67, 233, 123, 0.2), rgba(56, 249, 215, 0.2));
    backdrop-filter: blur(20px);
    padding: 40px;
    border-radius: 24px;
    border: 1px solid rgba(67, 233, 123, 0.4);
    text-align: center;
    font-size: 36px;
    font-weight: 900;
    color: #fff;
    box-shadow: 0 15px 35px rgba(67, 233, 123, 0.2);
    animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.result-box span {
    color: #43e97b;
    font-size: 48px;
    display: block;
    margin-top: 10px;
    text-shadow: 0 0 20px rgba(67, 233, 123, 0.5);
}

@keyframes slideUp {
    0% { transform: translateY(50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

/* ---------- Domain Stickers ---------- */
.sticker-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 20px 0;
    gap: 20px;
}

.sticker-box {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 25px 40px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    animation: floatSticker 3s ease-in-out infinite;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.sticker-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
}

.sticker-box:hover::before {
    opacity: 1;
}

@keyframes floatSticker {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-15px) rotate(2deg); }
}

.sticker-icon {
    font-size: 80px;
    line-height: 1;
    margin-bottom: 10px;
    display: inline-block;
    filter: drop-shadow(0 10px 15px rgba(0,0,0,0.4));
}

.sticker-label {
    font-size: 24px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Sticker Types */
.sticker-underweight { border-color: rgba(56, 189, 248, 0.5); box-shadow: 0 0 30px rgba(56, 189, 248, 0.2); }
.sticker-underweight .sticker-label { color: #38bdf8; }

.sticker-normal { border-color: rgba(74, 222, 128, 0.5); box-shadow: 0 0 30px rgba(74, 222, 128, 0.2); }
.sticker-normal .sticker-label { color: #4ade80; }

.sticker-overweight { border-color: rgba(250, 204, 21, 0.5); box-shadow: 0 0 30px rgba(250, 204, 21, 0.2); }
.sticker-overweight .sticker-label { color: #facc15; }

.sticker-obese { border-color: rgba(248, 113, 113, 0.5); box-shadow: 0 0 30px rgba(248, 113, 113, 0.2); }
.sticker-obese .sticker-label { color: #f87171; }
.sticker-obese { animation: alertPulse 2s infinite !important; }

@keyframes alertPulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.4); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(248, 113, 113, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
}

/* Metrics Styling */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #fff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: transform 0.3s ease, background 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#00f2fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 12px; filter: drop-shadow(0 0 10px #00f2fe);">
            <path d="M18 20V10"/><path d="M6 20V10"/><path d="M2 20h20"/>
            <path d="M4 10h16"/><path d="M12 4v6"/><path d="M9 7h6"/>
            <path d="M2 10l10-6 10 6"/>
        </svg>
        AI Health Insurance Charges Predictor
    </h1>
    <p style="font-size: 18px; opacity: 0.9; margin-top: 10px; font-weight: 500; letter-spacing: 1px;">🩺 Smart Medical Cost Estimation • Predictive Analytics 💊</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('''
<div class="card">
    <div class="medic-parade">
        <div class="medic-char" style="--walk-speed:15s; --walk-delay:0s;"><span>👨‍⚕️</span></div>
        <div class="medic-char" style="--walk-speed:15s; --walk-delay:3s;"><span>👩‍⚕️</span></div>
        <div class="medic-char" style="--walk-speed:15s; --walk-delay:6s;"><span>🧑‍🔬</span></div>
        <div class="medic-char" style="--walk-speed:15s; --walk-delay:9s;"><span>🚑</span></div>
        <div class="medic-char" style="--walk-speed:15s; --walk-delay:12s;"><span>💊</span></div>
    </div>
''', unsafe_allow_html=True)

st.subheader("📝 Patient Information")

with st.form("input_form"):

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        age = st.number_input("👤 Age", 18, 100, 30)
        bmi = st.slider("⚖ BMI", 10.0, 60.0, 25.0)
        children = st.number_input("👶 Number of Children", 0, 8, 0)

    with col2:
        gender = st.selectbox("🧑 Gender", le_gender.classes_)
        smoker = st.selectbox("🚬 Smoking Status", le_smoker.classes_)

    submitted = st.form_submit_button("🔮 Predict Insurance Charges")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame({
        "age":[age],
        "sex":[gender],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker]
    })

    input_data["sex"] = le_gender.transform(input_data["sex"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])

    input_data[["age","bmi","children"]] = scaler.transform(
        input_data[["age","bmi","children"]]
    )

    input_data = input_data.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-box">
        Estimated Insurance Cost
        <span>₹ {prediction:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # -------- BMI CATEGORY STICKER --------
    st.subheader("🏥 Patient Health Profile")

    if bmi < 18.5:
        bmi_status = "Underweight"
        sticker_class = "sticker-underweight"
        sticker_icon = "�"
    elif bmi < 25:
        bmi_status = "Normal"
        sticker_class = "sticker-normal"
        sticker_icon = "🩺"
    elif bmi < 30:
        bmi_status = "Overweight"
        sticker_class = "sticker-overweight"
        sticker_icon = "🍔"
    else:
        bmi_status = "Obese"
        sticker_class = "sticker-obese"
        sticker_icon = "🚨"

    # Sticker Display
    st.markdown(f"""
        <div class="sticker-container">
            <div class="sticker-box {sticker_class}">
                <div class="sticker-icon">{sticker_icon}</div>
                <div class="sticker-label">{bmi_status}</div>
                <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">BMI: {bmi:.1f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    risk_score = age*0.3 + bmi*0.4 + (1 if smoker=="yes" else 0)*30

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("👤 Age", f"{age} Yrs")
    col2.metric("⚖ BMI Class", bmi_status)
    col3.metric("🚬 Smoker", "Yes" if smoker == "yes" else "No")
    col4.metric("⚠ Risk Score", f"{risk_score:.1f}/100")

    # -------- CHARTS --------
    st.subheader("📊 Health Analytics")

    colA, colB = st.columns(2)

    with colA:
        categories = ["Underweight", "Normal", "Overweight", "Obese"]
        thresholds = [18.5, 24.9, 29.9, 40]
        colors = ["#38bdf8", "#4ade80", "#facc15", "#f87171"]
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=categories, y=thresholds,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}" for v in thresholds],
            textposition="outside",
            textfont=dict(color="white", size=13, family="Inter"),
            hovertemplate="%{x}: BMI < %{y}<extra></extra>"
        ))
        fig1.add_hline(y=bmi, line_dash="dot", line_color="#00f2fe", line_width=3,
                       annotation_text=f"Your BMI: {bmi:.1f}",
                       annotation_font_color="#00f2fe",
                       annotation_font_size=14)
        fig1.update_layout(
            title=dict(text="🏃 BMI Category Comparison", font=dict(size=18, color="white")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(showgrid=False, color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94a3b8", title="BMI Value"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=380,
            bargap=0.35
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        drivers = ["Age Factor", "BMI Factor", "Children Factor"]
        values = [age * 0.2, bmi * 0.3, children * 5]
        colors2 = ["#818cf8", "#f472b6", "#34d399"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=drivers, y=values,
            marker=dict(
                color=colors2,
                line=dict(width=0)
            ),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=13, family="Inter"),
            hovertemplate="%{x}: %{y:.1f}<extra></extra>"
        ))
        fig2.update_layout(
            title=dict(text="💰 Insurance Cost Drivers", font=dict(size=18, color="white")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(showgrid=False, color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#94a3b8", title="Impact Score"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=380,
            bargap=0.35
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -------- RISK BAR --------
    st.subheader("🎯 Health Risk Indicator")

    risk_percent=min(int(risk_score),100)
    st.progress(risk_percent)

    if risk_percent < 30:
        st.success("Low Risk Profile ✅")
    elif risk_percent < 60:
        st.warning("Moderate Risk Profile ⚠")
    else:
        st.error("High Risk Profile 🚨")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>🚀 Built with Streamlit • Machine Learning • Medical AI</center>
""", unsafe_allow_html=True)