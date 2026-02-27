import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

/* ---------- Animated Background ---------- */
body {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color:white;
}

@keyframes gradientBG {
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

/* ---------- Header ---------- */
.header-box{
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding:35px;
    border-radius:16px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    box-shadow:0 10px 25px rgba(0,0,0,0.3);
}

/* ---------- Glass Card ---------- */
.card{
    background:rgba(255,255,255,0.06);
    backdrop-filter:blur(14px);
    padding:30px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,0.15);
    margin-bottom:20px;
}

/* ---------- Inputs ---------- */
label{
    font-weight:600 !important;
    color:#e6f0ff !important;
}

/* Selectbox */
div[data-baseweb="select"] > div{
    background:rgba(255,255,255,0.08)!important;
    border-radius:14px!important;
    border:1px solid rgba(255,255,255,0.2)!important;
    transition:0.3s;
}

div[data-baseweb="select"] > div:hover{
    border:1px solid #00eaff!important;
    box-shadow:0 0 10px rgba(0,234,255,0.5);
}

div[data-baseweb="select"] span{
    color:white!important;
}

/* Number Input */
.stNumberInput input{
    background:rgba(255,255,255,0.08)!important;
    border-radius:14px!important;
    border:1px solid rgba(255,255,255,0.2)!important;
    color:white!important;
    padding:10px!important;
}

.stNumberInput input:focus{
    border:1px solid #00f2fe!important;
    box-shadow:0 0 12px rgba(0,242,254,0.6);
}

/* Button */
.stButton>button{
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    border-radius:12px;
    height:45px;
    font-size:16px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.05);
}

/* Result Box */
.result-box{
    background:linear-gradient(90deg,#43e97b,#38f9d7);
    padding:30px;
    border-radius:18px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    animation:pop 0.6s ease;
}

@keyframes pop{
0%{transform:scale(0.8);opacity:0;}
100%{transform:scale(1);opacity:1;}
}

/* Sticker Animation */
.sticker{
    text-align:center;
    font-size:75px;
    animation:bounce 2s infinite;
}

@keyframes bounce{
0%,100%{transform:translateY(0);}
50%{transform:translateY(-12px);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
<h1>🏥 AI Health Insurance Charges Predictor</h1>
<p>Smart Medical Cost Estimation • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

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
    💰 Estimated Insurance Cost<br><br>
    ₹ {prediction:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # -------- BMI CATEGORY --------
    if bmi < 18.5:
        bmi_status="Underweight"
        sticker="🥗"
    elif bmi < 25:
        bmi_status="Normal"
        sticker="💪"
    elif bmi < 30:
        bmi_status="Overweight"
        sticker="⚖"
    else:
        bmi_status="Obese"
        sticker="🚨"

    risk_score = age*0.3 + bmi*0.4 + (1 if smoker=="yes" else 0)*30

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("👤 Age", age)
    col2.metric("⚖ BMI Status", bmi_status)
    col3.metric("🚬 Smoker", smoker.upper())
    col4.metric("⚠ Risk Score", f"{risk_score:.1f}")

    # Sticker Display
    st.markdown(f"""
        <div class="sticker">{sticker}</div>
        <h3 style='text-align:center'>{bmi_status} Health Category</h3>
    """, unsafe_allow_html=True)

    # -------- CHARTS --------
    st.subheader("📊 Health Analytics")

    colA,colB = st.columns(2)

    with colA:
        fig1 = plt.figure()
        plt.bar(["Underweight","Normal","Overweight","Obese"],
                [18.5,25,30,35])
        plt.axhline(y=bmi, linestyle="--")
        plt.title("BMI Comparison")
        st.pyplot(fig1)

    with colB:
        fig2 = plt.figure()
        plt.bar(["Age","BMI","Children"],
                [age*0.2,bmi*0.3,children*5])
        plt.title("Cost Drivers")
        st.pyplot(fig2)

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