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

/* -------- Animated Background -------- */
body {
    background: linear-gradient(-45deg,#e0f7fa,#f1f8ff,#e8f5e9,#fce4ec);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

/* -------- Glass Cards -------- */
.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    margin-bottom:20px;
}

/* Header */
.header-box{
    background: linear-gradient(90deg,#007cf0,#00dfd8);
    padding:35px;
    border-radius:15px;
    color:white;
    text-align:center;
    animation: fadeIn 1.5s ease-in;
}

/* Result Card */
.result-box{
    background: linear-gradient(90deg,#43e97b,#38f9d7);
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    animation: pop 0.6s ease;
}

@keyframes pop{
0%{transform:scale(0.8);opacity:0;}
100%{transform:scale(1);opacity:1;}
}

/* Inputs Styling */
.stNumberInput input, .stSelectbox div {
    border-radius:10px !important;
    border:1px solid #d0e3ff !important;
}

/* Slider */
.stSlider > div > div {
    color:#007cf0 !important;
}

/* Button */
.stButton>button {
    background:linear-gradient(90deg,#007cf0,#00dfd8);
    color:white;
    border-radius:10px;
    height:45px;
    font-size:16px;
    font-weight:bold;
}

.stButton>button:hover{
    transform:scale(1.05);
}

/* Sticker */
.sticker{
    text-align:center;
    font-size:70px;
    animation:bounce 2s infinite;
}

@keyframes bounce{
0%,100%{transform:translateY(0);}
50%{transform:translateY(-10px);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
<h1>🏥 AI Health Insurance Charges Predictor</h1>
<p>Smart Medical Cost Estimation • Risk Intelligence • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📝 Patient Information")

with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", 18, 100, 30)
        bmi = st.slider("⚖ BMI", 10.0, 60.0, 25.0)
        children = st.number_input("👶 Children", 0, 8, 0)

    with col2:
        gender = st.selectbox("⚧ Gender", le_gender.classes_)
        smoker = st.selectbox("🚬 Smoker", le_smoker.classes_)

    submitted = st.form_submit_button("🔮 Predict Insurance Charges")

st.markdown("</div>", unsafe_allow_html=True)

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

    num_cols = ["age","bmi","children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    input_data = input_data.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]

    # RESULT BOX
    st.markdown(f"""
        <div class="result-box">
        💰 Estimated Insurance Cost<br><br>
        ₹ {prediction:,.2f}
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- BMI CATEGORY ----------------
    if bmi < 18.5:
        bmi_status = "Underweight"
        sticker = "🥗"
    elif bmi < 25:
        bmi_status = "Normal"
        sticker = "💪"
    elif bmi < 30:
        bmi_status = "Overweight"
        sticker = "⚖"
    else:
        bmi_status = "Obese"
        sticker = "🚨"

    risk_score = age*0.3 + bmi*0.4 + (1 if smoker=="yes" else 0)*30

    # ---------------- KPI ----------------
    col1,col2,col3,col4 = st.columns(4)

    col1.metric("👤 Age", age)
    col2.metric("⚖ BMI Status", bmi_status)
    col3.metric("🚬 Smoker", smoker.upper())
    col4.metric("⚠ Risk Score", f"{risk_score:.1f}")

    # ---------------- STICKER ----------------
    st.markdown(f"""
        <div class="sticker">{sticker}</div>
        <h3 style='text-align:center'>{bmi_status} Health Category</h3>
    """, unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    st.subheader("📊 Health Analytics")

    colA,colB = st.columns(2)

    with colA:
        categories=["Underweight","Normal","Overweight","Obese"]
        ranges=[18.5,25,30,35]

        fig1=plt.figure()
        plt.bar(categories,ranges)
        plt.axhline(y=bmi,linestyle="--")
        plt.title("BMI Comparison")
        st.pyplot(fig1)

    with colB:
        drivers=["Age","BMI","Children"]
        values=[age*0.2,bmi*0.3,children*5]

        fig2=plt.figure()
        plt.bar(drivers,values)
        plt.title("Cost Drivers")
        st.pyplot(fig2)

    # ---------------- RISK BAR ----------------
    st.subheader("🎯 Health Risk Indicator")

    risk_percent=min(int(risk_score),100)
    st.progress(risk_percent)

    if risk_percent<30:
        st.success("Low Risk Profile ✅")
    elif risk_percent<60:
        st.warning("Moderate Risk Profile ⚠")
    else:
        st.error("High Risk Profile 🚨")

# FOOTER
st.markdown("""
<hr>
<center>🚀 Built with Streamlit • Machine Learning • Medical AI</center>
""", unsafe_allow_html=True)