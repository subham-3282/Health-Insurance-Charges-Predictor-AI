import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_sex.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
}

.header-box {
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

.result-box {
    background: linear-gradient(90deg,#43e97b,#38f9d7);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: black;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div class="title">🏥 Health Insurance Charges Predictor</div>
    <div class="subtitle">
        Estimate medical insurance cost instantly using Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📝 Enter Your Details")

with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Number of Children", 0, 8, 0)

    with col2:
        gender = st.selectbox("Gender", options=le_gender.classes_)
        smoker = st.selectbox("Smoker", options=le_smoker.classes_)

    submitted = st.form_submit_button("🔮 Predict Insurance Charges")

st.markdown('</div>', unsafe_allow_html=True)

if submitted:

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [gender],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker]
    })

    input_data["sex"] = le_gender.transform(input_data["sex"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])

    num_cols = ["age", "bmi", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    expected_cols = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_cols)

    prediction = model.predict(input_data)[0]

    st.markdown(f"""
        <div class="result-box">
            💰 Estimated Insurance Cost<br><br>
            Rs.{prediction:,.2f}
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr>
<center>Built with ❤️ using Streamlit & Machine Learning</center>
""", unsafe_allow_html=True)