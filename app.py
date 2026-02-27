import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

.header-box {
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    text-align:center;
    color:white;
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
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
<h1>🏥 Health Insurance Charges Predictor</h1>
<p>AI-powered medical cost estimation dashboard</p>
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

    st.divider()

    st.subheader("📊 Key Performance Indicators")

    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    risk_score = (
        age * 0.3 +
        bmi * 0.4 +
        (1 if smoker == "yes" else 0) * 30
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👤 Age", age)
    col2.metric("⚖ BMI Category", bmi_status)
    col3.metric("🚬 Smoker", smoker.upper())
    col4.metric("⚠ Risk Score", f"{risk_score:.1f}")

    st.subheader("📈 Analytics Dashboard")

    colA, colB = st.columns(2)

    # -------- BMI Analysis Chart --------
    with colA:
        st.write("### BMI Health Range")

        categories = ["Underweight", "Normal", "Overweight", "Obese"]
        ranges = [18.5, 25, 30, 35]

        fig1 = plt.figure()
        plt.bar(categories, ranges,color='cyan')
        plt.axhline(y=bmi, linestyle="--")
        plt.title("BMI Category Comparison")
        st.pyplot(fig1)

    with colB:
        st.write("### Estimated Cost Drivers")

        drivers = ["Age Impact", "BMI Impact", "Children Impact"]
        values = [age * 0.2, bmi * 0.3, children * 5]

        fig2 = plt.figure()
        plt.bar(drivers, values,color='27F5A6')
        plt.title("Factors Influencing Cost")
        st.pyplot(fig2)

    st.subheader("🎯 Health Risk Indicator")

    risk_percent = min(int(risk_score), 100)
    st.progress(risk_percent)

    if risk_percent < 30:
        st.success("Low Risk Profile ✅")
    elif risk_percent < 60:
        st.warning("Moderate Risk Profile ⚠")
    else:
        st.error("High Risk Profile 🚨")


st.markdown("""
<hr>
<center>Built with ❤️ using Streamlit & Machine Learning</center>
""", unsafe_allow_html=True)