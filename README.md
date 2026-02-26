🏥 Health Insurance Charges Predictor (AI/ML Project)

📌 Project Overview

The Health Insurance Charges Predictor is a Machine Learning web application that estimates medical insurance costs based on user details such as age, BMI, smoking habits, and number of children.

The application uses a trained regression model and provides instant predictions through an interactive Streamlit web interface.

This project demonstrates the practical implementation of:

Machine Learning model training

Data preprocessing

Model deployment

Interactive UI development using Streamlit

🚀 Live Features

✅ Predict insurance charges instantly
✅ Clean and modern Streamlit UI
✅ Real-time user input processing
✅ Scaled and encoded ML inputs
✅ Fast prediction using trained model

🧠 Machine Learning Workflow

Data Collection (insurance.csv)

Data Preprocessing

Label Encoding (Gender, Smoker)

Feature Scaling

Model Training & Evaluation

Model Saving using joblib

Deployment using Streamlit Cloud

🛠️ Tech Stack
Category	Technology
Language	Python
ML Library	Scikit-learn
Data Handling	Pandas, NumPy
Model Saving	Joblib
Web Framework	Streamlit
Deployment	Streamlit Cloud
📂 Project Structure
health-insurance-charges-predictor-ai/
│
├── app.py
├── insurance.csv
├── best_model.pkl
├── scaler.pkl
├── label_encoder_sex.pkl
├── label_encoder_smoker.pkl
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/health-insurance-charges-predictor-ai.git
cd health-insurance-charges-predictor-ai
2️⃣ Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
streamlit run app.py
📊 Input Features

Age

Gender

BMI

Number of Children

Smoking Status

💰 Output

The system predicts:

👉 Estimated Health Insurance Charges (₹)

👨‍💻 Team Members

Subham Sahu

Priyabrat Jena

Mohit Samal

Sujeet Deo

Sandesh Ojha

🎯 Project Goal

To build a real-world AI application that helps users estimate medical insurance costs quickly using predictive analytics.

❤️ Built With

Streamlit

Machine Learning

Python

📜 License

This project is developed for educational and academic purposes.

