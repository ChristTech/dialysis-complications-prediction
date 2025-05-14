import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import datetime
from PIL import Image
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Load models
svm_model = None
rf_model = None
scaler = None
model_load_error = None

try:
    # Load individual models
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    # nn_model = load_model("nn_model.keras")  # Load the Keras model
    scaler = joblib.load("my_scaler.pkl")  # Corrected scaler filename
except FileNotFoundError as e:
    model_load_error = f"FileNotFoundError: {e}"
    st.warning("Model or scaler file not found. Check file paths.")
except AttributeError as e:
    model_load_error = f"AttributeError: {e}"
    st.warning("AttributeError during model loading. Check versions and custom objects.")
except Exception as e:
    model_load_error = f"Unexpected error: {e}"
    st.warning(f"An unexpected error occurred during model loading: {e}")

# Set background image
try:
    with open("background.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Background image not found. Check the file path.")
except Exception as e:
    st.warning(f"Error loading background image: {e}")

# Title
st.title("🩺 Dialysis Complication Risk Prediction")
st.markdown("""
This app predicts the risk of complications in dialysis patients using an ensemble of machine learning models.
""")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 18, 90, 65)
    systolic_bp = st.slider("Systolic Blood Pressure", 50, 200, 150)
    diastolic_bp = st.slider("Diastolic Blood Pressure", 40, 130, 70)
    weight_gain = st.number_input("Weight Gain (kg)", 0.0, 10.0, 3.5)
    sodium_level = st.slider("Sodium Level", 120, 160, 140)
    potassium_level = st.slider("Potassium Level", 2.0, 6.0, 3.5)
    bun = st.slider("BUN (Blood Urea Nitrogen)", 5, 60, 20)
    dialysis_duration = st.slider("Dialysis Duration (hours)", 1, 8, 4)
    comorbid_diabetes = st.radio("Comorbid Diabetes", [0, 1])
    comorbid_hypertension = st.radio("Comorbid Hypertension", [0, 1])
    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    if systolic_bp <= diastolic_bp:
        st.error("❌ Systolic blood pressure must be greater than diastolic blood pressure.")

    new_data = pd.DataFrame({
        'age': [age],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'weight_gain': [weight_gain],
        'sodium_level': [sodium_level],
        'potassium_level': [potassium_level],
        'bun': [bun],
        'dialysis_duration': [dialysis_duration],
        'comorbid_diabetes': [comorbid_diabetes],
        'comorbid_hypertension': [comorbid_hypertension]
    })

    final_proba = None
    if scaler is not None:
        scaled_input = scaler.transform(new_data)
    else:
        st.error("Scaler was not loaded. Please check the file path and version.")
        scaled_input = None

    if svm_model is not None and rf_model is not None and scaled_input is not None:
        # Get individual model predictions
        svm_proba = svm_model.predict_proba(scaled_input)[0][1]
        rf_proba = rf_model.predict_proba(scaled_input)[0][1]
        # nn_proba = nn_model.predict(scaled_input)[0][0]  # Assuming sigmoid output

        # Average the predictions (soft voting)
        final_proba = np.mean([svm_proba, rf_proba])

        st.markdown("### 🧪 Model Confidence Level")
        st.metric(label="Predicted Risk (%)", value=f"{final_proba * 100:.1f}")
        st.progress(int(final_proba * 100))

        # Risk Message
        st.subheader("📌 Risk Prediction")
        if final_proba > 0.5:
            st.error("High Risk of Dialysis Complication ❗")
        else:
            st.success("Low Risk of Dialysis Complication ✅")

        # Confidence Interpretation
        if final_proba > 0.75:
            st.markdown("**🟢 Confidence: Very High (High Risk)**")
        elif final_proba > 0.5:
            st.markdown("**🟠 Confidence: Moderate**")
        else:
            st.markdown("**🔴 Confidence: Low (Low Risk)**")
    else:
        st.error("One or more models were not loaded. Please check the file paths and versions.")


    # Timestamp
    st.caption(f"🕒 Prediction made on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # About Model
    with st.expander("📊 About the Model"):
        st.markdown("""
        This prediction is generated using a **voting ensemble model** consisting of:
        - 🎯 Support Vector Machine (SVM)
        - 🌲 Random Forest
        - 🧠 Neural Network (Keras)

        Each model contributes to the final decision through soft voting (probability averaging).
        """)

    # Downloadable Report
    if final_proba is not None:
        risk_status = "High Risk ❗" if final_proba > 0.5 else "Low Risk ✅"
        risk_score_text = f"{final_proba:.2f}"
    else:
        risk_status = "N/A"
        risk_score_text = "N/A"

    report = f"""
Dialysis Complication Risk Report
---------------------------------
Age: {age}
Systolic BP: {systolic_bp}
Diastolic BP: {diastolic_bp}
Weight Gain: {weight_gain}
Sodium Level: {sodium_level}
Potassium Level: {potassium_level}
BUN: {bun}
Dialysis Duration: {dialysis_duration}
Diabetes: {comorbid_diabetes}
Hypertension: {comorbid_hypertension}
Risk Score: {risk_score_text}
Prediction: {risk_status}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    st.download_button(
        label="📄 Download Prediction Report",
        data=report,
        file_name="dialysis_risk_report.txt"
    )

# Feedback section
st.markdown("---")
st.subheader("💬 Feedback")
with st.form("feedback_form"):
    feedback = st.text_input("Was this prediction helpful? Any suggestions?")
    submit_feedback = st.form_submit_button("Submit Feedback")
    if submit_feedback:
        st.success("✅ Thank you for your feedback!")

if model_load_error:
    st.error(f"The app may not function correctly due to the following error: {model_load_error}")
