import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import base64  # Import the base64 module

# Load models and scaler
try:
    scaler = joblib.load("scaler.pkl")
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    nn_model = load_model("nn_model.h5")
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Please check if the files exist.")
    st.stop()

# Set background image via base64 CSS injection

with open("background.jpg", "rb") as f:
    encoded = f.read()
# Use base64.b64encode instead of deprecated "base64" encoding
encoded_string = base64.b64encode(encoded).decode()
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-blend-mode: overlay;
    }}
    </style>
""",
    unsafe_allow_html=True,
)


# Optional: Uncomment if you want to use a background image
# add_bg_from_local("background.png")

# App Title and Description
st.title("Dialysis Complication Prediction App")
st.markdown(
    """
This tool predicts the risk of complications in dialysis patients based on basic health inputs. 
Predictions are generated using an ensemble of SVM, Random Forest, and Neural Network models.
"""
)

# Collect Input from User
with st.form("input_form"):
    age = st.slider("Age", 65)
    systolic_bp = st.slider("Systolic Blood Pressure", 120)
    diastolic_bp = st.slider("Diastolic Blood Pressure", 80)
    weight_gain = st.number_input("Weight Gain (kg)", 3.5)
    sodium_level = st.slider("Sodium Level", 140)
    potassium_level = st.slider("Potassium Level", 3.5)
    bun = st.slider("BUN (Blood Urea Nitrogen)", 20)
    dialysis_duration = st.slider("Dialysis Duration (hours)", 1, 8, 4)
    comorbid_diabetes = st.radio("Comorbid Diabetes", [0, 1])
    comorbid_hypertension = st.radio("Comorbid Hypertension", [0, 1])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Input Validation
    if systolic_bp <= diastolic_bp:
        st.error("Systolic blood pressure must be greater than diastolic blood pressure.")
        st.stop()
    # Prepare and scale input
    new_data = pd.DataFrame(
        {
            "age": [age],
            "systolic_bp": [systolic_bp],
            "diastolic_bp": [diastolic_bp],
            "weight_gain": [weight_gain],
            "sodium_level": [sodium_level],
            "potassium_level": [potassium_level],
            "bun": [bun],
            "dialysis_duration": [dialysis_duration],
            "comorbid_diabetes": [comorbid_diabetes],
            "comorbid_hypertension": [comorbid_hypertension],
        }
    )

    scaled_input = scaler.transform(new_data)

    # Predict and show individual model confidences
    svm_proba = svm_model.predict_proba(scaled_input)[0][1]
    rf_proba = rf_model.predict_proba(scaled_input)[0][1]
    nn_proba = nn_model.predict(scaled_input)[0][0]

    st.subheader("Model Confidence Levels")
    st.write(f"SVM: {svm_proba:.2f}")
    st.write(f"Random Forest: {rf_proba:.2f}")
    st.write(f"Neural Network: {nn_proba:.2f}")

    # Voting
    votes = [svm_proba > 0.5, rf_proba > 0.5, nn_proba > 0.5]
    final_vote = sum(votes) >= 2

    st.subheader("Final Ensemble Prediction")
    if final_vote:
        st.error("High Risk of Dialysis Complication ❗")
    else:
        st.success("Low Risk of Dialysis Complication ✅")

    # Risk level interpretation
    avg_confidence = np.mean([svm_proba, rf_proba, nn_proba])
    if avg_confidence > 0.75:
        st.markdown("**🟢 Confidence: Very High (Low Risk)**")
    elif avg_confidence > 0.5:
        st.markdown("**🟠 Confidence: Moderate**")
    else:
        st.markdown("**🔴 Confidence: Low (High Risk)**")

# Feedback Form
st.markdown("---")
st.subheader("💬 Feedback")
with st.form("feedback_form"):
    feedback = st.text_input("Was this prediction helpful? Any suggestions?")
    submit_feedback = st.form_submit_button("Submit Feedback")
    if submit_feedback:
        st.success("Thank you for your feedback!")
