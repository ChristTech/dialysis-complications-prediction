import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import base64
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow import keras  # Import the keras module

# Define neural network
def build_nn_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10,))) # Input shape needs to match your data
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load voting model and scaler
try:
    with keras.utils.custom_object_scope({'build_nn_model': build_nn_model}):
        voting_model = joblib.load("voting_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Please check if the files exist.")
    st.stop()

# Set background image via base64 CSS injection

with open("background.jpg", "rb") as f:
    encoded = f.read()
encoded_string = base64.b64encode(encoded).decode()
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-blend-mode: lighten;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# App Title and Description
st.title("Dialysis Complication Risk Prediction")
st.markdown(
    """
This tool predicts the risk of complications in dialysis patients based on basic health inputs. 
Predictions are generated using an ensemble model.
"""
)

# Collect Input from User
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

    # Predict using the voting model
    final_proba = voting_model.predict_proba(scaled_input)[0][1]

    st.subheader("Model Confidence Level")
    st.write(f"Ensemble Confidence: {final_proba:.2f}")

    st.subheader("Final Ensemble Prediction")
    if final_proba > 0.5:
        st.error("High Risk of Dialysis Complication ❗")
    else:
        st.success("Low Risk of Dialysis Complication ✅")

    # Risk level interpretation
    if final_proba > 0.75:
        st.markdown("**🟢 Confidence: Very High (Low Risk)**")
    elif final_proba > 0.5:
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
