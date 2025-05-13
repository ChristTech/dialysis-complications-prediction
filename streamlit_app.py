import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import datetime
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define the neural network architecture
def build_nn_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load models
try:
    # Use custom_object_scope to handle the Keras model
    with keras.utils.custom_object_scope({'build_nn_model': build_nn_model, 'KerasClassifier': KerasClassifier}):
        voting_model = joblib.load("voting_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()
except AttributeError as e:
    st.error(f"AttributeError during model loading: {e}.  Check versions and custom objects.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Set background image
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
        st.stop()

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

    scaled_input = scaler.transform(new_data)

    # Load the SavedModel as a Keras layer
    saved_model_path = "path/to/your/saved_model"  # Replace with the actual path
    tfsm_layer = TFSMLayer(saved_model_path, call_endpoint='serving_default')

    # Create a Keras input
    input_tensor = tf.keras.Input(shape=(10,))  # Adjust shape to match your input data

    # Connect the input to the TFSMLayer
    output_tensor = tfsm_layer(input_tensor)

    # Create a Keras model
    keras_model = Model(inputs=input_tensor, outputs=output_tensor)

    # Make predictions
    final_proba = keras_model.predict(scaled_input)[0][0]

    st.markdown("### 🧪 Model Confidence Level")
    st.metric(label="Predicted Risk (%)", value=f"{final_proba * 100:.1f}")

    # Gauge-style chart using progress bar
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
    Risk Score: {final_proba:.2f}
    Prediction: {"High Risk ❗" if final_proba > 0.5 else "Low Risk ✅"}
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
