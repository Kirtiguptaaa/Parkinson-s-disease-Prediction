import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("parkinson_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Parkinson's Disease Predictor")

# Input fields
features = []

for i in range(22):  # number of features
    val = st.number_input(f"Feature {i+1}")
    features.append(val)

if st.button("Predict"):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"Parkinson's Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"Healthy (Confidence: {1-prob:.2f})")