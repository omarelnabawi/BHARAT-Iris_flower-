import streamlit as st
import joblib
import numpy as np

st.title("Iris Classification")

# Load the trained model and label encoder
model = joblib.load('svc_model.pkl')
le = joblib.load('label_encoder.pkl')

# Input fields for user to enter feature values
SepalLengthCm = st.number_input("Enter Sepal Length (in cm):")
SepalWidthCm = st.number_input("Enter Sepal Width (in cm):")
PetalLengthCm = st.number_input("Enter Petal Length (in cm):")
PetalWidthCm = st.number_input("Enter Petal Width (in cm):")

# Predict button
if st.button("Predict"):
    # Predict the species based on user input
    input_features = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    pred = model.predict(input_features)
    predicted_species = le.inverse_transform(pred)

    # Display the result with a larger font size
    st.markdown(f"<h2 style='font-size: 28px; color: blue;'>Predicted Iris species: {predicted_species[0]}</h2>", unsafe_allow_html=True)
