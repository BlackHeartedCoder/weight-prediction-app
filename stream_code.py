import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.set_page_config(page_title="Weight Prediction App", page_icon="⚖️", layout="centered")

# Add a header and a subheader
st.title("⚖️ Weight Prediction App")
st.subheader("Predict your weight based on Height")

st.write(
    """
    Welcome to the Weight Prediction App! Enter your Height, and 
    the machine learning model will predict your weight.
    """
)

# Input fields with explanations
st.sidebar.header("Input Features")
feature_input = st.sidebar.number_input(
    "Enter your Height (Inches) :", value=0.0, step=0.1
)


# Display a "Predict" button
if st.sidebar.button('Predict'):
    # Prepare input as the model expects it
    features = np.array([[feature_input]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Show prediction result with some nice formatting
    st.success(f'🎯 The predicted weight is **{prediction:.2f} kg**')

    # Add some more context or insights
    st.write("The prediction is based on a **Random Forest** model that has been trained on relevant data features.")

# Add a footer
st.write("---")
st.write("Made by Dharmik")
