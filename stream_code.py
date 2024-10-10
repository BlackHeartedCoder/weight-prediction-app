import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.set_page_config(page_title="Weight Prediction App", page_icon="⚖️", layout="centered")

# Add a header and a subheader
st.title("⚖️ Weight Prediction App")
st.subheader("Predict your weight based on input features")

st.write(
    """
    Welcome to the Weight Prediction App! Enter your feature value below, and 
    the machine learning model will predict your weight. This app uses a Random Forest Regressor model 
    trained on relevant data. Feel free to experiment with different inputs!
    """
)

# Input fields with explanations
st.sidebar.header("Input Features")
feature_input = st.sidebar.number_input(
    "Enter your feature value (e.g., height or other relevant input):", value=0.0, step=0.1
)

# Add an informative help text for users
st.sidebar.write(
    """
    ### Instructions:
    1. Enter your feature value on the slider.
    2. Click **Predict** to get your predicted weight.
    3. The result will appear below.
    """
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
    st.write("The prediction is based on a Random Forest model that has been trained on relevant data features. Please note that this prediction is an estimate.")

# Add a footer
st.write("---")
st.write("Made with 🖤 by Dharmik")
