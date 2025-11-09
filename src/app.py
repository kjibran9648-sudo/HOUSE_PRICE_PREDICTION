import os
import sys

# --- Robust base directory detection (works when _file_ is missing) ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # _file_ not defined (happens in some Streamlit execution modes)
    if len(sys.argv) > 0 and sys.argv[0]:
        BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        BASE_DIR = os.getcwd()

# model path (one level up from src)
model_path = os.path.join(BASE_DIR, '..', 'models', 'model.joblib')
model_path = os.path.abspath(model_path)  # normalize
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'model.joblib')
model = joblib.load(model_path)

st.title("🏠 House Price Prediction App")

st.write("Enter the details below to estimate the house price:")

# User inputs
area = st.number_input("Area (in sq. ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{prediction:,.2f}")