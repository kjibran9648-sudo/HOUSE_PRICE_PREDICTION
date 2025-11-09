import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('../models/model.joblib')

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