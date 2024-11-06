import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load the trained pipeline (which includes the ColumnTransformer and the model)
def load_model():
    with open("model1.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# Load the dataset to extract unique locations
def load_data():
    data = pd.read_csv('Cleaned_data.csv')  # Use your dataset path
    locations = data['location'].unique()
    return data, locations


data, locations = load_data()


# Prediction function
def predict_price(location, bhk, bath, total_sqft):
    # Create a DataFrame with the input values (ensure the column names match the training data)
    input_df = pd.DataFrame([[location, bhk, bath, total_sqft]],
                            columns=['location', 'bhk', 'bath', 'total_sqft'])  # Use 'total_sqft' here

    # Pass the raw input directly to the pipeline (model)
    prediction = model.predict(input_df)

    return round(prediction[0], 3)


# Streamlit app layout
st.title("Welcome to Bangalore House Price Predictor")

st.write("Want to predict the price of a new House in Bangalore? Try filling the details below:")

# Input fields for the user to provide house details
location = st.selectbox('Select the Location', locations)
bhk = st.number_input('Enter BHK:', min_value=1, max_value=10, step=1)
bath = st.number_input('Enter Number of Bathrooms:', min_value=1, max_value=10, step=1)
total_sqft = st.number_input('Enter Total Square Feet:', min_value=300, max_value=10000, step=100)  # Use 'total_sqft'

# Button to trigger the prediction
if st.button("Predict Price"):
    prediction = predict_price(location, bhk, bath, total_sqft)
    st.success(f"Prediction: ₹{prediction}")
