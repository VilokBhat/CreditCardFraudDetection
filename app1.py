import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load('models/RandomForest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Title for the Streamlit app
st.title("Credit Card Fraud Detection App")

# Input form for user inputs
st.write("Enter the transaction details to predict if it's fraudulent:")

# Input fields
time = st.number_input("Transaction Time", min_value=0.0, max_value=1e6)
amount = st.number_input("Transaction Amount", min_value=0.0, max_value=1e6)

# Predict button
if st.button("Predict"):
    # Prepare data for prediction
    data = pd.DataFrame({
        'Time': [time],
        'Amount': [amount]
    })
    
    # Scale the data
    data['scaled_time'] = scaler.transform(data[['Time']])
    data['scaled_amount'] = scaler.transform(data[['Amount']])
    data = data.drop(['Time', 'Amount'], axis=1)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Output result
    if prediction == 1:
        st.error("This transaction is likely fraudulent!")
    else:
        st.success("This transaction is likely not fraudulent.")

