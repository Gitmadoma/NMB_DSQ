#main program file to run streamlit app from.
import streamlit as st

import numpy as np
import pandas as pd
import joblib

model = joblib.load()
st.title("Will they complain?? :scream: ")

# Creating a form for customer details
with st.form("customer_details_form"):
    st.write("Please enter the customer details:")

    # Text input for customer ID and other string-type information
    customer_id = st.text_input('Customer ID')
    customer_name = st.text_input('Customer Name')

    # Numeric inputs for age, balance, etc.
    customer_age = st.number_input('Customer Age', min_value=0, max_value=100)
    account_balance = st.number_input('Account Balance')

    # Date input for date-related information
    last_transaction_date = st.date_input('Last Transaction Date')

    # Selectbox for categorical data
    account_type = st.selectbox('Account Type', ['Savings', 'Checking', 'Loan', 'Credit Card'])

    # Radio buttons for binary choices
    enrolled_in_rewards_program = st.radio('Enrolled in Rewards Program', ['Yes', 'No'])

    # Checkboxes for multi-select options
    services_used = st.multiselect('Services Used', ['Online Banking', 'Mobile Banking', 'ATM', 'Branch Visit'])

    # Slider for range values
    transaction_amount_last_30_days = st.slider('Transaction Amount in Last 30 Days', 0, 10000, 500)

    # Submit button for the form
    submit_button = st.form_submit_button("Submit")

# def predict():
#     prediction = model.predict(preprocessed_data)
#     if prediction == 0:
#         st.success("Customer will not complain")
#     else:
#         st.error("Customer will complain")
        
        
# Processing the input data when the form is submitted
if submit_button:
    st.write("Customer ID:", customer_id)
    st.write("Customer Name:", customer_name)
    st.write("Age:", customer_age)
    st.write("Account Balance:", account_balance)
    st.write("Last Transaction Date:", last_transaction_date)
    st.write("Account Type:", account_type)
    st.write("Enrolled in Rewards Program:", enrolled_in_rewards_program)
    st.write("Services Used:", services_used)
    st.write("Transaction Amount in Last 30 Days:", transaction_amount_last_30_days)

    # Here you can add code to preprocess the input data and then make predictions using your model
    
    # prediction = model.predict(preprocessed_data)
    # st.write("Prediction:", prediction)
