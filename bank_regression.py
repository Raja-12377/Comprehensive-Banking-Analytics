import streamlit as st
import pandas as pd
import pickle

st.header("Bank Performance Prediction")

# Load the saved SVR model
with open('bank_svr_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the feature columns
feature_columns = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                   'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                   'Credit_Utilization_Ratio', 'Credit_History_Age', 'Monthly_Balance']

# Create a new DataFrame to store the user input
new_data = pd.DataFrame(columns=feature_columns)

# Ask the user for input
for column in feature_columns:
    new_data[column] = [st.number_input(f"{column}: ")]

# Make predictions using the SVR best model
prediction = loaded_model.predict(new_data)

# Display the multi-target output
if st.button("Make prediction"):
    st.header("Predicted Output:")
    st.write("Asset Growth: ", prediction[0][0])
    st.write("Revenue: ", prediction[0][1])
    st.write("Profitability: ", prediction[0][2])