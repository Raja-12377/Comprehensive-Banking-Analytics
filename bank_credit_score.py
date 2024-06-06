import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('bank_rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the input fields
st.header("Bank Credit Score Prediction")
st.subheader("Enter your details:")
annual_income = st.number_input("Annual Income: ")
monthly_inhand_salary = st.number_input("Monthly Inhand Salary: ")
num_bank_accounts = st.number_input("Number of Bank Accounts: ")
num_credit_cards = st.number_input("Number of Credit cards: ")
interest_rate = st.number_input("Interest rate: ")
num_loans = st.number_input("Number of Loans: ")
average_days_delayed = st.number_input("Average number of days delayed by the person: ")
num_delayed_payments = st.number_input("Number of delayed payments: ")
credit_mix = st.selectbox("Credit Mix (Bad: 0, Standard: 1, Good: 2)", ['Bad', 'Standard', 'Good'])
outstanding_debt = st.number_input("Outstanding Debt: ")
credit_history_age = st.number_input("Credit History Age: ")
monthly_balance = st.number_input("Monthly Balance: ")

# Create a numpy array from the input fields
features = np.array([[annual_income, monthly_inhand_salary, num_bank_accounts, num_credit_cards, interest_rate, num_loans, average_days_delayed, num_delayed_payments, credit_mix, outstanding_debt, credit_history_age, monthly_balance]])

# Apply label encoding to all numerical columns except credit_mix
le = LabelEncoder()
features[:, 0] = le.fit_transform(features[:, 0])
features[:, 1] = le.fit_transform(features[:, 1])
features[:, 2] = le.fit_transform(features[:, 2])
features[:, 3] = le.fit_transform(features[:, 3])
features[:, 4] = le.fit_transform(features[:, 4])
features[:, 5] = le.fit_transform(features[:, 5])
features[:, 6] = le.fit_transform(features[:, 6])
features[:, 7] = le.fit_transform(features[:, 7])
features[:, 8] = le.fit_transform(features[:, 8])
features[:, 9] = le.fit_transform(features[:, 9])
features[:, 10] = le.fit_transform(features[:, 10])
features[:, 11] = le.fit_transform(features[:, 11])

if st.button("Make Prediction"):
    # Predict the credit score
    credit_score = loaded_model.predict(features)
    # Display the predicted credit score
    st.header("Predicted Credit Score:")
    st.info(credit_score[0])