# inference.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from loan_oop_model import LoanModel
# Load model
@st.cache_resource
def load_model():
    with open('xgboost_full_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Loan Approbval Prediction")
st.header("Input Data Nasabah:")

person_age = st.number_input("Age", min_value=20, max_value=150, value=30)
person_income = st.number_input("Income", min_value=8000, max_value=1000000, value=80000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=35000, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.42, max_value=20.0, value=10.0)
previous_loan_defaults_on_file = st.selectbox("Previous Default on File", ["Yes", "No"])  
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=5)
loan_percent_income = st.number_input("Loan Amount as Percentage of Income", min_value=0.0, max_value=66.0, value=15.0)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=2, max_value=30, value=5)
credit_score = st.number_input("Credit Score", min_value=390, max_value=850, value=700)


if st.button("Prediksi Persetujuan Pinjaman"):
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_emp_exp': [person_emp_exp]
    })

    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Prediksi: NASABAH AMAN (Loan Diterima)")
    else:
        st.error("Prediksi: NASABAH BERPOTENSI DEFAULT (Loan Ditolak)")


