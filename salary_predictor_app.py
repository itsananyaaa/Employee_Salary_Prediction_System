import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="salary Predictor", layout="centered")

st.title("Employee Salary Prediction System")
st.markdown("Enter job details to predict salary (in USD)")

experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
job_titles = ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'AI Researcher', 'Software Engineer', 'ML Ops Engineer', 'Data Engineer', 'Research Scientist']
job_title = st.selectbox("Job Title", job_titles)
company_size = st.selectbox("Company Size", ["S", "M", "SL"])
employee_residence = st.text_input("Employee Residence (enter your country code): ")
remote_ratio = st.slider("Remote Ratio in %", 0, 100, 0)
company_location = st.text_input("Company Location (enter country code)")

if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title],
        "company_size": [company_size],
        "employee_residence": [employee_residence],
        "remote_ratio": [remote_ratio]
    })

    input_data_encoded = pd.get_dummies(input_data)

    model_cols = pickle.load(open("model_columns.pkl", "rb"))
    input_data_encoded = input_data_encoded.reindex(columns = model_cols, fill_value=0)

    prediction = model.predict(input_data_encoded)
    st.success(f"Predicted Salary: ${int(prediction[0])}")