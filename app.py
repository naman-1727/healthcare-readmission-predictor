import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler and features
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# App title
st.title("🏥 Hospital Readmission Predictor")
st.write("Enter patient details below to predict the risk of readmission within 30 days.")

# Input fields
st.header("Patient Information")

age = st.selectbox("Age Group", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
])

time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
num_medications = st.slider("Number of Medications", 1, 81, 15)
num_procedures = st.slider("Number of Procedures", 0, 6, 1)
num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
number_inpatient = st.slider("Inpatient Visits (past year)", 0, 21, 0)
number_emergency = st.slider("Emergency Visits (past year)", 0, 76, 0)

insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
change = st.selectbox("Medication Change", ["No", "Ch"])
diabetesMed = st.selectbox("On Diabetes Medication", ["Yes", "No"])

# Predict button
if st.button("Predict Readmission Risk"):

    # Build input row with all zeros
    input_dict = {col: 0 for col in feature_names}

    # Fill numeric values
    input_dict['time_in_hospital'] = time_in_hospital
    input_dict['num_medications'] = num_medications
    input_dict['num_procedures'] = num_procedures
    input_dict['num_lab_procedures'] = num_lab_procedures
    input_dict['number_diagnoses'] = number_diagnoses
    input_dict['number_inpatient'] = number_inpatient
    input_dict['number_emergency'] = number_emergency

    # Fill encoded categorical values
    age_col = f'age_{age}'
    if age_col in input_dict:
        input_dict[age_col] = 1

    insulin_col = f'insulin_{insulin}'
    if insulin_col in input_dict:
        input_dict[insulin_col] = 1

    change_col = f'change_{change}'
    if change_col in input_dict:
        input_dict[change_col] = 1

    diabetesMed_col = f'diabetesMed_{diabetesMed}'
    if diabetesMed_col in input_dict:
        input_dict[diabetesMed_col] = 1

    # Convert to dataframe and scale
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This patient has a {probability*100:.1f}% probability of readmission within 30 days")
        st.write("**Recommendation:** Consider enhanced discharge planning and follow-up care.")
    else:
        st.success(f"✅ LOW RISK — This patient has a {probability*100:.1f}% probability of readmission within 30 days")
        st.write("**Recommendation:** Standard discharge procedure appropriate.")