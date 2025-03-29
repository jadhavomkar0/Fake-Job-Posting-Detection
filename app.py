import streamlit as st
import joblib
import pickle
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load("model.joblib")  # Ensure the model file is in the same folder

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)  # Load TF-IDF Vectorizer

# Streamlit App Title
st.title("🚀 Fake Job Posting Detection System")
st.markdown("Enter the job details below to classify the job posting as **Fraudulent** or **Not Fraudulent**.")

# Input Form
title = st.text_input("Job Title")
location = st.text_input("Location")
department = st.text_input("Department")
salary_range = st.text_input("Salary Range (e.g., 50000-70000)")
company_profile = st.text_area("Company Profile", height=100)
description = st.text_area("Job Description", height=150)
requirements = st.text_area("Requirements", height=100)
benefits = st.text_area("Benefits", height=100)

telecommuting = st.selectbox("Telecommuting", [0, 1])
has_company_logo = st.selectbox("Has Company Logo", [0, 1])
has_questions = st.selectbox("Has Screening Questions", [0, 1])

employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])
required_experience = st.selectbox("Required Experience", ["Not Applicable", "Entry Level", "Mid-Senior level", "Director", "Executive"])
required_education = st.selectbox("Required Education", ["Not Applicable", "High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"])

industry = st.text_input("Industry")
function = st.text_input("Function")

# Prediction Logic
if st.button("Predict"):
    # Combine input text for vectorization
    combined_text = f"{title} {location} {department} {salary_range} {company_profile} {description} {requirements} {benefits}"

    # Vectorize the combined text using loaded TF-IDF Vectorizer
    vectorized_input = vectorizer.transform([combined_text])
    vectorized_df = pd.DataFrame(vectorized_input.toarray(), columns=vectorizer.get_feature_names_out())

    # One-hot encode manually
    input_data = pd.DataFrame({
        'telecommuting': [telecommuting],
        'has_company_logo': [has_company_logo],
        'has_questions': [has_questions],
        'employment_type_Full-time': [1 if employment_type == "Full-time" else 0],
        'employment_type_Part-time': [1 if employment_type == "Part-time" else 0],
        'employment_type_Contract': [1 if employment_type == "Contract" else 0],
        'employment_type_Temporary': [1 if employment_type == "Temporary" else 0],
        'required_experience_Entry Level': [1 if required_experience == "Entry Level" else 0],
        'required_experience_Mid-Senior level': [1 if required_experience == "Mid-Senior level" else 0],
        'required_experience_Director': [1 if required_experience == "Director" else 0],
        'required_experience_Executive': [1 if required_experience == "Executive" else 0],
        'required_education_High School': [1 if required_education == "High School" else 0],
        'required_education_Associate\'s Degree': [1 if required_education == "Associate's Degree" else 0],
        'required_education_Bachelor\'s Degree': [1 if required_education == "Bachelor's Degree" else 0],
        'required_education_Master\'s Degree': [1 if required_education == "Master's Degree" else 0],
        'required_education_Doctorate': [1 if required_education == "Doctorate" else 0]
    })

    # Combine vectorized and one-hot encoded data
    complete_input = pd.concat([vectorized_df, input_data], axis=1)

    # Ensure column alignment with training data
    missing_cols = set(model.feature_names_in_) - set(complete_input.columns)
    for col in missing_cols:
        complete_input[col] = 0  # Add missing columns with zero

    # Reorder columns to match the training set
    complete_input = complete_input[model.feature_names_in_]

    # Predict using the trained model
    prediction = model.predict(complete_input)
    result = "🟢 Not Fraudulent" if prediction[0] == 0 else "🔴 Fraudulent"

    # Display Result
    st.success(f"Prediction: {result}")
