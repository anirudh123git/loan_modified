import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved Random Forest model
model_path = 'random_forest_model.pkl'
with open(model_path, 'rb') as file:
    random_forest_model = pickle.load(file)

# Label Encoders for categorical features
le_education = LabelEncoder()
le_education.classes_ = np.array(['Graduate', 'Not Graduate'])

le_self_employed = LabelEncoder()
le_self_employed.classes_ = np.array(['No', 'Yes'])
# Streamlit App Title
st.title("🏦 Loan Approval Prediction System")

# Add an image in the sidebar
st.sidebar.image(r"C:\Users\vansh\OneDrive\Desktop\loan_modified\loan_image.jpg", caption="Loan Prediction System", use_container_width=True)  # Change filename as needed

st.sidebar.markdown("### Enter details to check loan approval status.")

# Label Encoder for loan_status (to decode predictions)
loan_status_encoder = LabelEncoder()
loan_status_encoder.classes_ = np.array(['Rejected', 'Approved'])

# Streamlit App Title
st.title("Loan Approval Prediction System")

# User Inputs
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
income_annum = st.number_input("Annual Income (in INR)", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount (in INR)", min_value=0, value=200000)
loan_term = st.number_input("Loan Term (in days)", min_value=0, value=360)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=0, value=300000)
commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=0, value=100000)
luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=0, value=50000)
bank_asset_value = st.number_input("Bank Asset Value (in INR)", min_value=0, value=250000)

# Encode the categorical inputs
education_encoded = le_education.transform([education])[0]
self_employed_encoded = le_self_employed.transform([self_employed])[0]

# Prepare input array for prediction
input_features = np.array([
    no_of_dependents,
    education_encoded,
    self_employed_encoded,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]).reshape(1, -1)

# Predict button
if st.button("Predict Loan Status"):
    prediction = random_forest_model.predict(input_features)
    predicted_status = loan_status_encoder.inverse_transform(prediction)
    st.success(f"Loan Status: {predicted_status[0]}")
