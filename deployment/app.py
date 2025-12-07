import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# Set page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .positive {
        background-color: #4CAF50;
        color: white;
    }
    .negative {
        background-color: #f44336;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
        # Download model artifacts from HuggingFace
        model_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="model.pkl",
            repo_type="model"
        )
        scaler_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="scaler.pkl",
            repo_type="model"
        )
        encoders_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="label_encoders.pkl",
            repo_type="model"
        )

        # Load artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)

        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Title and description
st.markdown('<p class="main-header">✈️ Wellness Tourism Package Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer purchase likelihood for Wellness Tourism Package</p>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model, scaler, label_encoders = load_artifacts()

if model is None:
    st.error("Failed to load model. Please check your HuggingFace credentials.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create input form
st.markdown("### 📋 Customer Information")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Details**")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    st.markdown("**💼 Professional Details**")
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=500000, value=50000, step=1000)

with col3:
    st.markdown("**🏙️ Location & Assets**")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns Car?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**👨‍👩‍👧‍👦 Travel Details**")
    number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
    number_of_children_visiting = st.number_input("Number of Children (<5 years)", min_value=0, max_value=5, value=0)
    number_of_trips = st.number_input("Average Trips Per Year", min_value=0, max_value=20, value=2)

with col5:
    st.markdown("**🏨 Preferences**")
    preferred_property_star = st.selectbox("Preferred Hotel Star Rating", [3, 4, 5])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col6:
    st.markdown("**📞 Interaction Details**")
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Predict button
st.markdown("---")
if st.button("🔮 Predict Purchase Likelihood", use_container_width=True):
    try:
        # Create dataframe with input data
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_person_visiting],
            'NumberOfFollowups': [number_of_followups],
            'ProductPitched': [product_pitched],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income]
        })

        # Feature engineering (same as training)
        input_data['Income_per_person'] = input_data['MonthlyIncome'] / (input_data['NumberOfPersonVisiting'] + 1)
        input_data['Trips_per_year_ratio'] = input_data['NumberOfTrips'] / (input_data['Age'] + 1)
        input_data['Children_ratio'] = input_data['NumberOfChildrenVisiting'] / (input_data['NumberOfPersonVisiting'] + 1)
        input_data['Followup_per_pitch'] = input_data['NumberOfFollowups'] / (input_data['DurationOfPitch'] + 1)

        # Encode categorical variables
        categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except:
                    # If category not seen during training, use first category
                    input_data[col] = 0

        # Get numerical columns (should match training)
        numerical_features = input_data.select_dtypes(include=[np.number]).columns.tolist()

        # Scale numerical features
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-box positive">✅ WILL PURCHASE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-box negative">❌ WON\'T PURCHASE</div>',
                    unsafe_allow_html=True
                )

        with col_result2:
            st.markdown("#### 📊 Prediction Confidence")
            purchase_prob = prediction_proba[1] * 100
            not_purchase_prob = prediction_proba[0] * 100

            st.metric("Purchase Probability", f"{purchase_prob:.2f}%")
            st.metric("Won't Purchase Probability", f"{not_purchase_prob:.2f}%")

            # Progress bar for visualization
            st.progress(purchase_prob / 100)

        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 Recommendation")
        if prediction == 1:
            if purchase_prob > 80:
                st.success("🎯 **High Priority Lead**: This customer shows strong interest. Recommend immediate follow-up with personalized package details.")
            else:
                st.info("📞 **Potential Customer**: Good prospect. Schedule a consultation call to address any concerns.")
        else:
            if not_purchase_prob > 80:
                st.warning("⏸️ **Low Priority**: Customer shows minimal interest. Consider follow-up after some time with different offerings.")
            else:
                st.info("🔄 **Nurture Lead**: Borderline case. Provide more information and follow up with special offers.")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug info:")
        st.write(f"Input data shape: {input_data.shape}")
        st.write(f"Columns: {input_data.columns.tolist()}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Powered by Gradient Boosting Model | Accuracy: 95.28% | ROC-AUC: 0.9769</p>
        <p>Built with Streamlit & HuggingFace 🤗</p>
    </div>
    """,
    unsafe_allow_html=True
)
