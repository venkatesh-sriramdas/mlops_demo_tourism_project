# Version: 1.0.5 - Simplified clean UI with main form layout
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os
import sys
import logging
from datetime import datetime

# Configure logging to output to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("STREAMLIT APP STARTING")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("=" * 80)

# Set page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)
logger.info("✅ Streamlit page configuration set")

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(120deg, #1E88E5 0%, #64B5F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 30px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .positive {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
    }

    .negative {
        background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
        color: white;
    }

    .section-divider {
        border-top: 2px solid #E3F2FD;
        margin: 30px 0 20px 0;
    }

    .section-title {
        color: #1E88E5;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
        logger.info("="*60)
        logger.info("STARTING MODEL ARTIFACT DOWNLOAD")
        logger.info("="*60)
        logger.info("Repository: svenkateshdotnet/tourism_project_model")

        # Download model artifacts from HuggingFace
        logger.info("Downloading model.pkl...")
        model_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="model.pkl",
            repo_type="model"
        )
        logger.info(f"✅ Model downloaded: {model_path}")

        logger.info("Downloading scaler.pkl...")
        scaler_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="scaler.pkl",
            repo_type="model"
        )
        logger.info(f"✅ Scaler downloaded: {scaler_path}")

        logger.info("Downloading label_encoders.pkl...")
        encoders_path = hf_hub_download(
            repo_id="svenkateshdotnet/tourism_project_model",
            filename="label_encoders.pkl",
            repo_type="model"
        )
        logger.info(f"✅ Encoders downloaded: {encoders_path}")

        # Load artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)

        logger.info("="*60)
        logger.info("ALL ARTIFACTS LOADED SUCCESSFULLY!")
        logger.info("="*60)
        return model, scaler, encoders
    except Exception as e:
        logger.error("="*60)
        logger.error("ERROR LOADING ARTIFACTS")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

# Header
st.markdown('<p class="main-header">✈️ Wellness Tourism Package Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Purchase Prediction</p>', unsafe_allow_html=True)

# Load model
logger.info("Loading artifacts...")
model, scaler, label_encoders = load_artifacts()

if model is None:
    st.error("⚠️ Failed to load model. Please check the logs.")
    st.stop()

logger.info("✅ Model ready for predictions")

# Main form container
with st.container():
    st.markdown("---")

    # Personal Information
    st.markdown('<p class="section-title">👤 Personal Information</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.slider("Age", 18, 100, 35)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    with col4:
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

    # Contact & Location
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📍 Contact & Location</p>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)

    with col5:
        type_of_contact = st.selectbox("Contact Type", ["Company Invited", "Self Enquiry"])
    with col6:
        city_tier = st.selectbox("City Tier", [1, 2, 3])
    with col7:
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    # Financial & Assets
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💰 Financial Information</p>', unsafe_allow_html=True)
    col8, col9 = st.columns(2)

    with col8:
        monthly_income = st.number_input("Monthly Income (₹)", 0, 1000000, 30000, 1000)
    with col9:
        own_car = st.checkbox("Own Car", value=False)

    # Travel Details
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">✈️ Travel Information</p>', unsafe_allow_html=True)
    col10, col11, col12, col13 = st.columns(4)

    with col10:
        number_of_persons_visiting = st.number_input("Group Size", 1, 10, 2)
    with col11:
        number_of_children_visiting = st.number_input("Children", 0, 5, 0)
    with col12:
        number_of_trips = st.slider("Previous Trips", 0, 20, 2)
    with col13:
        passport = st.checkbox("Has Passport", value=False)

    # Product & Sales
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎯 Product & Sales Details</p>', unsafe_allow_html=True)
    col14, col15, col16, col17 = st.columns(4)

    with col14:
        product_pitched = st.selectbox("Package", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    with col15:
        preferred_property_star = st.selectbox("Star Rating", [3.0, 4.0, 5.0])
    with col16:
        duration_of_pitch = st.slider("Pitch Duration (min)", 1.0, 120.0, 15.0, 0.5)
    with col17:
        number_of_followups = st.slider("Follow-ups", 0, 10, 3)

# Predict button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Purchase Likelihood", use_container_width=True)

if predict_button:
    logger.info("="*60)
    logger.info("PREDICTION REQUEST STARTED")
    logger.info("="*60)

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_persons_visiting],
        'NumberOfFollowups': [number_of_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [1 if passport else 0],
        'PitchSatisfactionScore': [3],
        'OwnCar': [1 if own_car else 0],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    })

    try:
        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        for col in label_encoders.keys():
            if col in input_data.columns:
                input_data[col] = label_encoders[col].transform(input_data[col])
        logger.info("✅ Encoding completed")

        # Create engineered features (must match training)
        logger.info("Creating engineered features...")
        input_data['Income_per_person'] = input_data['MonthlyIncome'] / (input_data['NumberOfPersonVisiting'] + 1)
        input_data['Trips_per_year_ratio'] = input_data['NumberOfTrips'] / (input_data['Age'] + 1)
        input_data['Children_ratio'] = input_data['NumberOfChildrenVisiting'] / (input_data['NumberOfPersonVisiting'] + 1)
        input_data['Followup_per_pitch'] = input_data['NumberOfFollowups'] / (input_data['DurationOfPitch'] + 1)
        logger.info("✅ Engineered features created")

        # Scale features
        logger.info("Scaling features...")
        input_scaled = scaler.transform(input_data)
        logger.info(f"✅ Scaling completed")

        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        logger.info(f"✅ Prediction: {prediction}, Probability: {probability}")
        logger.info("="*60)

        # Display results
        st.markdown("---")

        # Results in columns
        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box positive">✅ HIGH LIKELIHOOD TO PURCHASE</div>',
                    unsafe_allow_html=True
                )
                st.success(f"🎉 This customer has a **{probability[1]*100:.2f}%** probability of purchasing the package!")
                st.info("💡 **Recommendation:** Proceed with the offer and consider premium package options.")
            else:
                st.markdown(
                    f'<div class="prediction-box negative">❌ LOW LIKELIHOOD TO PURCHASE</div>',
                    unsafe_allow_html=True
                )
                st.warning(f"⚠️ This customer has only a **{probability[1]*100:.2f}%** probability of purchasing.")
                st.info("💡 **Recommendation:** Consider additional follow-ups or alternative package offerings.")

        with result_col2:
            st.markdown("### 📊 Confidence")
            st.metric("Purchase Probability", f"{probability[1]*100:.1f}%")
            st.metric("No Purchase", f"{probability[0]*100:.1f}%")
            st.progress(probability[1])

    except Exception as e:
        logger.error("="*60)
        logger.error("PREDICTION ERROR")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("="*60)

        st.error("❌ An error occurred during prediction.")
        st.error(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <p>Built with ❤️ using Streamlit | MLOps Project | Gradient Boosting (ROC-AUC: 0.9769)</p>
    </div>
    """,
    unsafe_allow_html=True
)
