import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Define the custom CSS style as a string for later use
custom_css = """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0f4c81;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0f4c81;
    }
    .risk-very-low {
        color: #1E8449;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #58D68D;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-moderate {
        color: #F4D03F;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-high {
        color: #E67E22;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-very-high {
        color: #C0392B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """

def load_models(file_path='thyroid_cancer_models.pkl'):
    """Load saved models from a file"""
    try:
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data
    except FileNotFoundError:
        st.error(f"Error: Model file {file_path} not found. Please ensure the model file is in the correct location.")
        return None

def predict_thyroid_cancer_risk(data, model_type, saved_data):
    """
    Predict thyroid cancer risk for new patient data.
    """
    # Extract components from saved data
    trained_models = saved_data['models']
    scaler = saved_data['scaler']
    label_encoders = saved_data['label_encoders']
    feature_columns = saved_data['feature_columns']
    target_encoder = saved_data['target_encoder']
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Create DataFrame with same columns and order as training data
    X = data[feature_columns].copy()
    
    # Apply label encoding to categorical features
    categorical_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                       'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                       'Obesity', 'Diabetes']
    
    for col in categorical_cols:
        if col in X.columns:
            le = label_encoders[col]
            # Handle new categories
            for category in X[col].unique():
                if category not in le.classes_:
                    X.loc[X[col] == category, col] = le.classes_[0]
            X[col] = le.transform(X[col])
    
    # Scale numerical features
    numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    num_cols_to_scale = [col for col in numerical_cols if col in X.columns]
    if num_cols_to_scale:
        X[num_cols_to_scale] = scaler.transform(X[num_cols_to_scale])
    
    # Make predictions based on model_type
    model = trained_models[model_type]
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    
    # Define risk levels based on probability
    def get_risk_level(prob):
        if prob < 0.2:
            return "Very Low"
        elif prob < 0.4:
            return "Low"
        elif prob < 0.6:
            return "Moderate"
        elif prob < 0.8:
            return "High"
        else:
            return "Very High"
    
    # Prepare results
    results = {
        'predicted_class': prediction[0],
        'predicted_prob': float(probability[0]),
        'risk_level': get_risk_level(probability[0]),
        'diagnosis': target_encoder.inverse_transform([prediction[0]])[0]
    }
    
    return results

def create_gauge_chart(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#1E8449'},
                {'range': [20, 40], 'color': '#58D68D'},
                {'range': [40, 60], 'color': '#F4D03F'},
                {'range': [60, 80], 'color': '#E67E22'},
                {'range': [80, 100], 'color': '#C0392B'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def create_feature_importance_chart(model_type, saved_data):
    """Create feature importance visualization if available"""
    model = saved_data['models'][model_type]
    feature_names = saved_data['feature_columns']
    
    # Check if model has feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # For ensemble models that might have estimators
        if hasattr(model, 'estimators_') and model_type == 'voting':
            # For voting classifier, use the average of feature importances from tree-based models
            tree_models = [est for name, est in model.named_estimators_ 
                          if hasattr(est, 'feature_importances_')]
            if tree_models:
                importances = np.mean([est.feature_importances_ for est in tree_models], axis=0)
            else:
                return None
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        return fig
    
    return None

def main():
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>Thyroid Cancer Risk Prediction Tool</h1>", unsafe_allow_html=True)
    
    st.write("""
    This application predicts the risk of thyroid cancer based on patient characteristics and clinical measurements.
    Enter the patient information below to get a risk assessment.
    """)
    
    # Load models
    saved_data = load_models()
    if saved_data is None:
        st.stop()
    
    # Create sidebar for inputs
    st.sidebar.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
    
    # Demographic information
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 90, 45)
    country = st.sidebar.selectbox("Country", ["USA", "Canada", "UK", "Australia", "Japan", "China", "India", "Other"])
    ethnicity = st.sidebar.selectbox("Ethnicity", ["Caucasian", "African", "Asian", "Hispanic", "Other"])
    
    # Risk factors
    st.sidebar.markdown("<h3>Risk Factors</h3>", unsafe_allow_html=True)
    family_history = st.sidebar.selectbox("Family History of Thyroid Cancer", ["No", "Yes"])
    radiation_exposure = st.sidebar.selectbox("History of Radiation Exposure", ["No", "Yes"])
    iodine_deficiency = st.sidebar.selectbox("Iodine Deficiency", ["No", "Yes"])
    smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
    obesity = st.sidebar.selectbox("Obesity", ["No", "Yes"])
    diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    
    # Clinical measurements
    st.sidebar.markdown("<h3>Clinical Measurements</h3>", unsafe_allow_html=True)
    tsh_level = st.sidebar.slider("TSH Level (mIU/L)", 0.1, 10.0, 2.5, 0.1)
    t3_level = st.sidebar.slider("T3 Level (nmol/L)", 0.5, 3.0, 1.2, 0.1)
    t4_level = st.sidebar.slider("T4 Level (µg/dL)", 4.0, 12.0, 8.5, 0.1)
    nodule_size = st.sidebar.slider("Nodule Size (cm)", 0.1, 5.0, 0.8, 0.1)
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Prediction Model", 
        ["voting", "RandomForest", "LogisticRegression", "DecisionTree"],
        index=0
    )
    
    # Create patient data dictionary
    patient_data = {
        'Gender': gender,
        'Age': age,
        'Country': country,
        'Ethnicity': ethnicity,
        'Family_History': family_history,
        'Radiation_Exposure': radiation_exposure,
        'Iodine_Deficiency': iodine_deficiency,
        'Smoking': smoking,
        'Obesity': obesity,
        'Diabetes': diabetes,
        'TSH_Level': tsh_level,
        'T3_Level': t3_level,
        'T4_Level': t4_level,
        'Nodule_Size': nodule_size
    }
    
    # Predict button
    if st.sidebar.button("Predict Risk"):
        # Make prediction
        with st.spinner("Calculating risk..."):
            result = predict_thyroid_cancer_risk(patient_data, model_type, saved_data)
        
        # Display results
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
            
            # Display diagnosis
            st.markdown(f"**Diagnosis Prediction:** {result['diagnosis']}")
            
            # Display risk level with appropriate color
            risk_level = result['risk_level']
            risk_class = f"risk-{risk_level.lower().replace(' ', '-')}"
            st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Display probability
            st.markdown(f"**Probability:** {result['predicted_prob']:.2%}")
            
            # Display gauge chart
            st.plotly_chart(create_gauge_chart(result['predicted_prob']))
            
            # Display feature importance if available
            importance_chart = create_feature_importance_chart(model_type, saved_data)
            if importance_chart:
                st.plotly_chart(importance_chart)
        
        with col2:
            st.markdown("<h2 class='sub-header'>Risk Factors Analysis</h2>", unsafe_allow_html=True)
            
            # Create a DataFrame for the patient's risk factors
            risk_factors = pd.DataFrame({
                'Factor': [
                    'Age', 'Gender', 'Family History', 'Radiation Exposure',
                    'Iodine Deficiency', 'Smoking', 'Obesity', 'Diabetes',
                    'TSH Level', 'Nodule Size'
                ],
                'Value': [
                    f"{age} years",
                    gender,
                    family_history,
                    radiation_exposure,
                    iodine_deficiency,
                    smoking,
                    obesity,
                    diabetes,
                    f"{tsh_level:.1f} mIU/L",
                    f"{nodule_size:.1f} cm"
                ],
                'Risk Impact': [
                    'Higher risk with increasing age',
                    'Higher risk in females',
                    'Significant risk if positive',
                    'Major risk factor if exposed',
                    'Moderate risk factor if deficient',
                    'Minor risk factor if positive',
                    'Minor risk factor if present',
                    'Minor risk factor if present',
                    'Abnormal levels increase risk',
                    'Larger nodules have higher risk'
                ]
            })
            
            # Display the risk factors table
            st.dataframe(risk_factors, hide_index=True, use_container_width=True)
            
            # Add recommendations based on risk level
            st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
            
            if result['risk_level'] in ["Very Low", "Low"]:
                st.write("""
                - Regular check-ups as recommended by your physician
                - Monitor for any changes in symptoms
                - Maintain a healthy lifestyle
                """)
            elif result['risk_level'] == "Moderate":
                st.write("""
                - Consider additional diagnostic tests
                - More frequent monitoring may be needed
                - Consult with an endocrinologist
                - Address modifiable risk factors
                """)
            else:  # High or Very High
                st.write("""
                - Immediate consultation with a specialist is recommended
                - Additional diagnostic tests (ultrasound, biopsy) are likely needed
                - Close monitoring and follow-up
                - Comprehensive treatment plan should be developed
                """)
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            **Disclaimer:** This tool provides an estimate based on machine learning models and should not replace 
            professional medical advice. Always consult with healthcare providers for diagnosis and treatment decisions.
            """)
    
    else:
        # Display information about the app when no prediction is made
        st.markdown("<h2 class='sub-header'>About This Tool</h2>", unsafe_allow_html=True)
        st.write("""
        This application uses machine learning to predict thyroid cancer risk based on patient characteristics and clinical measurements.
        
        **Available prediction models:**
        - **Voting Ensemble**: Combines multiple models for more robust predictions
        - **Random Forest**: Good for capturing complex patterns in data
        - **Logistic Regression**: Simple but effective for many medical predictions
        - **Decision Tree**: Provides easily interpretable rules
        
        **How to use:**
        1. Enter patient information in the sidebar
        2. Select your preferred prediction model
        3. Click "Predict Risk" to get results
        
        **Interpretation:**
        - The tool provides a risk level from Very Low to Very High
        - A probability percentage indicates the likelihood of thyroid cancer
        - Feature importance shows which factors most influenced the prediction
        """)

if __name__ == "__main__":
    main()
