import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import kagglehub
import os

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Cache the model loading/training function
@st.cache_data
def load_and_train_model():
    """Load dataset and train the model"""
    # Download latest version
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    
    # Load the dataset
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if csv_file:
        df = pd.read_csv(csv_file)
        
        # Handle categorical variables
        categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                           'airconditioning', 'prefarea', 'furnishingstatus']
        
        df_processed = df.copy()
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le
        
        # Prepare features and target
        feature_columns = [col for col in df_processed.columns if col != 'price']
        X = df_processed[feature_columns].values
        y = df_processed['price'].values
        
        # Train model
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X, y)
        
        return regressor, feature_columns, label_encoders, df
    else:
        st.error("CSV file not found!")
        return None, None, None, None

# Load model and data
with st.spinner("Loading model and data..."):
    model, feature_columns, encoders, original_df = load_and_train_model()

if model is not None:
    # Title and description
    st.title("🏠 House Price Prediction App")
    st.markdown("### Enter house details to get a price prediction")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 House Specifications")
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=2)
        parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1)
    
    with col2:
        st.subheader("🏘️ Amenities & Features")
        mainroad = st.selectbox("Main Road Access", ["Yes", "No"], index=0)
        guestroom = st.selectbox("Guest Room", ["Yes", "No"], index=1)
        basement = st.selectbox("Basement", ["Yes", "No"], index=1)
        hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"], index=1)
        airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"], index=0)
        prefarea = st.selectbox("Preferred Area", ["Yes", "No"], index=0)
        furnishingstatus = st.selectbox("Furnishing Status", 
                                       ["Furnished", "Semi-Furnished", "Unfurnished"], 
                                       index=0)
    
    # Convert categorical inputs to encoded values
    def encode_categorical_input(value, column):
        if column in encoders:
            # Create a temporary array with the input value
            try:
                encoded = encoders[column].transform([value])[0]
                return encoded
            except:
                # If value not seen during training, use mode
                return 0
        return value
    
    # Convert Yes/No to 1/0
    mainroad_encoded = 1 if mainroad == "Yes" else 0
    guestroom_encoded = 1 if guestroom == "Yes" else 0
    basement_encoded = 1 if basement == "Yes" else 0
    hotwaterheating_encoded = 1 if hotwaterheating == "Yes" else 0
    airconditioning_encoded = 1 if airconditioning == "Yes" else 0
    prefarea_encoded = 1 if prefarea == "Yes" else 0
    
    # Convert furnishing status
    furnishing_map = {"Furnished": 1, "Semi-Furnished": 2, "Unfurnished": 0}
    furnishingstatus_encoded = furnishing_map[furnishingstatus]
    
    # Create prediction button
    if st.button("🔮 Predict House Price", type="primary"):
        # Prepare input data
        input_data = [
            area, bedrooms, bathrooms, stories, mainroad_encoded,
            guestroom_encoded, basement_encoded, hotwaterheating_encoded,
            airconditioning_encoded, parking, prefarea_encoded, furnishingstatus_encoded
        ]
        
        # Make prediction
        prediction = model.predict([input_data])
        predicted_price = prediction[0]
        
        # Display results
        st.success(f"### 💰 Predicted House Price: ${predicted_price:,.2f}")
    
    # Feature importance section
    with st.expander("📊 Feature Importance"):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
        
        st.dataframe(importance_df, use_container_width=True)
    
else:
    st.error("Failed to load the model. Please check your internet connection and try again.")

st.markdown("---")
st.markdown("### 📝 How to use:")
st.markdown("""
1. **Enter house specifications** in the left column (area, bedrooms, etc.)
2. **Select amenities and features** in the right column
3. **Click 'Predict House Price'** to get your estimate
4. **Explore feature importance** to understand what affects prices most
5. **Check sample predictions** to compare different house types
""")
