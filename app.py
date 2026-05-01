import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and scaler
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="House Valuation Tool", page_icon="🏠")

st.title("🏠 Real Estate Price Predictor")
st.write("Enter property details to estimate market value based on our high-accuracy Linear Regression model.")

# User Inputs based on your dataset features
col1, col2 = st.columns(2)

with col1:
    sq_ft = st.number_input("Square Footage", value=2000)
    beds = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5])
    baths = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5])
    year = st.number_input("Year Built", value=2000)

with col2:
    lot = st.number_input("Lot Size (Acres)", value=1.0)
    garage = st.selectbox("Garage Size (Cars)", options=[0, 1, 2])
    quality = st.slider("Neighborhood Quality", 1, 10, 5)

if st.button("Calculate Valuation"):
    # 1. Scaling: Apply the same scaler used in training
    scaled_features = scaler.transform([[sq_ft, lot, year]])
    
    # 2. Prepare Feature Array (Ordered exactly as your training X)
    # Order: Square_Footage, Num_Bedrooms, Num_Bathrooms, Year_Built, Lot_Size, Garage_Size, Neighborhood_Quality
    final_features = np.array([[
        scaled_features[0][0], # Scaled SqFt
        beds, 
        baths, 
        scaled_features[0][2], # Scaled Year
        scaled_features[0][1], # Scaled Lot
        garage, 
        quality
    ]])
    
    # 3. Predict (This returns the LOG price)
    log_prediction = model.predict(final_features)
    
    # 4. Reverse Log Transformation to get actual Dollars
    actual_price = np.exp(log_prediction)[0]
    
    st.success(f"Estimated Market Price: ${actual_price:,.2f}")
    st.info("Note: This prediction uses Log Transformation for improved accuracy.")