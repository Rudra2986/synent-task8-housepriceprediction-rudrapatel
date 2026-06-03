"""
Streamlit Web Application - House Price Prediction Dashboard
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

An interactive interface to predict California housing prices and view
model evaluation statistics.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration for a professional layout
st.set_page_config(
    page_title="California Housing Valuation Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Load Model & Preprocessor
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "house_price_model.joblib")
    prep_path = os.path.join(current_dir, "models", "preprocessor.joblib")
    
    # Try importing prediction components
    from src.prediction_pipeline import load_champion_model, load_preprocessor, run_inference
    
    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = load_champion_model(model_path)
        preprocessor = load_preprocessor(prep_path)
        return model, preprocessor, run_inference
    return None, None, None


model, preprocessor, run_inference = load_assets()

# Header Section
st.title("California Housing Price Predictor")
st.markdown(
    """
    This dashboard provides pricing estimates and performance analyses for housing blocks in California. 
    It uses historical census data (including geographic location, income levels, and housing age) to predict 
    the median house value of a block group.
    """
)

st.markdown("---")

if model is None:
    st.error(
        "⚠️ The model files could not be found. Please run the training pipeline (`python main.py`) "
        "first to train and save the regression models."
    )
else:
    # Navigation Tabs at the top
    tab1, tab2 = st.tabs(["Interactive Prediction Tool", "Model Evaluation & Metrics"])
    
    with tab1:
        st.subheader("Interactive Valuation Form")
        st.markdown(
            "Adjust the parameters below to calculate the estimated median house price for a specific block group."
        )
        
        # Simple two-column layout
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("### 📍 Location Parameters")
            longitude = st.slider("Longitude", -124.35, -114.31, -118.24, step=0.01)
            latitude = st.slider("Latitude", 32.54, 41.95, 34.05, step=0.01)
            
            # Simple Map
            map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
            st.map(map_data, zoom=7, size=35)
            
        with col2:
            st.markdown("### 🏡 Structural & Income Parameters")
            
            # Sub-columns for clean grouping
            sub_col1, sub_col2 = st.columns(2)
            
            with sub_col1:
                median_age = st.slider("Median House Age (Years)", 1.0, 52.0, 28.0)
                total_rooms = st.number_input("Total Rooms in Block", min_value=1.0, max_value=40000.0, value=2000.0, step=50.0)
                total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1.0, max_value=7000.0, value=400.0, step=10.0)
                
            with sub_col2:
                population = st.number_input("Block Group Population", min_value=3.0, max_value=36000.0, value=1000.0, step=50.0)
                households = st.number_input("Number of Households", min_value=1.0, max_value=6000.0, value=350.0, step=10.0)
                median_income = st.slider("Median Household Income (x $10,000)", 0.5, 15.0, 3.5, step=0.1)
                
            ocean_proximity = st.selectbox(
                "Ocean Proximity Category", 
                ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
            )
            
            st.markdown(" ")
            
            # Predict Button
            if total_bedrooms > total_rooms:
                st.warning("⚠️ Total bedrooms cannot exceed total rooms. Please check the structural parameters.")
                predict_disabled = True
            else:
                predict_disabled = False
                
            if st.button("Predict Median House Value", type="primary", disabled=predict_disabled):
                # Construct raw input DataFrame
                raw_input = pd.DataFrame([{
                    "longitude": longitude,
                    "latitude": latitude,
                    "housing_median_age": median_age,
                    "total_rooms": total_rooms,
                    "total_bedrooms": total_bedrooms,
                    "population": population,
                    "households": households,
                    "median_income": median_income,
                    "ocean_proximity": ocean_proximity
                }])
                
                try:
                    preds = run_inference(model, preprocessor, raw_input)
                    predicted_price = preds[0]
                    
                    st.success("Calculated Valuation Successfully!")
                    st.metric(label="Predicted Median House Value", value=f"${predicted_price:,.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    
    with tab2:
        st.subheader("Model Validation Results")
        st.markdown(
            "We trained and evaluated three regression algorithms on an 80% training split, "
            "scoring them on the remaining 20% validation split. Below are the actual performance metrics."
        )
        
        # Load comparison CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comp_csv = os.path.join(current_dir, "data", "processed", "model_comparison.csv")
        
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)
            st.dataframe(
                comp_df.style.highlight_max(subset=["R2"], color="#E2E8F0").highlight_min(subset=["RMSE", "MAE"], color="#E2E8F0"),
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.info("Metrics comparison table is not available. Please run the training pipeline to generate it.")
            
        st.markdown("---")
        st.subheader("Performance Charts")
        
        # Display charts
        col_img1, col_img2 = st.columns(2)
        
        rmse_plot = os.path.join(current_dir, "images", "model_comparison_rmse.png")
        r2_plot = os.path.join(current_dir, "images", "model_comparison_r2.png")
        residuals_plot = os.path.join(current_dir, "images", "residuals_plot.png")
        importances_plot = os.path.join(current_dir, "images", "feature_importances.png")
        
        with col_img1:
            if os.path.exists(rmse_plot):
                st.image(rmse_plot, caption="Figure 1: Root Mean Squared Error (lower is better)", use_container_width=True)
            if os.path.exists(residuals_plot):
                st.image(residuals_plot, caption="Figure 3: Predicted vs. Actual Values (Random Forest)", use_container_width=True)
                
        with col_img2:
            if os.path.exists(r2_plot):
                st.image(r2_plot, caption="Figure 2: R-squared Score (higher is better)", use_container_width=True)
            if os.path.exists(importances_plot):
                st.image(importances_plot, caption="Figure 4: Feature Importance Weights (Random Forest)", use_container_width=True)

st.markdown("---")
st.markdown(
    """
    **Project Metadata:**
    * **Developer:** Rudra Patel (Intern ID: `SYN/J2/IP806`)
    * **Email:** [rudrapatel2156@gmail.com](mailto:rudrapatel2156@gmail.com)
    * **GitHub:** [Rudra2986](https://github.com/Rudra2986)
    """
)
