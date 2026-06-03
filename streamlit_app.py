"""
Streamlit Web Application - House Price Prediction Dashboard
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

An interactive, professional interface to predict California housing prices, 
explore spatial coordinates, and view regression model performance metrics.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page Configuration for a professional layout
st.set_page_config(
    page_title="California Housing Valuation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS for layout structure
st.markdown("""
<style>
    .prediction-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1A365D;
    }
    .footer-bar {
        background-color: #F8FAFC;
        border-top: 1px solid #E2E8F0;
        padding: 1.5rem;
        margin-top: 3rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


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


# Load a sample of raw data for interactive map visualization
@st.cache_data
def get_map_sample():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "raw", "housing.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Sample 1200 points for smooth browser rendering
            return df.sample(n=1200, random_state=42)
        except Exception:
            return None
    return None


model, preprocessor, run_inference = load_assets()
sample_df = get_map_sample()

# Dashboard Title
st.title("California Housing Valuation & Analysis")
st.markdown(
    "An analytical tool to estimate median house values across California. "
    "This system uses historical census block group details—such as geographical position, "
    "income levels, and local housing density—to estimate market valuation."
)

st.markdown("---")

if model is None:
    st.error(
        "⚠️ The model files could not be found. Please run the training pipeline (`python main.py`) "
        "first to train and save the regression models."
    )
else:
    # Sidebar Metadata Information
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.markdown(
        "The models are trained on the **California Housing** dataset containing **20,640 records** "
        "across California block groups. Features describe room counts, household counts, and demographics."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Details")
    st.sidebar.markdown(
        "- **Primary Model:** Random Forest Regressor\n"
        "- **Test Split:** 20% (4,128 records)\n"
        "- **Target Metric:** Median House Value"
    )
    
    # Navigation Tabs
    tab1, tab2 = st.tabs(["Interactive Predictor", "Model Performance & Analytics"])
    
    with tab1:
        # KPI metrics section at the top of predictor
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.metric(label="Dataset Total Blocks", value="20,640")
        with kpi_col2:
            st.metric(label="Average House Value", value="$206,855")
        with kpi_col3:
            st.metric(label="Key Pricing Predictor", value="Median Income")
            
        st.markdown("---")
        
        # Grid layout for predictor inputs
        col1, col2 = st.columns([1.1, 0.9])
        
        with col1:
            st.markdown("### 📍 Location Coordinates")
            st.markdown("Use the sliders to select coordinates. The selected location is highlighted as a red star on the map.")
            
            longitude = st.slider("Longitude (Degrees West)", -124.35, -114.31, -118.24, step=0.01)
            latitude = st.slider("Latitude (Degrees North)", 32.54, 41.95, 34.05, step=0.01)
            
            # Interactive Plotly Map
            if sample_df is not None:
                fig = px.scatter(
                    sample_df,
                    x="longitude",
                    y="latitude",
                    color="median_house_value",
                    color_continuous_scale="Blues",
                    labels={"median_house_value": "Actual Value (USD)"},
                    opacity=0.6,
                    title="California Housing Values (Background Census Blocks)"
                )
                # Add user selected point
                fig.add_scatter(
                    x=[longitude],
                    y=[latitude],
                    mode="markers",
                    marker=dict(color="#DC2626", size=14, symbol="star", line=dict(color="black", width=1.5)),
                    name="Your Selection"
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to standard streamlit map
                map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
                st.map(map_data, zoom=7, size=35)
            
        with col2:
            st.markdown("### 🏡 Structural & Census Parameters")
            
            # Sub-columns for parameters input
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                median_age = st.slider("Median House Age (Years)", 1.0, 52.0, 28.0)
                total_rooms = st.number_input("Total Rooms in Block", min_value=1.0, max_value=40000.0, value=2000.0, step=100.0)
                total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1.0, max_value=7000.0, value=400.0, step=10.0)
                
            with sub_col2:
                population = st.number_input("Total Population in Block", min_value=3.0, max_value=36000.0, value=1000.0, step=50.0)
                households = st.number_input("Total Households in Block", min_value=1.0, max_value=6000.0, value=350.0, step=10.0)
                median_income = st.slider("Median Household Income (x $10,000)", 0.5, 15.0, 3.5, step=0.1)
                
            ocean_proximity = st.selectbox(
                "Ocean Proximity Category", 
                ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
            )
            
            st.markdown("---")
            
            # Validation
            if total_bedrooms > total_rooms:
                st.warning("⚠️ Total bedrooms cannot exceed total rooms. Please adjust block counts.")
                predict_disabled = True
            else:
                predict_disabled = False
                
            if st.button("Calculate Estimated Value", type="primary", disabled=predict_disabled):
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
                    
                    # Typical error range based on Random Forest MAE (~$32,373)
                    lower_bound = max(15000, predicted_price - 32373)
                    upper_bound = predicted_price + 32373
                    
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div style="font-size: 0.9rem; color: #475569; text-transform: uppercase; font-weight: bold;">Estimated Median House Value</div>
                            <div class="metric-value">${predicted_price:,.0f}</div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.5rem;">
                                Expected typical valuation range: <b>${lower_bound:,.0f} — ${upper_bound:,.0f}</b> (based on model MAE of $32,373).
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Inference prediction failed: {e}")
                    
    with tab2:
        st.subheader("Model Comparison & Diagnostics")
        st.markdown(
            "Below are the final evaluation metrics computed on the 20% test split. "
            "The table highlights relative performance across the models."
        )
        
        # Load comparison CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comp_csv = os.path.join(current_dir, "data", "processed", "model_comparison.csv")
        
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)
            
            # Display using native st.dataframe with custom column formatting, resolving background color contrast bugs
            st.dataframe(
                comp_df,
                column_config={
                    "Model": st.column_config.TextColumn("Regression Model"),
                    "MAE": st.column_config.NumberColumn("Mean Absolute Error (USD)", format="$%,.2f"),
                    "RMSE": st.column_config.NumberColumn("Root Mean Squared Error (USD)", format="$%,.2f"),
                    "R2": st.column_config.NumberColumn("R-squared (Variance Explained)", format="%.4f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Metrics comparison table is not available. Please run the training pipeline to generate it.")
            
        st.markdown("---")
        st.subheader("Model Diagnostic Charts")
        
        # Display charts
        col_img1, col_img2 = st.columns(2)
        
        rmse_plot = os.path.join(current_dir, "images", "model_comparison_rmse.png")
        r2_plot = os.path.join(current_dir, "images", "model_comparison_r2.png")
        residuals_plot = os.path.join(current_dir, "images", "residuals_plot.png")
        importances_plot = os.path.join(current_dir, "images", "feature_importances.png")
        
        with col_img1:
            if os.path.exists(rmse_plot):
                st.image(rmse_plot, caption="Figure 1: Comparative Root Mean Squared Error (lower error is better)", use_container_width=True)
            if os.path.exists(residuals_plot):
                st.image(residuals_plot, caption="Figure 3: Predicted vs. Actual Values (Random Forest)", use_container_width=True)
                
        with col_img2:
            if os.path.exists(r2_plot):
                st.image(r2_plot, caption="Figure 2: Comparative R-squared Metric (higher R2 is better)", use_container_width=True)
            if os.path.exists(importances_plot):
                st.image(importances_plot, caption="Figure 4: Relative Feature Importance Weights (Random Forest)", use_container_width=True)

# Styled Footer Banner
st.markdown(
    """
    <div class="footer-bar">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <b>Synent Technologies</b> — Data Science Internship (Summer 2026)<br>
                Task 8: House Price Prediction ML Pipeline
            </div>
            <div style="text-align: right;">
                Developer: <b>Rudra Patel</b> (ID: <code>SYN/J2/IP806</code>)<br>
                Email: <a href="mailto:rudrapatel2156@gmail.com">rudrapatel2156@gmail.com</a> | 
                GitHub: <a href="https://github.com/Rudra2986" target="_blank">Rudra2986</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
