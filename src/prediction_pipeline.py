"""
Prediction Pipeline Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles loading the saved champion model and running inference 
on new, raw housing data by applying identical preprocessing and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import joblib

# Import cleaning and engineering functions to reuse them during inference
from src.data_cleaning import clean_house_data
from src.feature_engineering import create_derived_features, encode_categorical_features


def load_champion_model(model_path: str):
    """
    Loads serialized model artifact.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at: {model_path}")
    print(f"Loading champion model from: {model_path}")
    return joblib.load(model_path)


def load_preprocessor(preprocessor_path: str) -> dict:
    """
    Loads serialized preprocessor metadata and encoder objects.
    """
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor artifacts not found at: {preprocessor_path}")
    print(f"Loading preprocessor from: {preprocessor_path}")
    return joblib.load(preprocessor_path)


def run_inference(model, preprocessor: dict, input_features: pd.DataFrame) -> np.ndarray:
    """
    Executes pricing predictions on raw input housing attributes.
    Applies imputation, derived feature creation, category encoding, and feature alignment.
    
    Args:
        model: Trained regressor.
        preprocessor (dict): Preprocessor metadata dictionary containing:
                             - 'imputer_values': median values for imputation
                             - 'categorical_cols': list of categorical columns
                             - 'encoder': fitted OneHotEncoder
                             - 'feature_names_out': list of columns model expects
        input_features (pd.DataFrame): Raw input features (e.g. columns like longitude, total_rooms, ocean_proximity).
        
    Returns:
        np.ndarray: Predicted house prices.
    """
    df = input_features.copy()
    
    # 1. Imputation
    imputer_values = preprocessor.get("imputer_values", {})
    # For inference, we only run missing value imputation, not outlier dropping.
    for col, value in imputer_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
            
    # 2. Create derived ratio features
    df = create_derived_features(df)
    
    # 3. Categorical encoding
    categorical_cols = preprocessor.get("categorical_cols", [])
    encoder = preprocessor.get("encoder")
    if categorical_cols and encoder:
        df, _ = encode_categorical_features(df, categorical_cols, encoder=encoder)
        
    # 4. Feature Alignment
    # Ensure all columns in training feature list exist in inference dataframe, in the correct order.
    expected_cols = preprocessor.get("feature_names_out", [])
    if not expected_cols:
        raise ValueError("Preprocessor metadata does not contain 'feature_names_out'. Cannot align features.")
        
    # Check if target column is in expected columns and remove it if it is
    expected_features = [col for col in expected_cols if col != "median_house_value"]
    
    # Reindex columns, filling missing ones with 0.0
    df_aligned = df.reindex(columns=expected_features, fill_value=0.0)
    
    print(f"Features aligned. Input shape: {input_features.shape} -> Preprocessed shape: {df_aligned.shape}")
    
    # 5. Make predictions
    predictions = model.predict(df_aligned)
    return predictions


if __name__ == "__main__":
    print("Testing Prediction Pipeline starter script.")
    # Example raw inputs
    example_raw = pd.DataFrame([{
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }])
    
    model_p = os.path.join("models", "house_price_model.joblib")
    prep_p = os.path.join("models", "preprocessor.joblib")
    
    if os.path.exists(model_p) and os.path.exists(prep_p):
        model = load_champion_model(model_p)
        prep = load_preprocessor(prep_p)
        preds = run_inference(model, prep, example_raw)
        print("Predicted Price:", preds[0])
    else:
        print("Models and preprocessor are not trained/saved yet.")
