"""
Feature Engineering Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles transformation of raw categories into numerical representations
and creation of new interaction terms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def encode_categorical_features(
    df: pd.DataFrame, categorical_cols: list, encoder: OneHotEncoder = None
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Applies one-hot encoding to categorical columns using scikit-learn's OneHotEncoder.
    This guarantees that the encoded columns match between training, testing, and inference.
    
    Args:
        df (pd.DataFrame): Input housing dataset.
        categorical_cols (list): List of categorical column names to encode.
        encoder (OneHotEncoder, optional): A pre-fitted OneHotEncoder. If None, fits a new one.
        
    Returns:
        tuple[pd.DataFrame, OneHotEncoder]: Transformed dataframe and the fitted encoder object.
    """
    df_encoded = df.copy()
    
    if not categorical_cols:
        return df_encoded, encoder
        
    # Check which categorical columns actually exist in the dataframe
    cols_to_encode = [col for col in categorical_cols if col in df_encoded.columns]
    
    if not cols_to_encode:
        return df_encoded, encoder
        
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df_encoded[cols_to_encode])
        
    # Transform categorical features
    encoded_arr = encoder.transform(df_encoded[cols_to_encode])
    
    # Get proper feature names for one-hot encoded columns
    encoded_feature_names = encoder.get_feature_names_out(cols_to_encode)
    
    # Create DataFrame from encoded array
    encoded_df = pd.DataFrame(
        encoded_arr, 
        columns=encoded_feature_names, 
        index=df_encoded.index
    )
    
    # Drop original categorical columns and concatenate encoded columns
    df_encoded = df_encoded.drop(columns=cols_to_encode)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded, encoder


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives new ratio features (e.g. rooms per household, bedrooms per room, 
    population per household) to capture density and average dwelling size.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        
    Returns:
        pd.DataFrame: Dataset with added features.
    """
    df_feat = df.copy()
    
    # Prevent divide by zero issues with epsilon values (though cleaning filters out <= 0 counts)
    epsilon = 1e-5
    
    df_feat["rooms_per_household"] = df_feat["total_rooms"] / (df_feat["households"] + epsilon)
    df_feat["bedrooms_per_room"] = df_feat["total_bedrooms"] / (df_feat["total_rooms"] + epsilon)
    df_feat["population_per_household"] = df_feat["population"] / (df_feat["households"] + epsilon)
    
    print("Successfully created derived features: ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']")
    return df_feat


if __name__ == "__main__":
    print("Testing Feature Engineering module...")
    # Mock data to test OneHotEncoder integration
    data = pd.DataFrame({
        "total_rooms": [10, 20],
        "total_bedrooms": [2, 4],
        "population": [50, 100],
        "households": [5, 10],
        "ocean_proximity": ["NEAR BAY", "INLAND"]
    })
    data_feat = create_derived_features(data)
    data_encoded, enc = encode_categorical_features(data_feat, ["ocean_proximity"])
    print("Transformed data shape:", data_encoded.shape)
    print("Columns:", data_encoded.columns.tolist())
