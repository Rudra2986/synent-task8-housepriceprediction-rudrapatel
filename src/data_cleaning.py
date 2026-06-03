"""
Data Cleaning Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles initial loading, missing value imputation, and outlier treatment.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw house price transactional data from a CSV file.
    
    Args:
        file_path (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset, or empty DataFrame on failure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
        
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded raw data from: {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading raw data from {file_path}: {e}")
        raise e


def clean_house_data(df: pd.DataFrame, imputer_values: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Imputes missing values and removes extreme invalid outliers.
    
    Args:
        df (pd.DataFrame): Input housing dataset.
        imputer_values (dict, optional): Dictionary of precalculated imputation values
                                          (e.g., {'total_bedrooms': 435.0}). If None, values
                                          are calculated from the dataset (for training).
                                          
    Returns:
        tuple[pd.DataFrame, dict]: The cleaned dataframe and the imputer values used.
    """
    df_clean = df.copy()
    
    # 1. Missing Value Imputation
    if imputer_values is None:
        imputer_values = {
            "total_bedrooms": float(df_clean["total_bedrooms"].median())
        }
        print(f"Calculated training median for 'total_bedrooms': {imputer_values['total_bedrooms']}")
        
    # Apply imputation
    df_clean["total_bedrooms"] = df_clean["total_bedrooms"].fillna(imputer_values["total_bedrooms"])
    
    # 2. Outlier / Invalid Data Filtering (Only for training/cleaning phase, check logic)
    # We remove rows with negative or zero values in columns where it's physically impossible.
    non_negative_cols = [
        "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "median_income", "median_house_value"
    ]
    
    # We check columns that exist in the dataframe (median_house_value might not be in inference)
    cols_to_check = [c for c in non_negative_cols if c in df_clean.columns]
    
    initial_rows = len(df_clean)
    for col in cols_to_check:
        df_clean = df_clean[df_clean[col] > 0]
        
    removed_invalid = initial_rows - len(df_clean)
    if removed_invalid > 0:
        print(f"Removed {removed_invalid} rows with negative, zero, or missing values in numeric fields.")
        
    # 3. Filter extreme physical outliers (e.g. total_bedrooms > total_rooms)
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean["total_bedrooms"] <= df_clean["total_rooms"]]
    removed_rooms = initial_rows - len(df_clean)
    if removed_rooms > 0:
        print(f"Removed {removed_rooms} rows where total_bedrooms was greater than total_rooms.")
        
    # We can also filter extreme housing price values or rooms per household ratios in training,
    # but let's keep it clean and robust without over-filtering valid dense block groups.
    
    return df_clean, imputer_values


if __name__ == "__main__":
    print("Testing House Data Cleaning module...")
    raw_path = os.path.join("data", "raw", "housing.csv")
    if os.path.exists(raw_path):
        df = load_raw_data(raw_path)
        df_clean, imputers = clean_house_data(df)
        print("Imputers:", imputers)
        print("Cleaned shape:", df_clean.shape)
    else:
        print(f"Raw data file not found at: {raw_path}")
