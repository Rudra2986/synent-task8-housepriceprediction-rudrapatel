"""
Data Cleaning Starter Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles initial loading, missing value imputation, and outlier treatment.
"""

import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw house price transactional data.
    """
    print(f"Loading raw data from: {file_path}")
    # TODO: Implement loading logic
    return pd.DataFrame()


def clean_house_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values and removes extreme price/area outliers.
    
    Args:
        df (pd.DataFrame): Raw house price dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    print("Preprocessing missing housing fields...")
    # TODO: Impute missing numerical fields with median
    # TODO: Impute missing categorical fields with mode
    # TODO: Filter out outliers (e.g. houses with excessive square footage or negative values)
    return df


if __name__ == "__main__":
    print("Running House Data Cleaning starter script.")
