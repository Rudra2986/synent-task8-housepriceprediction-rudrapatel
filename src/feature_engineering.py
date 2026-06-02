"""
Feature Engineering Starter Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles transformation of raw categories into numerical representations
and creation of new interaction terms.
"""

import pandas as pd


def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Applies one-hot or label encoding to categorical columns.
    """
    print(f"Encoding categorical features: {categorical_cols}")
    # TODO: Implement encoding logic (e.g. pd.get_dummies)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives new fields such as price per area, age of property, or total amenity scores.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        
    Returns:
        pd.DataFrame: Dataset with added features.
    """
    print("Creating derived features...")
    # TODO: Implement domain-specific feature creation (e.g. Total_SF = GrLivArea + TotalBsmtSF)
    return df


if __name__ == "__main__":
    print("Running Feature Engineering starter script.")
