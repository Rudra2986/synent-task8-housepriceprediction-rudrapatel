"""
Model Training Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles splitting data, training regressors, and serializing models.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataframe into training and testing sets.
    
    Args:
        df (pd.DataFrame): Dataset.
        target_col (str): Name of the target variable.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Controls shuffling.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data with target column '{target_col}', test_size={test_size}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Trains a Linear Regression model.
    """
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 10, random_state: int = 42
) -> DecisionTreeRegressor:
    """
    Trains a Decision Tree model.
    To avoid extreme overfitting, we cap max_depth.
    """
    print(f"Training Decision Tree Regressor (max_depth={max_depth})...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42
) -> RandomForestRegressor:
    """
    Trains a Random Forest model.
    """
    print(f"Training Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, output_path: str):
    """
    Serializes a trained model to a file using joblib.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    try:
        joblib.dump(model, output_path)
        print(f"Successfully saved model to: {output_path}")
    except Exception as e:
        print(f"Error saving model to {output_path}: {e}")
        raise e


if __name__ == "__main__":
    print("Running Model Training Module demo...")
