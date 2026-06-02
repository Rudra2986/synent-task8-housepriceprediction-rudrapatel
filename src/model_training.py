"""
Model Training Starter Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module handles splitting data, initializing regressors, and fitting models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataframe into training and testing sets.
    """
    print(f"Splitting data with test_size={test_size}...")
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


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeRegressor:
    """
    Trains a Decision Tree model.
    """
    print("Training Decision Tree Regressor...")
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a Random Forest model.
    """
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("Running Model Training starter script.")
