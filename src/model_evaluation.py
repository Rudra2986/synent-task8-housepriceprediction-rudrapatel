"""
Model Evaluation Starter Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module computes standard regression metrics (MAE, RMSE, R-squared).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Computes performance metrics for a trained regressor on test data.
    
    Args:
        model: Trained scikit-learn regressor.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test actual values.
        
    Returns:
        dict: Dictionary containing MAE, RMSE, and R2.
    """
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
    return metrics


if __name__ == "__main__":
    print("Running Model Evaluation starter script.")
