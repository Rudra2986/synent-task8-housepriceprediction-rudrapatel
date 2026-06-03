"""
Main Runner Script - House Price Prediction ML Pipeline
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This script automates the complete data science workflow:
1. Data ingestion & cleaning
2. Feature engineering & encoding
3. Data partition (Train/Test splits saved to processed folder)
4. Training multiple regression models
5. Model evaluation and metrics generation (MAE, RMSE, R-squared)
6. Model selection and saving the champion model
7. Creating professional performance plots
8. Run sample inference checks
"""

import os
import sys

# Redirection for Streamlit Cloud deployment:
# If this file (main.py) is executed under a Streamlit context or if the raw dataset is missing,
# we redirect execution to streamlit_app.py to run the dashboard.
is_streamlit = any("streamlit" in arg for arg in sys.argv)
raw_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "housing.csv")

if is_streamlit or not os.path.exists(raw_data_path):
    import streamlit_app
    sys.exit(0)

import pandas as pd
import numpy as np
import joblib

# Import modular components from src/
from src.data_cleaning import load_raw_data, clean_house_data
from src.feature_engineering import create_derived_features, encode_categorical_features
from src.model_training import split_data, train_linear_regression, train_decision_tree, train_random_forest, save_model
from src.model_evaluation import evaluate_regression_model
from src.model_comparison import compare_models, plot_metric_comparison, plot_residuals, plot_feature_importance
from src.prediction_pipeline import load_champion_model, load_preprocessor, run_inference


def main():
    print("=" * 60)
    print("STARTING HOUSE PRICE PREDICTION ML PIPELINE")
    print("=" * 60)
    
    # 1. Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, "data", "raw", "housing.csv")
    processed_dir = os.path.join(current_dir, "data", "processed")
    models_dir = os.path.join(current_dir, "models")
    images_dir = os.path.join(current_dir, "images")
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 2. Ingest & Clean Raw Data
    df_raw = load_raw_data(raw_data_path)
    df_cleaned, imputer_values = clean_house_data(df_raw)
    
    # 3. Feature Engineering
    df_engineered = create_derived_features(df_cleaned)
    categorical_cols = ["ocean_proximity"]
    df_processed, encoder = encode_categorical_features(df_engineered, categorical_cols)
    
    # Keep track of features the model expects
    feature_names_out = df_processed.columns.tolist()
    
    # Save a small sample of coordinates for the interactive map visualization in Streamlit
    map_sample = df_cleaned[["longitude", "latitude", "median_house_value"]].sample(n=1200, random_state=42)
    map_sample.to_csv(os.path.join(processed_dir, "map_sample.csv"), index=False)
    print(f"Saved interactive map sample to: {os.path.join(processed_dir, 'map_sample.csv')}")
    
    # 4. Save Processed Datasets
    # Split features and target to save them
    target_col = "median_house_value"
    
    X_train, X_test, y_train, y_test = split_data(
        df_processed, 
        target_col=target_col, 
        test_size=0.2, 
        random_state=42
    )
    
    # Save train and test sets to CSV
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(processed_dir, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_data.csv"), index=False)
    print(f"Saved processed train and test sets to: {processed_dir}")
    
    # 5. Train Models
    models = {}
    models["Linear Regression"] = train_linear_regression(X_train, y_train)
    models["Decision Tree"] = train_decision_tree(X_train, y_train, max_depth=10)
    models["Random Forest"] = train_random_forest(X_train, y_train, n_estimators=60, max_depth=14)
    
    # 6. Evaluate Models
    metrics = {}
    predictions_dict = {}
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        metrics[name] = evaluate_regression_model(model, X_test, y_test)
        predictions_dict[name] = model.predict(X_test)
        
    # 7. Model Comparison & Selection
    comparison_df = compare_models(metrics)
    comparison_df.to_csv(os.path.join(processed_dir, "model_comparison.csv"), index=False)
    
    # Select the champion model (highest R2 score)
    best_model_name = comparison_df.loc[comparison_df["R2"].idxmax(), "Model"]
    best_model = models[best_model_name]
    print(f"\n[CHAMPION] Best Performing Model: {best_model_name}")
    
    # 8. Save Champion Model & Preprocessor
    model_output_path = os.path.join(models_dir, "house_price_model.joblib")
    preprocessor_output_path = os.path.join(models_dir, "preprocessor.joblib")
    
    save_model(best_model, model_output_path)
    
    # Package preprocessor values
    preprocessor_metadata = {
        "imputer_values": imputer_values,
        "categorical_cols": categorical_cols,
        "encoder": encoder,
        "feature_names_out": feature_names_out
    }
    joblib.dump(preprocessor_metadata, preprocessor_output_path)
    print(f"Successfully saved preprocessor to: {preprocessor_output_path}")
    
    # 9. Generate & Save Visualizations
    plot_metric_comparison(comparison_df, "RMSE", os.path.join(images_dir, "model_comparison_rmse.png"))
    plot_metric_comparison(comparison_df, "R2", os.path.join(images_dir, "model_comparison_r2.png"))
    
    # Residual plot for Champion model
    plot_residuals(
        y_test, 
        predictions_dict[best_model_name], 
        best_model_name, 
        os.path.join(images_dir, "residuals_plot.png")
    )
    
    # Feature Importance for Champion model (if applicable)
    if hasattr(best_model, "feature_importances_"):
        # Features are columns of X_train
        plot_feature_importance(
            best_model.feature_importances_,
            X_train.columns.tolist(),
            best_model_name,
            os.path.join(images_dir, "feature_importances.png")
        )
        
    # 10. Verification Prediction Check using Pipeline
    print("\n" + "=" * 40)
    print("VERIFYING INFERENCE PIPELINE")
    print("=" * 40)
    loaded_model = load_champion_model(model_output_path)
    loaded_prep = load_preprocessor(preprocessor_output_path)
    
    # Use first few rows of raw dataset (dropping target variable if present)
    test_sample = df_raw.iloc[:3].drop(columns=[target_col], errors="ignore")
    sample_preds = run_inference(loaded_model, loaded_prep, test_sample)
    
    print("\nValidation Sample Inputs:")
    print(test_sample[["longitude", "latitude", "total_rooms", "median_income", "ocean_proximity"]])
    print("\nInference Pipeline Predicted House Prices:")
    for idx, pred in enumerate(sample_preds):
        actual = df_raw.loc[idx, target_col]
        print(f"Sample {idx+1}: Predicted = ${pred:,.2f} | Actual = ${actual:,.2f} | Error = ${abs(pred-actual):,.2f}")
        
    print("\n" + "=" * 60)
    print("HOUSE PRICE PREDICTION ML PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
