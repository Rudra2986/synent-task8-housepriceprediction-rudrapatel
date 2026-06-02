"""
Model Comparison Starter Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module generates comparison tables and summary charts of model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_models(metrics_dict: dict) -> pd.DataFrame:
    """
    Formats multi-model metrics into a comparative DataFrame.
    
    Args:
        metrics_dict (dict): Dictionary mapping Model Name -> Metrics Dictionary
        
    Returns:
        pd.DataFrame: Summarized comparison table.
    """
    print("Compiling model comparisons...")
    # TODO: Construct DataFrame from dictionary metrics
    return pd.DataFrame()


def plot_metric_comparison(comparison_df: pd.DataFrame, metric: str, output_path: str):
    """
    Saves a comparison bar chart for a specific metric (e.g. RMSE).
    """
    print(f"Generating plot for metric: {metric}...")
    # TODO: plt.figure(), sns.barplot(), plt.savefig()
    pass


if __name__ == "__main__":
    print("Running Model Comparison starter script.")
