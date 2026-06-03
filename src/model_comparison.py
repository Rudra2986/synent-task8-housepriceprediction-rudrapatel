"""
Model Comparison Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module generates comparison tables and summary charts of model performance,
including actual vs. predicted plots and feature importance charts.
All charts use a clean, professional, and consistent aesthetic.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compare_models(metrics_dict: dict) -> pd.DataFrame:
    """
    Formats multi-model metrics into a comparative DataFrame.
    
    Args:
        metrics_dict (dict): Dictionary mapping Model Name -> Metrics Dictionary
                             e.g., {"Linear Regression": {"MAE": 50000, ...}}
                             
    Returns:
        pd.DataFrame: Summarized comparison table.
    """
    print("Compiling model comparisons...")
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.index.name = "Model"
    df = df.reset_index()
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    return df


def plot_metric_comparison(comparison_df: pd.DataFrame, metric: str, output_path: str):
    """
    Saves a comparison bar chart for a specific metric (e.g. RMSE, R2, MAE)
    using a clean, professional single-color palette.
    """
    print(f"Generating plot for metric: {metric}...")
    
    # Check if directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    plt.figure(figsize=(7, 4.5))
    
    # Clean professional navy/slate theme
    accent_color = "#1A365D"  # Deep Navy Blue
    
    sns.set_theme(style="ticks")
    
    # Highlight the best model with a slightly different shade if R2, or just use clean navy
    # We sort values to keep layout readable
    sorted_df = comparison_df.sort_values(by=metric, ascending=(metric != "R2"))
    
    ax = sns.barplot(
        x="Model", 
        y=metric, 
        data=sorted_df, 
        color=accent_color,
        edgecolor="#0F172A",
        linewidth=1
    )
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height):
            if metric == "R2":
                ax.annotate(f"{height:.4f}", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha="center", va="bottom", 
                            fontsize=10, color="#1E293B", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points")
            else:
                ax.annotate(f"${height:,.0f}", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha="center", va="bottom", 
                            fontsize=10, color="#1E293B", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points")
            
    plt.title(f"Model Comparison: {metric}", fontsize=12, fontweight="bold", color="#0F172A", pad=15)
    plt.xlabel("Model", fontsize=10, color="#334155")
    plt.ylabel(metric if metric == "R2" else f"{metric} (USD)", fontsize=10, color="#334155")
    
    # Clean top/right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved metric comparison chart to: {output_path}")


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str, output_path: str):
    """
    Saves a scatter plot of Actual vs. Predicted values along with the identity line.
    """
    print(f"Generating actual vs. predicted plot for {model_name}...")
    
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    plt.figure(figsize=(6.5, 5.5))
    sns.set_theme(style="ticks")
    
    # Professional slate blue points with low alpha to show density
    plt.scatter(y_true, y_pred, alpha=0.25, color="#2B6CB0", edgecolors="none", s=20)
    
    # Simple, solid identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#E53E3E", linestyle="-", linewidth=1.5, label="Perfect Agreement")
    
    plt.title(f"Actual vs. Predicted Values ({model_name})", fontsize=12, fontweight="bold", color="#0F172A", pad=15)
    plt.xlabel("Actual Median House Value (USD)", fontsize=10, color="#334155")
    plt.ylabel("Predicted Median House Value (USD)", fontsize=10, color="#334155")
    plt.legend(loc="upper left", frameon=True)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved residual plot to: {output_path}")


def plot_feature_importance(importances: np.ndarray, feature_names: list, model_name: str, output_path: str):
    """
    Saves a bar chart of feature importances.
    """
    print(f"Generating feature importance plot for {model_name}...")
    
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances = feat_importances.sort_values(ascending=True)
    
    plt.figure(figsize=(8, 5.5))
    sns.set_theme(style="ticks")
    
    # Consistent Navy Blue
    feat_importances.plot(kind="barh", color="#1A365D", edgecolor="#0F172A", linewidth=1)
    
    plt.title(f"Feature Importances ({model_name})", fontsize=12, fontweight="bold", color="#0F172A", pad=15)
    plt.xlabel("Relative Importance Weight", fontsize=10, color="#334155")
    plt.ylabel("Dataset Features", fontsize=10, color="#334155")
    
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to: {output_path}")


if __name__ == "__main__":
    print("Testing Model Comparison module...")
