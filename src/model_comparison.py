"""
Model Comparison Module
Synent Technologies - Data Science Internship (Summer 2026)
Task 8: House Price Prediction (ML Model)

This module generates comparison tables and summary charts of model performance,
including actual vs. predicted plots and feature importance charts.
All charts use a clean, professional, and consistent muted olive-green aesthetic (Wabi-Sabi style).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compare_models(metrics_dict: dict) -> pd.DataFrame:
    """
    Formats multi-model metrics into a comparative DataFrame.
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
    using a clean, professional muted olive-green theme.
    """
    print(f"Generating plot for metric: {metric}...")
    
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    plt.figure(figsize=(7, 4.5))
    
    # Warm cream background and olive accent
    accent_color = "#4A5844"  # Muted Olive Green
    
    sns.set_theme(style="ticks")
    
    # Sort values to keep layout readable
    sorted_df = comparison_df.sort_values(by=metric, ascending=(metric != "R2"))
    
    ax = sns.barplot(
        x="Model", 
        y=metric, 
        data=sorted_df, 
        color=accent_color,
        edgecolor="#2C2B29",
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
                            fontsize=10, color="#2C2B29", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points")
            else:
                ax.annotate(f"${height:,.0f}", 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha="center", va="bottom", 
                            fontsize=10, color="#2C2B29", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points")
            
    plt.title(f"Model Comparison: {metric}", fontsize=12, fontweight="bold", color="#2C2B29", pad=15)
    plt.xlabel("Model", fontsize=10, color="#6B6862")
    plt.ylabel(metric if metric == "R2" else f"{metric} (USD)", fontsize=10, color="#6B6862")
    
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
    
    # Muted sage/olive green points
    plt.scatter(y_true, y_pred, alpha=0.25, color="#5A6C56", edgecolors="none", s=20)
    
    # Soft rust/brick red identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#8C3B32", linestyle="-", linewidth=1.5, label="Perfect Agreement")
    
    plt.title(f"Actual vs. Predicted Values ({model_name})", fontsize=12, fontweight="bold", color="#2C2B29", pad=15)
    plt.xlabel("Actual Median House Value (USD)", fontsize=10, color="#6B6862")
    plt.ylabel("Predicted Median House Value (USD)", fontsize=10, color="#6B6862")
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
    
    # Muted Olive Green
    feat_importances.plot(kind="barh", color="#4A5844", edgecolor="#2C2B29", linewidth=1)
    
    plt.title(f"Feature Importances ({model_name})", fontsize=12, fontweight="bold", color="#2C2B29", pad=15)
    plt.xlabel("Relative Importance Weight", fontsize=10, color="#6B6862")
    plt.ylabel("Dataset Features", fontsize=10, color="#6B6862")
    
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to: {output_path}")


if __name__ == "__main__":
    print("Testing Model Comparison module...")
