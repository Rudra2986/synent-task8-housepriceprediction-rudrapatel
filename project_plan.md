# Project Plan: House Price Prediction
> **Synent Technologies - Data Science Internship (Summer 2026)**
> **Task 8: Advanced Level**

---

## 🎯 Objectives
The primary objective of this project is to construct a robust regression-based Machine Learning pipeline to predict house prices.

Key targets:
1. Conduct detailed preprocessing on raw housing data (imputation, encoding, outlier removal).
2. Perform feature engineering to extract high-relevance predictors (e.g. ratios, quality weights).
3. Train three machine learning models: Linear Regression, Decision Tree, and Random Forest.
4. Assess and compare model performance using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$) coefficients.
5. Create an inference script/pipeline to save and load the best model and make predictions.

---

## 🔄 Expected Workflow
The project will follow a structured machine learning lifecycle:
```
[1. Project Setup] ➔ [2. Data Preprocessing] ➔ [3. Feature Engineering]
                                                         │
[6. Export & Submission] 🡨 [5. Evaluation & Comparison] 🡨 [4. Training Models]
```

1. **Project Setup (Current State):** Creating folder directories, python configurations, and project templates.
2. **Data Preprocessing:** Impute missing categorical/numerical values, drop records with invalid fields, filter out outliers (extremely large prices or areas).
3. **Feature Engineering:** Build interaction terms, apply category encoding, split columns where needed, and perform correlation filter passes.
4. **Training Models:** Train Linear Regression, Decision Tree, and Random Forest estimators on an 80/20 train/test split.
5. **Evaluation & Comparison:** Run diagnostic plots on residuals. Tabulate MAE, RMSE, and $R^2$ values to select the champion model.
6. **Export & Submission:** Save model artifacts via joblib, document code and conclusions, and construct the demonstration video.

---

## 📦 Deliverables
The final submission must include:
1. **GitHub Repository:** Publicly accessible, following naming format: `synent-task8-housepriceprediction-rudrapatel`.
2. **Data Folders:** Structured as `data/raw/` and `data/processed/` (train/test datasets).
3. **Jupyter Notebooks:** Complete execution workflow documenting data science steps and models.
4. **Python Scripts:** Clean code files in `src/` directory.
5. **Model Binaries:** Saved `.pkl` or `.joblib` model weights inside the `models/` directory.
6. **Visualizations:** Performance plots under `images/` directory.
7. **Video Demonstration:** A 1 to 3-minute video walk-through demonstrating execution, code architecture, model evaluations, and predictions.

---

## 🖥️ GitHub Submission Requirements
* **Repository Name:** `synent-task8-housepriceprediction-rudrapatel`
* **Privacy:** Public repositories only.
* **Documentation:** The `README.md` must be thoroughly written, capturing problem statements, datasets, results, and environment installation instructions.
* **Version Control:** Regular, logical commits describing the stages of development.

---

## 🎥 Video Demonstration Requirements
* **Duration:** 1 to 3 minutes.
* **Voiceover:** Preferred (explain clear audio, screen sharing, and code walk-through).
* **Key Components to Cover:**
  - Introduction of yourself (your name).
  - High-level overview of the problem statement and dataset.
  - Brief code walk-through of the preprocessing and feature engineering steps.
  - Show code for splitting data, training models, and print the resulting performance comparison tables (RMSE, MAE, $R^2$).
  - Demonstrate calling the prediction pipeline script on sample inputs to produce estimated house prices.

---

## 📈 Expected Outputs
Upon project completion, the following output targets should be realized:
- **Cleaned Data Splits:** Saved in `data/processed/` directory.
- **Model Comparison Table:** Summary table evaluating metrics across the models.
- **Saved Model:** Champion model saved at `models/house_price_model.joblib`.
- **Evaluation Plots:** Feature importance charts and residual plots saved under the `images/` directory.
- **Inference Pipeline:** A python script `src/prediction_pipeline.py` which loads the saved model and executes predictions on user inputs.
