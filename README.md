# Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk

An end-to-end biomedical data science, statistical analysis, and deep learning project investigating cardiovascular disease risk determinants and predictive modeling using real-world patient survey data.

---

## 📌 Project Overview
Cardiovascular disease (CVD) is the leading global cause of mortality. This academic project provides a comprehensive, leak-free computational pipeline that analyzes demographic, lifestyle, and comorbid factors from the CDC Behavioral Risk Factor Surveillance System (BRFSS) cohort ($N = 999$).

The project implements:
1. **Dataset Profiling:** Audit of attributes, missing values, duplicates, and class imbalance.
2. **Leak-Free Preprocessing:** Three-way stratified partitioning (70% Train, 15% Validation, 15% Test) with pipelines fitted strictly on training data.
3. **Exploratory Data Analysis (EDA):** Descriptive statistics, distribution parameters, and Chi-square bivariate association tests.
4. **Data Visualization:** 8 high-resolution publication-quality figures covering distributions, outlier detection, correlations, and clinical trends.
5. **Feature Importance Analysis:** Information-theoretic Mutual Information and model-agnostic Permutation Importance.
6. **Deep Learning Model:** Multi-Layer Perceptron (MLP) built in TensorFlow/Keras with Dropout, Batch Normalization, and class-weighted cross-entropy.
7. **Hyperparameter Tuning:** Validation-driven sweep across 10 architectural configurations without test-set contamination.
8. **Final Benchmark:** Independent evaluation on 150 untouched test samples, with decision threshold calibration ($\theta = 0.480$) and 4 diagnostic plots.
9. **Interactive Dashboard:** Interactive web application deployed via Streamlit (`app.py`).

---

## 📂 Project Structure
```text
heart-disease-risk-prediction/
│
├── data/
│   ├── raw/
│   │   └── heart.csv                   # Raw survey dataset (999 rows, 11 columns)
│   └── processed/
│       ├── train.csv                   # 70% stratified training split (699 rows)
│       ├── val.csv                     # 15% validation split (150 rows)
│       ├── test.csv                    # 15% untouched test holdout (150 rows)
│       └── feature_metadata.json       # Feature mappings and class counts
│
├── notebooks/
│   └── heart_disease_eda.ipynb         # Complete, runnable Jupyter Notebook
│
├── outputs/
│   ├── figures/                        # 14 High-resolution publication plots (300 DPI)
│   │   ├── 01_target_distribution.png
│   │   ├── 02_numerical_histograms.png
│   │   ├── 03_numerical_boxplots.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_feature_vs_target.png
│   │   ├── 06_categorical_risk_factors.png
│   │   ├── 07_bmi_vs_weight_scatter.png
│   │   ├── 08_missing_values_matrix.png
│   │   ├── 09_feature_importance.png
│   │   ├── 10_hyperparameter_tuning_comparison.png
│   │   ├── 11_training_loss_curves.png
│   │   ├── 12_training_accuracy_curves.png
│   │   ├── 13_confusion_matrix.png
│   │   └── 14_roc_curve.png
│   ├── reports/
│   │   ├── dataset_profile_report.md
│   │   ├── exploratory_data_analysis.md
│   │   ├── visualization_interpretations.md
│   │   ├── feature_analysis_report.md
│   │   ├── final_project_report.md     # 21-section comprehensive academic report
│   │   └── assignment_checklist.md     # Traceability matrix
│   └── metrics/
│       ├── numerical_descriptive_stats.csv
│       ├── feature_importance.csv
│       ├── hyperparameter_tuning_results.csv
│       ├── classification_report.csv
│       └── test_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Dataset loading and profiling
│   ├── data_preprocessing.py           # Preprocessing pipeline and splits
│   ├── exploratory_analysis.py         # Statistical summaries and crosstabs
│   ├── visualizations.py               # Generation of EDA figures
│   ├── feature_analysis.py             # Mutual Information & Permutation Importance
│   ├── model_training.py               # Keras MLP model builder and trainer
│   ├── hyperparameter_tuning.py        # Grid search on validation split
│   ├── model_evaluation.py             # Test set evaluation and diagnostic plots
│   └── predict.py                      # Standalone prediction module
│
├── models/
│   ├── final_model.keras               # Serialized TensorFlow/Keras model
│   ├── preprocessor.joblib             # Serialized scikit-learn preprocessor
│   ├── best_hyperparameters.json       # Optimal hyperparameters
│   └── model_metadata.json             # Architecture & threshold metadata
│
├── app.py                              # Streamlit interactive web application
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation and instructions
└── .gitignore                          # Git ignore rules
```

---

## 📊 Key Findings & Experimental Results

### Dataset Summary
- **Total Patients:** 999 records
- **Class Balance:** 858 No Heart Disease (85.89%), 141 Heart Disease (14.11%)
- **Imbalance Ratio:** 6.09:1
- **Engineered Feature Dimensions:** 32 (via One-Hot Encoding and StandardScaler)

### Final Test Set Performance (150 Untouched Samples)
| Metric | Optimal Threshold ($\theta = 0.480$) | Default Threshold ($\theta = 0.500$) |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7305** | 0.7305 |
| **Sensitivity (Recall)** | **52.38%** (11 / 21) | 42.86% (9 / 21) |
| **Specificity** | **77.52%** (100 / 129) | 78.29% (101 / 129) |
| **Precision** | **27.50%** (11 / 40) | 24.32% (9 / 37) |
| **Accuracy** | **74.00%** (111 / 150) | 73.33% (110 / 150) |
| **F1-Score** | **0.3607** | 0.3103 |

**Test Confusion Matrix:**
- **True Negatives (TN):** 100
- **False Positives (FP):** 29
- **False Negatives (FN):** 10
- **True Positives (TP):** 11

---

## 🚀 Execution Guide (Windows PowerShell)

### 1. Environment Setup
Clone or navigate to the project directory:
```powershell
cd d:\heart-disease-risk-prediction
```

Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run Complete Pipeline Modules
Execute each phase sequentially:

```powershell
# Phase 1: Data Profiling & Audit
python src/data_loader.py

# Phase 3: Leak-Free Preprocessing & Stratified Splits
python src/data_preprocessing.py

# Phase 4: Exploratory Data Analysis & Descriptive Statistics
python src/exploratory_analysis.py

# Phase 5: High-Resolution Visualizations (Generates Figures 1-8)
python src/visualizations.py

# Phase 6: Feature Importance Analysis (Generates Figure 9)
python src/feature_analysis.py

# Phase 8: Validation-Driven Hyperparameter Tuning (Generates Figure 10)
python src/hyperparameter_tuning.py

# Phase 9: Final Model Training & Test Evaluation (Generates Figures 11-14)
python src/model_evaluation.py

# Phase 10: Verify Standalone Prediction Engine
python src/predict.py
```

### 3. Launch Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
*Access the application in your browser at `http://localhost:8501`.*

### 4. Run Jupyter Notebook
```powershell
jupyter notebook notebooks/heart_disease_eda.ipynb
```

---

## 🔬 Educational Medical Disclaimer
> **IMPORTANT:** This software application, deep learning model, and academic analysis were produced strictly for instructional, educational, and research purposes. The predicted probabilities represent computational estimates derived from retrospective survey observations and **do not constitute clinical diagnostic opinions, personalized medical advice, or prognostic validation**. Individuals seeking medical evaluation must consult a qualified physician or licensed cardiologist.
