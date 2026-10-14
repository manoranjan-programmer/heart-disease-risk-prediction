# College Assignment Requirement Traceability Checklist

**Project Title:** Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk  
**Status:** 100% Fully Implemented and Verified  

---

| Phase / Requirement | Specific Deliverable / Feature | Status | Implementation File / Artifact Location |
| :--- | :--- | :--- | :--- |
| **Phase 1: Dataset Inspection** | Locate and inspect dataset (`heart.csv`) | Completed | [`data/raw/heart.csv`](file:///d:/heart-disease-risk-prediction/data/raw/heart.csv), [`src/data_loader.py`](file:///d:/heart-disease-risk-prediction/src/data_loader.py) |
| | Report rows, cols, dtypes, missing, duplicates | Completed | [`outputs/reports/dataset_profile_report.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/dataset_profile_report.md) |
| | Statistical summary of numerical features | Completed | [`outputs/metrics/numerical_descriptive_stats.csv`](file:///d:/heart-disease-risk-prediction/outputs/metrics/numerical_descriptive_stats.csv) |
| | Target class distribution and imbalance | Completed | 858 No (85.89%) vs 141 Yes (14.11%), 6.09:1 ratio |
| | Identifier/leakage columns audit | Completed | 0 ID or leakage columns found |
| **Phase 2: Project Structure** | Modular folder layout (`src/`, `data/`, `outputs/`, `notebooks/`, `models/`) | Completed | Root workspace directory structure |
| **Phase 3: Data Preprocessing** | Missing-value counts, matrix plot, imputation | Completed | [`outputs/figures/08_missing_values_matrix.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/08_missing_values_matrix.png), [`src/data_preprocessing.py`](file:///d:/heart-disease-risk-prediction/src/data_preprocessing.py) |
| | Duplicate records audit | Completed | 0 duplicates documented in profile report |
| | Outlier detection (IQR, boxplots, clinical rationale) | Completed | 22 Weight, 30 BMI outliers preserved; [`outputs/figures/03_numerical_boxplots.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/03_numerical_boxplots.png) |
| | Categorical encoding (One-Hot) & Scaling (StandardScaler) | Completed | 32 engineered features; fitted strictly on train |
| | Stratified split (70% Train, 15% Val, 15% Test) | Completed | [`data/processed/train.csv`](file:///d:/heart-disease-risk-prediction/data/processed/train.csv), [`data/processed/val.csv`](file:///d:/heart-disease-risk-prediction/data/processed/val.csv), [`data/processed/test.csv`](file:///d:/heart-disease-risk-prediction/data/processed/test.csv) |
| | Serialization of preprocessor pipeline | Completed | [`models/preprocessor.joblib`](file:///d:/heart-disease-risk-prediction/models/preprocessor.joblib) |
| **Phase 4: Exploratory Analysis** | Mean, median, mode, std, IQR, skewness, kurtosis | Completed | [`outputs/metrics/numerical_descriptive_stats.csv`](file:///d:/heart-disease-risk-prediction/outputs/metrics/numerical_descriptive_stats.csv) |
| | Contingency tables, prevalence, Chi-square tests | Completed | [`outputs/reports/exploratory_data_analysis.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/exploratory_data_analysis.md) |
| | Correlation analysis & collinearity evaluation | Completed | Weight vs BMI ($r = 0.85$); documented in EDA report |
| **Phase 5: Data Visualization** | 1. Target Class Distribution | Completed | [`outputs/figures/01_target_distribution.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/01_target_distribution.png) |
| | 2. Numerical Feature Histograms with KDE | Completed | [`outputs/figures/02_numerical_histograms.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/02_numerical_histograms.png) |
| | 3. Boxplots for Numerical Features | Completed | [`outputs/figures/03_numerical_boxplots.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/03_numerical_boxplots.png) |
| | 4. Correlation Heatmap | Completed | [`outputs/figures/04_correlation_heatmap.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/04_correlation_heatmap.png) |
| | 5. Feature-vs-Target Comparison | Completed | [`outputs/figures/05_feature_vs_target.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/05_feature_vs_target.png) |
| | 6. Categorical Risk Factor Analysis | Completed | [`outputs/figures/06_categorical_risk_factors.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/06_categorical_risk_factors.png) |
| | 7. Pairplot / Scatterplot Analysis | Completed | [`outputs/figures/07_bmi_vs_weight_scatter.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/07_bmi_vs_weight_scatter.png) |
| | 8. Missing Value Visualization | Completed | [`outputs/figures/08_missing_values_matrix.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/08_missing_values_matrix.png) |
| | Exhaustive report with 5-point interpretation per chart | Completed | [`outputs/reports/visualization_interpretations.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/visualization_interpretations.md) |
| **Phase 6: Feature Analysis** | Mutual Information & Permutation Importance | Completed | [`src/feature_analysis.py`](file:///d:/heart-disease-risk-prediction/src/feature_analysis.py), [`outputs/metrics/feature_importance.csv`](file:///d:/heart-disease-risk-prediction/outputs/metrics/feature_importance.csv) |
| | Feature importance high-resolution figure | Completed | [`outputs/figures/09_feature_importance.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/09_feature_importance.png) |
| | Discussion on MLP interpretability & causation caveats | Completed | [`outputs/reports/feature_analysis_report.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/feature_analysis_report.md) |
| **Phase 7: Deep Learning Model** | Multi-Layer Perceptron (TensorFlow/Keras) | Completed | [`src/model_training.py`](file:///d:/heart-disease-risk-prediction/src/model_training.py), Dense(64)-Dropout-Dense(32)-Dropout-Sigmoid |
| | Imbalance handling with class weights | Completed | Balanced weights: $w_0 = 0.582, w_1 = 3.530$ |
| | Callbacks: EarlyStopping, ModelCheckpoint, ReduceLR | Completed | Implemented and monitored on validation AUC |
| **Phase 8: Hyperparameter Tuning** | Systematic grid across 10 configurations | Completed | [`src/hyperparameter_tuning.py`](file:///d:/heart-disease-risk-prediction/src/hyperparameter_tuning.py) |
| | Evaluation strictly on validation split | Completed | Untouched test set; results logged to [`outputs/metrics/hyperparameter_tuning_results.csv`](file:///d:/heart-disease-risk-prediction/outputs/metrics/hyperparameter_tuning_results.csv) |
| | Comparative visualization | Completed | [`outputs/figures/10_hyperparameter_tuning_comparison.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/10_hyperparameter_tuning_comparison.png) |
| | Optimal configuration saved | Completed | [`models/best_hyperparameters.json`](file:///d:/heart-disease-risk-prediction/models/best_hyperparameters.json) (Val AUC = 0.7973) |
| **Phase 9: Model Evaluation** | Evaluation on untouched test set ($N=150$) | Completed | [`src/model_evaluation.py`](file:///d:/heart-disease-risk-prediction/src/model_evaluation.py) |
| | Decision threshold calibration | Completed | Optimal threshold: $\theta = 0.480$ |
| | Metrics: Acc, Prec, Rec/Sens, Spec, F1, ROC-AUC | Completed | Acc: 74.00%, Rec: 52.38%, Spec: 77.52%, ROC-AUC: 0.7305 |
| | Figure 11: Training Loss vs Epoch | Completed | [`outputs/figures/11_training_loss_curves.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/11_training_loss_curves.png) |
| | Figure 12: Training Accuracy vs Epoch | Completed | [`outputs/figures/12_training_accuracy_curves.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/12_training_accuracy_curves.png) |
| | Figure 13: Test Confusion Matrix | Completed | [`outputs/figures/13_confusion_matrix.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/13_confusion_matrix.png) (TN=100, FP=29, FN=10, TP=11) |
| | Figure 14: Test ROC Curve & PR Curve | Completed | [`outputs/figures/14_roc_curve.png`](file:///d:/heart-disease-risk-prediction/outputs/figures/14_roc_curve.png) |
| | Metrics export in JSON and CSV | Completed | [`outputs/metrics/test_metrics.json`](file:///d:/heart-disease-risk-prediction/outputs/metrics/test_metrics.json), [`outputs/metrics/classification_report.csv`](file:///d:/heart-disease-risk-prediction/outputs/metrics/classification_report.csv) |
| **Phase 10: Saving & Reproducibility** | Serialization of final Keras model and metadata | Completed | [`models/final_model.keras`](file:///d:/heart-disease-risk-prediction/models/final_model.keras), [`models/model_metadata.json`](file:///d:/heart-disease-risk-prediction/models/model_metadata.json) |
| | Standalone prediction function with risk tiers | Completed | [`src/predict.py`](file:///d:/heart-disease-risk-prediction/src/predict.py) |
| **Phase 11: Web Application** | Streamlit interactive dashboard | Completed | [`app.py`](file:///d:/heart-disease-risk-prediction/app.py) |
| | Form inputs, probability estimation, disclaimer | Completed | Dynamic inputs, metrics cards, figure viewers, educational disclaimer |
| **Phase 12: Literature Review** | 5 verified academic references with DOIs | Completed | Documented in Section 7 of final report |
| **Phase 13: Final Report** | 21-section academic final project report | Completed | [`outputs/reports/final_project_report.md`](file:///d:/heart-disease-risk-prediction/outputs/reports/final_project_report.md) |
| **Phase 14 & 15: Execution & Verification** | Requirements file with tested versions | Completed | [`requirements.txt`](file:///d:/heart-disease-risk-prediction/requirements.txt) |
| | Comprehensive README.md with PowerShell run commands | Completed | [`README.md`](file:///d:/heart-disease-risk-prediction/README.md) |
| | End-to-end Jupyter Notebook | Completed | [`notebooks/heart_disease_eda.ipynb`](file:///d:/heart-disease-risk-prediction/notebooks/heart_disease_eda.ipynb) |
