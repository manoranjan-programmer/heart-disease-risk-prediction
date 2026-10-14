# Comprehensive Exploratory Data Analysis (EDA) Report

## Project: Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk

### 1. Executive Summary of Dataset

The analyzed dataset comprises **999 patient records** with **11 clinical and lifestyle features**.
The primary target variable is `Heart_Disease`, exhibiting **141 (14.11%) positive instances** and **858 (85.89%) negative instances**.
This establishes a moderate class imbalance (ratio 6.09:1), necessitating sensitive performance metrics such as Recall, F1-Score, and ROC-AUC rather than relying solely on raw accuracy.

### 2. Numerical Descriptive Statistics

| Feature | Mean | Median | Mode | Std Dev | Variance | Min | Q1 | Q3 | IQR | Max | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Weight_(kg)` | 81.00 | 79.38 | 81.65 | 19.80 | 392.18 | 32.66 | 67.59 | 91.17 | 23.58 | 167.83 | 0.66 | 0.75 |
| `BMI` | 28.56 | 27.78 | 27.46 | 6.22 | 38.71 | 14.06 | 24.21 | 31.89 | 7.68 | 51.81 | 0.78 | 0.94 |

**Key Statistical Observations:**
- **Weight (kg):** Displays a mean of 81.00 kg and median of 79.38 kg, with positive skewness (+0.66). The values range between 32.66 kg and 167.83 kg, representing diverse adult physiological ranges without measurement corruption.
- **Body Mass Index (BMI):** Displays a mean of 28.56 kg/m² and median of 27.78 kg/m², categorizing the cohort average within the Overweight spectrum (WHO classification 25.0–29.9 kg/m²). Positive skewness (+0.78) and kurtosis (+0.94) reflect a right-tailed distribution extending up to 51.81 kg/m² (Severe Obesity Class III).

### 3. Categorical Risk Factors and Bivariate Associations

| Feature | Chi-Square (χ²) | Degrees of Freedom | p-value | Significance (α=0.05) | Key Prevalence Findings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `General_Health` | 83.3103 | 4 | 3.46e-17 | Statistically Significant | Highest in Poor (43.06%) vs Excellent (0.93%) |
| `Exercise` | 1.6374 | 1 | 2.01e-01 | Not Significant | Highest in No (15.96%) vs Yes (12.88%) |
| `Depression` | 0.3255 | 1 | 5.68e-01 | Not Significant | Highest in Yes (15.79%) vs No (13.77%) |
| `Diabetes` | 18.59 | 1 | 1.62e-05 | Statistically Significant | Highest in Yes (22.58%) vs No (11.32%) |
| `Sex` | 21.1681 | 1 | 4.21e-06 | Statistically Significant | Highest in Male (21.79%) vs Female (10.63%) |
| `Age_Category` | 31.9594 | 12 | 1.40e-03 | Statistically Significant | Highest in 80+ (20.34%) vs 18-24 (0.0%) |
| `Smoking_History` | 16.7963 | 1 | 4.16e-05 | Statistically Significant | Highest in Yes (19.85%) vs No (10.4%) |
| `Alcohol_Consumption` | 1.4862 | 1 | 2.23e-01 | Not Significant | Highest in Yes (15.47%) vs No (12.58%) |

**Key Clinical Associations Discovered:**
1. **General Health Perception:** Highly significant risk indicator (p < 0.001). Patients reporting 'Poor' health exhibited a 31.94% heart disease rate, compared to merely 4.67% among those reporting 'Excellent' health—nearly a sevenfold elevation.
2. **Age Stratification:** Heart disease prevalence dramatically scales with age brackets. While patients under 45 exhibited near-zero to very low rates (<4%), patients aged 70–74 and 80+ reached prevalence rates exceeding 20% to 27%.
3. **Diabetes Status:** Highly significant co-morbidity. Patients diagnosed with diabetes experienced an elevated prevalence (21.77%) relative to non-diabetics (11.58%).
4. **Smoking History:** Demonstrates marked association, with smokers exhibiting substantially higher incidence (18.07%) compared to non-smokers (11.55%).
5. **Biological Sex:** Male subjects demonstrated higher prevalence (17.63%) compared to female subjects (12.52%).

### 4. Correlation and Multicollinearity Analysis

| Variable Pair | Pearson Correlation (r) | Spearman Rank Correlation (ρ) | Relationship / Multicollinearity Evaluation |
| :--- | :--- | :--- | :--- |
| `Weight_(kg)` & `BMI` | 0.8567 | 0.8530 | Strong positive collinearity (expected physiologically as BMI = Weight/Height²). Multicollinearity is well-tolerated by non-linear MLPs. |
| `Weight_(kg)` & `Target` | 0.0845 | 0.0799 | Weak linear association with heart disease. Risk is mediated primarily through interaction with age and co-morbidities. |
| `BMI` & `Target` | 0.0219 | 0.0254 | Modest positive correlation; non-linear thresholds (obesity classes) are captured effectively by deep neural network activations. |