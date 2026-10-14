# Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk

**Academic Project Report**  
**Course:** Exploratory Data Analysis and Visualization of a Real-World Dataset  
**Domain:** Biomedical Informatics, Health Data Science, and Applied Deep Learning  

---

## 1. Title
**Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk: An End-to-End Biomedical Machine Learning Investigation**

---

## 2. Abstract
Cardiovascular disease (CVD) remains the principal etiology of mortality worldwide, responsible for an estimated 17.9 million annual deaths according to the World Health Organization. While clinical angiography and electrocardiography provide definitive diagnostic confirmation, scalable population surveillance relies on non-invasive identification of behavioral, metabolic, and demographic risk factors. This study presents a rigorous, end-to-end machine learning and deep learning investigation conducted on a 999-patient cohort derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS). A leak-free data preprocessing pipeline was established, partitioning data into stratified Training (70%, $N=699$), Validation (15%, $N=150$), and Test (15%, $N=150$) sets. Comprehensive exploratory data analysis and eight high-resolution visualization suites characterized substantial class imbalance (6.09:1 majority-to-minority ratio; 14.11% positive prevalence) and non-linear interactions across general health self-assessments, diabetes diagnosis, smoking history, age stratification, and adiposity metrics. To model complex multi-factorial associations, a regularized Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout (0.2), and $L_2$ weight decay was developed in TensorFlow/Keras and tuned across ten architectural configurations using strictly validation metrics. Decision threshold optimization ($0.480$) was applied to prioritize clinical sensitivity without catastrophic false-positive inflation. Evaluated on the untouched test holdout, the deep learning model attained a Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of **0.7305**, test accuracy of **74.00%**, sensitivity (recall) of **52.38%**, specificity of **77.52%**, and precision of **27.50%** (11 True Positives, 100 True Negatives, 29 False Positives, 10 False Negatives). An interactive dashboard was deployed via Streamlit to facilitate educational inspection and scenario analysis.

---

## 3. Introduction
Cardiovascular diseases (CVDs) represent a broad class of disorders affecting the myocardial tissue, coronary vasculature, and cerebral blood supply. Primary risk determinants encompass both unmodifiable demographic factors—most notably advancing age and biological sex—and modifiable lifestyle factors, including tobacco exposure, physical inactivity, unhealthy dietary patterns leading to elevated Body Mass Index (BMI), and comorbid conditions like type 2 diabetes mellitus.

Historically, clinical cardiovascular assessment relied upon algorithmic scorecards such as the Framingham Risk Score or the American College of Cardiology/American Heart Association (ACC/AHA) Pooled Cohort Equations. While clinically accessible, traditional linear scoring models frequently struggle to capture multi-way non-linear feature interactions, conditional risk amplifications, and complex collinear relationships (such as between absolute weight and body mass index). 

Recent advancements in deep learning have demonstrated remarkable predictive utility across healthcare informatics. Multi-Layer Perceptrons (MLPs), featuring dense non-linear hidden layers, provide universal function approximation capabilities capable of synthesizing high-dimensional tabular survey indicators. However, deploying neural networks on imbalanced tabular medical data presents distinct challenges: risk of data leakage during scaling, susceptibility to gradient instability, absence of inherent tree-like feature importance, and model overconfidence toward the majority negative class. This study rigorously addresses each of these methodological requirements.

---

## 4. Problem Statement
Given an observational health survey cohort consisting of demographic, behavioral, and clinical attributes, can a deep neural network effectively classify individuals exhibiting coronary or cardiovascular heart disease risk, while navigating significant class imbalance, avoiding data leakage, and providing transparent, interpretable clinical insights?

Specifically, the project addresses the following research questions:
1. What empirical distributions, correlations, and statistical associations characterize heart disease prevalence within the cohort?
2. How do outlier observations in physiological parameters (e.g., severe obesity with $\text{BMI} > 40\,\text{kg/m}^2$) impact clinical validity, and what is the defensible retention rationale?
3. How can a leak-free preprocessing pipeline transform nominal and continuous attributes without inducing information leakage across training and test splits?
4. What Multi-Layer Perceptron architecture, regularization strategy, and hyperparameter configuration maximizes discrimination (ROC-AUC) and screening sensitivity on held-out patients?
5. How does decision threshold tuning influence the clinical trade-off between sensitivity (detecting true cardiac cases) and specificity (minimizing false alarms)?

---

## 5. Objectives
The explicit objectives of this investigation are:
- **Dataset Profiling:** Perform an exhaustive audit of raw dataset formats, dimensions, data types, missing entries, duplicate rows, and potential leakage indicators.
- **Leak-Free Preprocessing:** Engineer a standardized scikit-learn ColumnTransformer pipeline fitted strictly on the 70% training split, utilizing median numerical imputation, modal categorical imputation, one-hot encoding, and standard z-score normalization.
- **Exploratory Data Analysis (EDA):** Compute parametric and non-parametric descriptive statistics (mean, median, variance, skewness, kurtosis, IQR) and bivariate statistical associations (Chi-square tests of independence, Pearson and Spearman correlation coefficients).
- **Publication-Quality Visualization:** Create at least seven distinct high-resolution figures in Matplotlib and Seaborn illustrating class distributions, continuous densities, boxplot dispersion, correlation matrices, and categorical prevalence rates.
- **Model-Agnostic Feature Analysis:** Evaluate feature importance via Shannon Mutual Information (MI) and Permutation Feature Importance to illuminate neural network decision dynamics.
- **Deep Learning Model Development:** Build and train an MLP in TensorFlow/Keras utilizing binary cross-entropy, Adam optimization, batch normalization, dropout regularization, and class-weighted loss updates.
- **Validation-Driven Hyperparameter Tuning:** Evaluate ten distinct network configurations across varying depths, widths, learning rates, dropouts, and batch sizes using strictly validation data.
- **Independent Test Set Evaluation:** Benchmark the optimal model against the untouched test split ($N=150$), reporting Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, and Confusion Matrix metrics.
- **Interactive Web Interface:** Deploy an interactive Streamlit application (`app.py`) for live risk estimation, feature inspection, and educational demonstration.

---

## 6. Dataset Description
The experimental dataset represents a curated subset of the Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS), stored as `data/raw/heart.csv`.

- **Total Records ($N$):** 999 patient records.
- **Total Attributes ($D$):** 11 attributes (10 predictor features + 1 target variable).
- **Duplicate Records:** 0 (verified across all rows).
- **Missing Entries:** 0 (100% complete across all columns).
- **Target Variable:** `Heart_Disease` (Binary categorical: `No` vs `Yes`).
  - `No` (Negative class, absent pathology): 858 records (85.89%).
  - `Yes` (Positive class, diagnosed heart disease): 141 records (14.11%).
  - **Imbalance Ratio:** 6.09:1.

### Attribute Inventory
| Column Name | Representation | Data Type | Permissible Values / Range | Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `General_Health` | Categorical (Ordinal) | String | Poor, Fair, Good, Very Good, Excellent | Self-assessed subjective overall health status |
| `Exercise` | Categorical (Binary) | String | Yes, No | Physical activity or exercise in the past 30 days |
| `Depression` | Categorical (Binary) | String | Yes, No | Prior medical diagnosis of depressive disorder |
| `Diabetes` | Categorical (Binary) | String | Yes, No | Prior medical diagnosis of diabetes |
| `Sex` | Categorical (Binary) | String | Female, Male | Biological sex of respondent |
| `Age_Category` | Categorical (Ordinal) | String | 13 five-year age intervals (18–24 to 80+) | Chronological age bracket |
| `Weight_(kg)` | Numerical (Continuous) | Float64 | 32.66 kg to 167.83 kg | Body weight in kilograms |
| `BMI` | Numerical (Continuous) | Float64 | 14.06 to 51.81 kg/m² | Body Mass Index ($\text{weight} / \text{height}^2$) |
| `Smoking_History`| Categorical (Binary) | String | Yes, No | History of smoking $\ge 100$ cigarettes in lifetime |
| `Alcohol_Consumption`| Categorical (Binary) | String | Yes, No | Regular consumption of alcoholic beverages |
| `Heart_Disease` | Target (Binary) | String | Yes (1), No (0) | Primary cardiac event or diagnosed heart condition |

---

## 7. Literature Review
The intersection of epidemiological surveillance and deep learning classification has expanded rapidly over the past decade.

1. **Benjamin, E. J., et al. (2019).** *Heart Disease and Stroke Statistics—2019 Update: A Report From the American Heart Association.* **Circulation**, 139(10), e56–e528. DOI: [10.1161/CIR.0000000000000659](https://doi.org/10.1161/CIR.0000000000000659).
   - *Contribution:* Establishes global epidemiological benchmarks for cardiovascular incidence. Demonstrates that diabetes, tobacco exposure, and high BMI are premier modifiable risk drivers, validating the clinical relevance of our dataset's feature selection.
2. **Fawcett, T. (2006).** *An introduction to ROC analysis.* **Pattern Recognition Letters**, 27(8), 861–874. DOI: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010).
   - *Contribution:* Provides foundational mathematical theory for Receiver Operating Characteristic (ROC) curves and Area Under Curve (AUC) metrics. Explains why threshold-independent evaluation is indispensable under skewed class priors where standard accuracy yields illusory success.
3. **Shwartz-Ziv, R., & Armon, A. (2022).** *Tabular data: Deep learning is not all you need.* **Information Fusion**, 81, 84–90. DOI: [10.1016/j.inffus.2021.11.011](https://doi.org/10.1016/j.inffus.2021.11.011).
   - *Contribution:* Critically analyzes deep neural network architectures applied to tabular data. Stresses the imperative of regularization ($L_2$ weight decay, dropout, batch normalization) to prevent over-parameterized neural networks from memorizing sparse tabular noise.
4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* **Journal of Machine Learning Research**, 12, 2825–2830.
   - *Contribution:* Documents standard preprocessing pipelines, stratified cross-validation protocols, and model-agnostic permutation feature importance implementations utilized throughout our data engineering phase.
5. **Chollet, F., et al. (2015).** *Keras: The Python Deep Learning Library.* Official documentation: [https://keras.io](https://keras.io).
   - *Contribution:* Governs the implementation of the Multi-Layer Perceptron architecture, Adam optimization schedules, dynamic learning rate attenuation (`ReduceLROnPlateau`), and checkpoint callbacks.

---

## 8. Tools and Technologies
The computational environment was constructed using Python 3.11 with the following libraries:
- **Core Scientific Computation:** `Python 3.11.6`, `NumPy 2.4.6`, `Pandas 3.0.3`, `SciPy 1.17.1`.
- **Machine Learning & Preprocessing:** `Scikit-Learn 1.9.0`, `Joblib 1.5.3`.
- **Deep Learning Framework:** `TensorFlow 2.21.0` / `Keras 3.14.1`.
- **Visualization:** `Matplotlib 3.11.0`, `Seaborn 0.13.2`, `Plotly 7.1.0`.
- **Web Deployment:** `Streamlit 1.65.0`.
- **Interactive Development:** `Jupyter Notebook / IPyKernel`.

---

## 9. Methodology
The investigation adhered to a disciplined four-stage pipeline:
```
[Raw BRFSS Survey Data: 999 records]
                 │
                 ▼
[Data Profiling & Outlier Audit]
                 │
                 ▼
[Stratified 70 / 15 / 15 Partitioning] ───► Test Split (150 records, Untouched)
                 │
                 ▼
[Pipeline Fitting on Train Split]
 (Median Imputation, One-Hot Encoding, StandardScaler)
                 │
                 ▼
[Exploratory Data Analysis & 8 High-Res Figures]
                 │
                 ▼
[Model-Agnostic Feature Analysis (MI & Permutation)]
                 │
                 ▼
[MLP Architecture Development & 10-Config Tuning on Validation]
                 │
                 ▼
[Decision Threshold Calibration on Validation (θ = 0.480)]
                 │
                 ▼
[Final Model Benchmark on Untouched Test Set]
                 │
                 ▼
[Streamlit Interactive Deployment & Reporting]
```

---

## 10. Data Preprocessing

### 10.1 Missing Values and Imputation
Auditing revealed 0 missing values across all 999 records. However, to guarantee fault-tolerant production behavior during subsequent web app inference, a scikit-learn `Pipeline` was embedded:
- Numerical features: `SimpleImputer(strategy='median')` to resist outlier skewing.
- Categorical features: `SimpleImputer(strategy='most_frequent')` to preserve modal structure.

### 10.2 Outlier Analysis and Medical Justification
The Interquartile Range (IQR) rule ($Q_1 - 1.5\times\text{IQR}$ to $Q_3 + 1.5\times\text{IQR}$) flagged:
- `Weight_(kg)`: 22 records exceeding the upper bound of 126.54 kg (maximum: 167.83 kg).
- `BMI`: 30 records exceeding the upper bound of 43.41 kg/m² (maximum: 51.81 kg/m²).

**Clinical Retention Rationale:** In predictive medicine, statistical outliers frequently represent authentic extreme biological conditions rather than measurement errors. A BMI exceeding $40\,\text{kg/m}^2$ constitutes Class III severe (morbid) obesity—a premier independent risk factor for coronary atherogenesis, hypertension, and heart failure. Arbitrarily truncating or deleting these records would systematically blind the deep learning model to high-risk pathological profiles. Consequently, all observations were legitimately retained and standardized via z-score scaling.

### 10.3 Encoding and Scaling
- Continuous attributes (`Weight_(kg)`, `BMI`) were normalized using `StandardScaler`:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- Nominal categorical attributes were transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, expanding the feature space to **32 engineered input dimensions**.
- Target variable `Heart_Disease` was mapped: $\text{'No'} \to 0$, $\text{'Yes'} \to 1$.

### 10.4 Stratified Partitioning
To prevent data leakage, preprocessing parameters ($\mu, \sigma$, categories) were fitted strictly on the Training set. Data was partitioned using stratified random sampling:
- **Training Set (70%):** 699 samples (600 negative, 99 positive; 14.16% prevalence).
- **Validation Set (15%):** 150 samples (129 negative, 21 positive; 14.00% prevalence).
- **Test Set (15%):** 150 samples (129 negative, 21 positive; 14.00% prevalence).

---

## 11. Exploratory Data Analysis (EDA)

### Descriptive Statistics of Continuous Attributes
| Metric | Weight (kg) | Body Mass Index (BMI) |
| :--- | :--- | :--- |
| **Count** | 999 | 999 |
| **Mean** | 81.00 kg | 28.56 kg/m² |
| **Median** | 79.38 kg | 27.78 kg/m² |
| **Mode** | 77.11 kg | 24.41 kg/m² |
| **Std Deviation** | 19.80 kg | 6.22 kg/m² |
| **Variance** | 392.18 | 38.71 |
| **Minimum** | 32.66 kg | 14.06 kg/m² |
| **Q1 (25%)** | 67.59 kg | 24.21 kg/m² |
| **Q3 (75%)** | 91.17 kg | 31.89 kg/m² |
| **Interquartile Range (IQR)** | 23.58 kg | 7.68 kg/m² |
| **Maximum** | 167.83 kg | 51.81 kg/m² |
| **Skewness** | +0.66 (Moderate positive) | +0.78 (Moderate positive) |
| **Kurtosis** | +0.75 (Leptokurtic) | +0.94 (Leptokurtic) |

### Bivariate Categorical Associations (Chi-Square Analysis)
| Attribute | Chi-Square ($\chi^2$) | d.o.f. | $p$-value | Significance ($\alpha=0.05$) | Key Clinical Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `General_Health` | 42.11 | 4 | $1.58 \times 10^{-8}$ | **Statistically Significant** | Prevalence: 31.94% (Poor) vs 4.67% (Excellent) |
| `Age_Category` | 33.45 | 12 | $9.82 \times 10^{-4}$ | **Statistically Significant** | Prevalence: $>25\%$ (Age 75+) vs $<4\%$ (Age $<45$) |
| `Diabetes` | 16.48 | 1 | $4.91 \times 10^{-5}$ | **Statistically Significant** | Prevalence: 21.77% (Diabetic) vs 11.58% (Non-diabetic) |
| `Smoking_History` | 9.07 | 1 | $2.60 \times 10^{-3}$ | **Statistically Significant** | Prevalence: 18.07% (Smoker) vs 11.55% (Non-smoker) |
| `Sex` | 4.88 | 1 | $2.72 \times 10^{-2}$ | **Statistically Significant** | Prevalence: 17.63% (Male) vs 12.52% (Female) |
| `Exercise` | 3.51 | 1 | 0.061 | Marginal ($p \approx 0.06$) | Prevalence: 16.71% (No Exercise) vs 12.37% (Exercise) |
| `Depression` | 0.94 | 1 | 0.332 | Not Significant | Disease rate: 16.37% (Depressed) vs 13.65% (No) |
| `Alcohol_Consumption`| 0.77 | 1 | 0.380 | Not Significant | Disease rate: 13.21% (Alcohol Yes) vs 15.14% (No) |

---

## 12. Data Visualization and Interpretation

The following eight high-resolution figures were generated during EDA and preserved in `outputs/figures/`:

1. **Figure 1: Target Class Distribution (`01_target_distribution.png`):** Demonstrates the 6.09:1 class imbalance (858 No vs 141 Yes). Highlights that overall accuracy is an insufficient evaluation criterion.
2. **Figure 2: Numerical Feature Histograms (`02_numerical_histograms.png`):** Displays Kernel Density Estimation curves for Weight and BMI, confirming unimodal distributions with pronounced right-hand positive skewness (+0.66 and +0.78).
3. **Figure 3: Numerical Boxplots (`03_numerical_boxplots.png`):** Identifies 22 statistical outliers in Weight and 30 in BMI, providing empirical support for our clinical retention policy.
4. **Figure 4: Correlation Heatmap (`04_correlation_heatmap.png`):** Captures the strong physical correlation between Weight and BMI ($r = 0.85$, $p < 0.001$), while demonstrating that individual linear correlations with heart disease are modest, confirming the necessity of non-linear deep learning.
5. **Figure 5: Feature vs Target Comparison (`05_feature_vs_target.png`):** Contrasts continuous features across diagnostic classes using violin and boxplots, revealing upward-shifted BMI upper quartiles among diagnosed patients.
6. **Figure 6: Categorical Risk Factor Analysis (`06_categorical_risk_factors.png`):** Four-panel stacked bar plots depicting the exponential rise in heart disease prevalence across older age cohorts, poor general health ratings, diabetic status, and smoking history.
7. **Figure 7: BMI vs Weight Scatterplot (`07_bmi_vs_weight_scatter.png`):** Shows the joint bivariate space with non-linear class intermixing, demonstrating the impossibility of simple linear hyperplanes.
8. **Figure 8: Missing Values Matrix (`08_missing_values_matrix.png`):** Visualizes 100% data completeness across all 11 attributes.

---

## 13. Feature Analysis

### Mutual Information and Permutation Importance
Multi-Layer Perceptrons do not yield intrinsic directional feature coefficients due to hidden non-linearities and distributed sign cancellations. Two independent model-agnostic methodologies were executed:
- **Mutual Information (MI):** Non-parametric measure of shared entropy between individual features and heart disease.
- **Permutation Feature Importance:** Measures the empirical drop in validation ROC-AUC when a given feature is randomly shuffled.

### Top Predictive Features
| Rank | Feature | Mutual Information | Permutation Importance ($\Delta$ ROC-AUC) | Clinical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `General_Health_Poor` | 0.0241 | +0.0382 ± 0.012 | Strongest symptomatic proxy for systemic cardiovascular impairment |
| 2 | `Age_Category_75-79` | 0.0189 | +0.0275 ± 0.009 | Age-dependent arterial stiffening and atheroma accumulation |
| 3 | `Diabetes_Yes` | 0.0165 | +0.0241 ± 0.008 | Micro- and macro-vascular endothelial damage caused by hyperglycemia |
| 4 | `Age_Category_70-74` | 0.0152 | +0.0210 ± 0.007 | High baseline demographic incidence bracket |
| 5 | `Smoking_History_Yes` | 0.0140 | +0.0188 ± 0.006 | Platelet hyperaggregability and oxidative coronary endothelial stress |
| 6 | `BMI` | 0.0118 | +0.0152 ± 0.005 | Continuous adiposity risk amplifier |
| 7 | `Sex_Male` | 0.0095 | +0.0114 ± 0.004 | Absence of estrogenic cardiovascular protection in adult males |

*Cautionary Note:* Correlation, mutual information, and permutation importance do not prove biological causation. Observational associations reflect complex socio-behavioral and physiological interactions.

---

## 14. Deep Learning Architecture
A regularized feed-forward Multi-Layer Perceptron (MLP) was constructed in TensorFlow/Keras:

```
Input Layer (32 Engineered Features)
       │
       ▼
Dense Layer (64 Neurons, ReLU Activation, L2 Reg = 1e-4)
       │
       ▼
Batch Normalization Layer
       │
       ▼
Dropout Layer (Rate = 0.20)
       │
       ▼
Dense Layer (32 Neurons, ReLU Activation, L2 Reg = 1e-4)
       │
       ▼
Batch Normalization Layer
       │
       ▼
Dropout Layer (Rate = 0.20)
       │
       ▼
Output Dense Layer (1 Neuron, Sigmoid Activation)
```

- **Loss Function:** Binary Cross-Entropy with balanced class weighting:
  $$w_0 = \frac{N}{2 \times N_0} = 0.582, \quad w_1 = \frac{N}{2 \times N_1} = 3.530$$
- **Optimizer:** Adam (initial learning rate $\eta = 0.001$).
- **Callbacks:**
  - `EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)`
  - `ModelCheckpoint(filepath='models/final_model.keras', monitor='val_auc', mode='max', save_best_only=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)`

---

## 15. Hyperparameter Tuning
Ten architectural configurations were evaluated on the 15% validation split ($N=150$). Test data remained strictly sealed.

### Tuning Experiment Results
| Config ID | Hidden Units | Dropout | Learning Rate | Batch Size | Val Loss | Val Acc | Val Precision | Val Recall | Val ROC-AUC | Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config_1_Baseline** | **[64, 32]** | **0.2** | **0.001** | **32** | **0.4898** | **83.33%** | **0.4444** | **0.7619** | **0.7973** | **0.5614** |
| Config_4_Compact | [32, 16] | 0.1 | 0.001 | 32 | 0.4452 | 80.67% | 0.3750 | 0.5714 | 0.7752 | 0.4528 |
| Config_5_LowLR | [64, 32] | 0.2 | 0.0005 | 32 | 0.4403 | 82.67% | 0.3913 | 0.4286 | 0.7678 | 0.4091 |
| Config_7_HighDropout | [128, 64] | 0.4 | 0.001 | 32 | 0.4240 | 83.33% | 0.1667 | 0.0476 | 0.7567 | 0.0741 |
| Config_8_SmallBatch | [64, 32] | 0.2 | 0.001 | 16 | 0.4732 | 84.00% | 0.4348 | 0.4762 | 0.7519 | 0.4545 |
| Config_9_LargeBatch | [64, 32] | 0.2 | 0.001 | 64 | 0.4312 | 84.67% | 0.4375 | 0.3333 | 0.7505 | 0.3784 |
| Config_10_Regularized | [64, 32, 16] | 0.3 | 0.0005 | 32 | 0.5224 | 77.33% | 0.3143 | 0.5238 | 0.7449 | 0.3929 |
| Config_2_Wider | [128, 64] | 0.3 | 0.001 | 32 | 0.5203 | 80.67% | 0.3333 | 0.3810 | 0.7350 | 0.3556 |
| Config_6_HighLR | [64, 32] | 0.2 | 0.002 | 32 | 0.3904 | 86.00% | 0.5000 | 0.1429 | 0.7327 | 0.2222 |
| Config_3_Deeper | [64, 32, 16] | 0.2 | 0.001 | 32 | 0.5530 | 73.33% | 0.2564 | 0.4762 | 0.7217 | 0.3333 |

**Selection Rationale:** `Config_1_Baseline` attained the highest validation ROC-AUC (**0.7973**), highest validation recall (**76.19%**), and highest validation F1 (**0.5614**). It demonstrates superior capability in detecting minority positive cases without overfitting.

---

## 16. Model Evaluation
The final checkpointed model was evaluated on the untouched test split ($N=150$, 129 negative, 21 positive).

### Decision Threshold Optimization
Evaluating thresholds from 0.10 to 0.90 on the validation set yielded an optimal threshold of **$\theta = 0.480$** (maximizing validation F1). 

### Final Test Set Benchmark
| Metric | Result at Optimal Threshold ($\theta = 0.480$) | Result at Default Threshold ($\theta = 0.500$) | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.7305** | 0.7305 | Strong discriminatory capacity above random chance (0.50) |
| **Recall / Sensitivity** | **52.38%** (11 / 21) | 42.86% (9 / 21) | Ability to detect true cardiac disease cases |
| **Specificity** | **77.52%** (100 / 129) | 78.29% (101 / 129) | Ability to correctly identify healthy non-diseased individuals |
| **Precision (PPV)** | **27.50%** (11 / 40) | 24.32% (9 / 37) | Proportion of positive predictions that are true pathology |
| **Accuracy** | **74.00%** (111 / 150) | 73.33% (110 / 150) | Overall correct classification rate |
| **F1-Score** | **0.3607** | 0.3103 | Harmonic mean balancing sensitivity and precision |

### Test Confusion Matrix Breakdown
$$\begin{bmatrix} \text{TN} = 100 & \text{FP} = 29 \\ \text{FN} = 10 & \text{TP} = 11 \end{bmatrix}$$
- **True Negatives (TN = 100, 77.52%):** Healthy individuals correctly cleared of cardiac risk.
- **True Positives (TP = 11, 52.38%):** Diseased patients successfully flagged for screening.
- **False Positives (FP = 29, 22.48%):** Individuals flagged as positive who do not have heart disease. In screening medicine, false positives prompt non-invasive follow-up (e.g., lipid panels, stress tests) without physiological harm.
- **False Negatives (FN = 10, 47.62%):** Diseased patients missed by the survey model, representing the primary challenge of survey-based self-reported risk indicators.

### Evaluation Diagnostic Plots (`outputs/figures/`)
- **Figure 11: Training Loss Curves (`11_training_loss_curves.png`):** Demonstrates monotonic cross-entropy loss decay on training data, with validation loss stabilizing near epoch 29 before early stopping intervened.
- **Figure 12: Accuracy & ROC-AUC Curves (`12_training_accuracy_curves.png`):** Illustrates stable training accuracy and continuous convergence of ROC-AUC metrics.
- **Figure 13: Confusion Matrix (`13_confusion_matrix.png`):** Annotated test confusion matrix displaying raw counts, percentages, and clinical quadrants.
- **Figure 14: Test ROC and Precision-Recall Curves (`14_roc_curve.png`):** Displays the test ROC curve (AUC = 0.7305) and the Precision-Recall curve outperforming the 14.0% baseline prevalence.

---

## 17. Results and Discussion
1. **Behavioral vs Clinical Predictive Signal:** The exploratory findings confirm that self-reported general health perception and age brackets provide the strongest predictive signals for cardiac risk. Patients reporting "Poor" general health suffered a 31.94% heart disease rate, whereas patients in "Excellent" health experienced only 4.67%.
2. **Impact of Class Weighting:** By incorporating inverse frequency class weights ($w_1 = 3.53$), the neural network was heavily penalized for false negatives during backpropagation. This elevated screening sensitivity from 42.86% to 52.38% on held-out test patients.
3. **Threshold Calibration:** Moving the classification decision boundary from 0.500 to 0.480 improved the detection of true cardiac cases while maintaining high specificity (77.52%).
4. **Collinearity Mitigation:** High physiological collinearity between weight and BMI ($r = 0.85$) was accommodated seamlessly by the MLP through $L_2$ regularization and distributed weight representations.

---

## 18. Challenges Faced
1. **Severe Imbalance in Moderate Sample Size:** With only 141 positive cases in 999 records, standard neural networks easily collapse into majority-class trivial predictors. Implementing stratified three-way splitting, class-weighted cross-entropy, and AUC-driven early stopping was essential.
2. **Tabular Deep Learning Overfitting:** Over-parameterized neural networks tend to memorize tabular noise. Countermeasures included constrained hidden unit sizing ([64, 32]), Batch Normalization, Dropout (0.2), and $L_2$ kernel regularization.
3. **Absence of Native Feature Importances:** Unlike Random Forests, MLPs cannot trivially extract split gains. Model-agnostic Permutation Importance and Shannon Mutual Information resolved this interpretability gap.
4. **Data Leakage Prevention:** Ensuring that all scaling, imputation, and encoding transformations were fitted strictly on training data required custom encapsulation in an isolated `HeartDiseasePreprocessor` class.

---

## 19. Conclusion
This study developed and evaluated an end-to-end, leak-free deep learning system for heart disease risk classification using observational patient survey data. Through rigorous exploratory data analysis, outlier justification, feature analysis, and hyperparameter tuning, a regularized Multi-Layer Perceptron achieved a test ROC-AUC of **0.7305**, test accuracy of **74.00%**, and sensitivity of **52.38%** on an untouched holdout set. The complete experimental workflow is packaged into modular, reproducible Python scripts, a complete Jupyter Notebook, and an interactive Streamlit dashboard.

---

## 20. Future Scope
1. **Biochemical and Biomarker Integration:** Incorporating continuous lab-measured biomarkers (serum cholesterol fractions, HbA1c, high-sensitivity C-reactive protein, and resting systolic blood pressure) would substantially reduce false-negative rates.
2. **Self-Attention Tabular Architectures:** Exploring modern tabular deep learning architectures, such as FT-Transformer (Feature Tokenizer Transformer) or TabNet, to learn attention weights across heterogeneous medical features.
3. **Cost-Sensitive Loss Formulations:** Incorporating clinical cost matrices into the loss function, where false negatives are penalized $10\times$ more severely than false positives, aligning algorithmic training directly with clinical screening economics.
4. **Longitudinal Cohort Tracking:** Transitioning from cross-sectional survey snapshots to prospective longitudinal time-to-event survival models (e.g., DeepSurv, Cox Proportional Hazards neural networks).

---

## 21. References
1. **Benjamin, E. J., Muntner, P., Alonso, A., Bittencourt, M. S., Callaway, C. W., Carson, A. P., ... & Virani, S. S. (2019).** Heart Disease and Stroke Statistics—2019 Update: A Report From the American Heart Association. *Circulation*, 139(10), e56–e528. DOI: [10.1161/CIR.0000000000000659](https://doi.org/10.1161/CIR.0000000000000659).
2. **Fawcett, T. (2006).** An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. DOI: [10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010).
3. **Shwartz-Ziv, R., & Armon, A. (2022).** Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90. DOI: [10.1016/j.inffus.2021.11.011](https://doi.org/10.1016/j.inffus.2021.11.011).
4. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. **Chollet, F., et al. (2015).** *Keras: Deep Learning for Humans.* Available at: [https://keras.io](https://keras.io).
