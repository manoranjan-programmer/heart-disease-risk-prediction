# Feature Analysis and Predictive Importance Report

## Project: Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk

### 1. Architectural Context: Feature Interpretability in Deep Learning

Unlike tree-based ensembles (which expose Gini impurity decrease or split frequencies) or generalized linear models (which provide explicit linear coefficients), **Multi-Layer Perceptrons (MLPs) do not inherently yield straightforward directional feature importances**.
In an MLP, input features undergo multiple affine matrix multiplications interspersed with non-linear activation functions (ReLU, Sigmoid). Features become deeply entangled in dense hidden representations, meaning a single feature's influence is mediated across dozens of distributed weight paths with alternating positive and negative signs.
To achieve rigorous, scientifically defensible interpretability, this study employs two independent, model-agnostic methodologies:
1. **Information-Theoretic Mutual Information (MI):** Measures non-parametric dependency and shared entropy between each feature and the cardiac diagnosis.
2. **Permutation Feature Importance:** Measures the empirical degradation in validation ROC-AUC when a given feature is randomly shuffled, breaking its relationship with the target.

### 2. Top Informative Features Ranked by Mutual Information

| Rank | Feature Name | Mutual Information Score | Domain Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `General_Health_Poor` | 0.0301 | Substantial non-linear dependency with cardiac outcomes |
| 2 | `General_Health_Excellent` | 0.0268 | Substantial non-linear dependency with cardiac outcomes |
| 3 | `Weight_(kg)` | 0.0252 | Substantial non-linear dependency with cardiac outcomes |
| 4 | `General_Health_Very Good` | 0.0172 | Substantial non-linear dependency with cardiac outcomes |
| 5 | `Age_Category_45-49` | 0.0164 | Substantial non-linear dependency with cardiac outcomes |
| 6 | `Sex_Female` | 0.0151 | Substantial non-linear dependency with cardiac outcomes |
| 7 | `Age_Category_70-74` | 0.0129 | Substantial non-linear dependency with cardiac outcomes |
| 8 | `Sex_Male` | 0.0118 | Substantial non-linear dependency with cardiac outcomes |
| 9 | `Diabetes_No` | 0.0111 | Substantial non-linear dependency with cardiac outcomes |
| 10 | `Diabetes_Yes` | 0.0109 | Substantial non-linear dependency with cardiac outcomes |

### 3. Top Informative Features Ranked by Permutation Importance

| Rank | Feature Name | Mean ROC-AUC Decrease | Standard Error (±) |
| :--- | :--- | :--- | :--- |
| 1 | `General_Health_Poor` | 0.0534 | ±0.0184 |
| 2 | `General_Health_Excellent` | 0.0316 | ±0.0261 |
| 3 | `General_Health_Fair` | 0.0158 | ±0.0094 |
| 4 | `Age_Category_60-64` | 0.0069 | ±0.0057 |
| 5 | `General_Health_Good` | 0.0062 | ±0.0040 |
| 6 | `Diabetes_Yes` | 0.0060 | ±0.0042 |
| 7 | `Exercise_No` | 0.0053 | ±0.0034 |
| 8 | `Age_Category_45-49` | 0.0045 | ±0.0044 |
| 9 | `BMI` | 0.0040 | ±0.0078 |
| 10 | `Diabetes_No` | 0.0037 | ±0.0047 |

### 4. Multicollinearity and Feature Redundancy

- **Weight and BMI Collinearity:** A high correlation ($r = 0.85$) exists between `Weight_(kg)` and `BMI`. In linear regression, severe collinearity inflates coefficient variance. In deep learning architectures with $L_2$ weight regularization (weight decay) and dropout, the network smoothly distributes gradient updates across correlated inputs without mathematical degeneracy.
- **Low-Variance & Weak Predictors:** Low-frequency age categories (e.g., 18–24, 25–29) exhibit minimal mutual information due to near-zero disease incidence in young adults. Rather than manually dropping them and losing pediatric/young adult coverage, the deep learning network suppresses irrelevant feature weights dynamically via dropout and regularization.

### 5. Critical Scientific Disclaimer on Causation

> **IMPORTANT:** Statistical association, mutual information gain, and permutation importance quantify predictive dependence within the observed observational sample. **Statistical correlation does not establish medical causation.** For example, while 'Poor General Health' is a strong predictor of heart disease, it serves as a symptomatic proxy rather than a biochemical cause. Definitive causal mechanisms require controlled clinical trials and longitudinal epidemiological investigations.
