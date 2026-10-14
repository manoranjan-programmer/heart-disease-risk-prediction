# Dataset Profile and Exploratory Inspection Report

## Project: Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk

- **Total Records (Rows):** 999
- **Total Attributes (Columns):** 11
- **Duplicate Rows:** 0
- **Potential Leakage / Identifier Columns:** None detected

### 1. Attribute Inventory and Data Types

| Attribute Name | Data Type | Feature Class | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- | :--- |
| `General_Health` | str | Categorical | 0 | 0.0% |
| `Exercise` | str | Categorical | 0 | 0.0% |
| `Depression` | str | Categorical | 0 | 0.0% |
| `Diabetes` | str | Categorical | 0 | 0.0% |
| `Sex` | str | Categorical | 0 | 0.0% |
| `Age_Category` | str | Categorical | 0 | 0.0% |
| `Weight_(kg)` | float64 | Numerical (Continuous) | 0 | 0.0% |
| `BMI` | float64 | Numerical (Continuous) | 0 | 0.0% |
| `Smoking_History` | str | Categorical | 0 | 0.0% |
| `Alcohol_Consumption` | str | Categorical | 0 | 0.0% |
| `Heart_Disease` | str | Binary Target | 0 | 0.0% |

### 2. Numerical Features Statistical Summary

| Feature | Mean | Std Dev | Median | Min | Max | IQR | Skewness | Kurtosis | IQR Outliers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Weight_(kg)` | 81.00 | 19.80 | 79.38 | 32.66 | 167.83 | 23.58 | 0.66 | 0.75 | 22 |
| `BMI` | 28.56 | 6.22 | 27.78 | 14.06 | 51.81 | 7.68 | 0.78 | 0.94 | 30 |

### 3. Categorical Features Distribution

#### Attribute: `General_Health` (5 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| Good | 340 | 34.03% |
| Very Good | 269 | 26.93% |
| Fair | 211 | 21.12% |
| Excellent | 107 | 10.71% |
| Poor | 72 | 7.21% |

#### Attribute: `Exercise` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| Yes | 598 | 59.86% |
| No | 401 | 40.14% |

#### Attribute: `Depression` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| No | 828 | 82.88% |
| Yes | 171 | 17.12% |

#### Attribute: `Diabetes` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| No | 751 | 75.18% |
| Yes | 248 | 24.82% |

#### Attribute: `Sex` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| Female | 687 | 68.77% |
| Male | 312 | 31.23% |

#### Attribute: `Age_Category` (13 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| 70-74 | 202 | 20.22% |
| 80+ | 177 | 17.72% |
| 65-69 | 152 | 15.22% |
| 75-79 | 133 | 13.31% |
| 60-64 | 127 | 12.71% |
| 55-59 | 65 | 6.51% |
| 45-49 | 35 | 3.5% |
| 50-54 | 30 | 3.0% |
| 40-44 | 24 | 2.4% |
| 35-39 | 19 | 1.9% |
| 30-34 | 15 | 1.5% |
| 18-24 | 14 | 1.4% |
| 25-29 | 6 | 0.6% |

#### Attribute: `Smoking_History` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| No | 606 | 60.66% |
| Yes | 393 | 39.34% |

#### Attribute: `Alcohol_Consumption` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| Yes | 530 | 53.05% |
| No | 469 | 46.95% |

#### Attribute: `Heart_Disease` (2 unique levels)
| Category Level | Frequency | Percentage (%) |
| :--- | :--- | :--- |
| No | 858 | 85.89% |
| Yes | 141 | 14.11% |

### 4. Target Variable Analysis

- **Identified Target Column:** `Heart_Disease`
- **Class Distribution:**
  - `No`: 858 samples (85.89%)
  - `Yes`: 141 samples (14.11%)
- **Imbalance Ratio:** 6.09:1 (Majority to Minority)
- **Imbalance Status:** Significant class imbalance present. Specialized handling (e.g. class weighting, threshold calibration) required.