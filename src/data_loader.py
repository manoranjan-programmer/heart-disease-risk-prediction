"""
Data Loader and Initial Dataset Inspection Module.

Handles dataset loading, structure validation, and profiling for the
Heart Disease Risk Prediction project.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load tabular dataset from CSV or Excel file.
    
    Args:
        filepath: Path to the data file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    logger.info(f"Loading dataset from: {filepath} (extension: {ext})")
    
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .csv or .xlsx")
        
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns successfully.")
    return df

def inspect_dataset(df: pd.DataFrame, target_candidate: str = "Heart_Disease") -> Dict[str, Any]:
    """
    Perform comprehensive dataset inspection and profile generation.
    
    Args:
        df: Input DataFrame.
        target_candidate: Anticipated target column name.
        
    Returns:
        Dict[str, Any]: Dictionary containing detailed dataset profile.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    
    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    
    # Numerical summaries
    num_summary = {}
    for col in num_cols:
        col_data = df[col].dropna()
        q25 = float(col_data.quantile(0.25))
        q75 = float(col_data.quantile(0.75))
        iqr = float(q75 - q25)
        num_summary[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "median": float(col_data.median()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
            "outliers_iqr_count": int(((col_data < (q25 - 1.5 * iqr)) | (col_data > (q75 + 1.5 * iqr))).sum())
        }
        
    # Categorical summaries
    cat_summary = {}
    for col in cat_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        cat_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "frequencies": {str(k): int(v) for k, v in val_counts.items()},
            "percentages": {str(k): round(float(v) / len(df) * 100, 2) for k, v in val_counts.items()}
        }
        
    # Target distribution
    target_info = {}
    if target_candidate in df.columns:
        counts = df[target_candidate].value_counts().to_dict()
        percentages = (df[target_candidate].value_counts(normalize=True) * 100).round(2).to_dict()
        target_info = {
            "target_column": target_candidate,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): float(v) for k, v in percentages.items()},
            "is_imbalanced": (max(percentages.values()) / min(percentages.values())) > 1.5,
            "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 2)
        }
    else:
        logger.warning(f"Target column '{target_candidate}' not found among columns: {df.columns.tolist()}")
        
    profile = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "duplicate_rows": duplicate_count,
        "numerical_summary": num_summary,
        "categorical_summary": cat_summary,
        "target_profile": target_info,
        "potential_leakage_or_id": [col for col in df.columns if "id" in col.lower() or "patient" in col.lower()]
    }
    
    return profile

def generate_profile_report(profile: Dict[str, Any], output_path: str) -> None:
    """
    Generate an academic Markdown report of the dataset profile.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    md = []
    md.append("# Dataset Profile and Exploratory Inspection Report\n")
    md.append("## Project: Exploratory Data Analysis and Deep Learning-Based Prediction of Heart Disease Risk\n")
    md.append(f"- **Total Records (Rows):** {profile['num_rows']}")
    md.append(f"- **Total Attributes (Columns):** {profile['num_columns']}")
    md.append(f"- **Duplicate Rows:** {profile['duplicate_rows']}")
    md.append(f"- **Potential Leakage / Identifier Columns:** {profile['potential_leakage_or_id'] if profile['potential_leakage_or_id'] else 'None detected'}\n")
    
    md.append("### 1. Attribute Inventory and Data Types\n")
    md.append("| Attribute Name | Data Type | Feature Class | Missing Count | Missing (%) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for col in profile["columns"]:
        ftype = "Numerical (Continuous)" if col in profile["numerical_features"] else "Categorical"
        if col == profile["target_profile"].get("target_column"):
            ftype = "Binary Target"
        dtype = profile["dtypes"][col]
        miss_cnt = profile["missing_counts"][col]
        miss_pct = profile["missing_percentages"][col]
        md.append(f"| `{col}` | {dtype} | {ftype} | {miss_cnt} | {miss_pct}% |")
        
    md.append("\n### 2. Numerical Features Statistical Summary\n")
    md.append("| Feature | Mean | Std Dev | Median | Min | Max | IQR | Skewness | Kurtosis | IQR Outliers |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for col, stats in profile["numerical_summary"].items():
        md.append(f"| `{col}` | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['iqr']:.2f} | {stats['skewness']:.2f} | {stats['kurtosis']:.2f} | {stats['outliers_iqr_count']} |")
        
    md.append("\n### 3. Categorical Features Distribution\n")
    for col, stats in profile["categorical_summary"].items():
        md.append(f"#### Attribute: `{col}` ({stats['unique_count']} unique levels)")
        md.append("| Category Level | Frequency | Percentage (%) |")
        md.append("| :--- | :--- | :--- |")
        for lvl, cnt in stats["frequencies"].items():
            pct = stats["percentages"][lvl]
            md.append(f"| {lvl} | {cnt} | {pct}% |")
        md.append("")
        
    md.append("### 4. Target Variable Analysis\n")
    t_info = profile["target_profile"]
    if t_info:
        md.append(f"- **Identified Target Column:** `{t_info['target_column']}`")
        md.append(f"- **Class Distribution:**")
        for k, v in t_info["counts"].items():
            pct = t_info["percentages"][k]
            md.append(f"  - `{k}`: {v} samples ({pct}%)")
        md.append(f"- **Imbalance Ratio:** {t_info['imbalance_ratio']}:1 (Majority to Minority)")
        md.append(f"- **Imbalance Status:** {'Significant class imbalance present. Specialized handling (e.g. class weighting, threshold calibration) required.' if t_info['is_imbalanced'] else 'Balanced distribution'}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    logger.info(f"Dataset profile report written to {output_path}")

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "heart.csv")
    df = load_data(raw_path)
    profile = inspect_dataset(df)
    report_path = os.path.join("outputs", "reports", "dataset_profile_report.md")
    generate_profile_report(profile, report_path)
    print("Dataset inspection completed successfully.")
