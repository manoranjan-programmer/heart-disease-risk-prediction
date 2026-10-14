"""
Data Preprocessing Pipeline Module.

Implements leak-free data preprocessing, outlier analysis, imputation,
categorical encoding, numerical feature scaling, and stratified train/val/test splitting.
"""

import os
import json
import logging
import joblib
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HeartDiseasePreprocessor:
    """
    Robust preprocessing pipeline for tabular heart disease risk data.
    Ensures zero data leakage by fitting exclusively on training data.
    """
    def __init__(self, numerical_cols: List[str] = None, categorical_cols: List[str] = None):
        self.numerical_cols = numerical_cols or ["Weight_(kg)", "BMI"]
        self.categorical_cols = categorical_cols or [
            "General_Health", "Exercise", "Depression", "Diabetes",
            "Sex", "Age_Category", "Smoking_History", "Alcohol_Consumption"
        ]
        self.pipeline: ColumnTransformer = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def _build_pipeline(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols)
            ],
            remainder="drop"
        )
        return transformer

    def fit(self, X: pd.DataFrame):
        """Fit preprocessor exclusively on training data."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X)
        self._is_fitted = True
        
        # Extract feature names
        num_names = self.numerical_cols
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["encoder"]
        cat_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = num_names + cat_names
        logger.info(f"Preprocessor fitted successfully. Total engineered features: {len(self.feature_names)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset using fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        arr = self.pipeline.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training dataset."""
        return self.fit(X).transform(X)

    def save(self, filepath: str) -> None:
        """Serialize preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "is_fitted": self._is_fitted
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "HeartDiseasePreprocessor":
        """Deserialize preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        data = joblib.load(filepath)
        obj = cls(numerical_cols=data["numerical_cols"], categorical_cols=data["categorical_cols"])
        obj.pipeline = data["pipeline"]
        obj.feature_names = data["feature_names"]
        obj._is_fitted = data["is_fitted"]
        logger.info(f"Preprocessor loaded successfully from {filepath}")
        return obj

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    Detect outliers using Interquartile Range (IQR) rule and report medical justification.
    """
    outlier_report = {}
    for col in columns:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_report[col] = {
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
            "min_observed": float(df[col].min()),
            "max_observed": float(df[col].max()),
            "clinical_decision": (
                "Preserved without truncation. Values reflect authentic clinical variations "
                "(e.g., severe obesity class III with BMI > 40 is a genuine risk factor for heart disease)."
            )
        }
    return outlier_report

def prepare_splits(
    df: pd.DataFrame,
    target_col: str = "Heart_Disease",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified three-way split: 70% Train, 15% Validation, 15% Test.
    """
    # Verify no missing target
    clean_df = df.dropna(subset=[target_col]).copy()
    
    # Encode target: Yes -> 1, No -> 0
    y_raw = clean_df[target_col].astype(str).str.strip()
    y = y_raw.map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    if y.isnull().any():
        raise ValueError(f"Target column contains unrecognized classes: {clean_df[target_col].unique()}")
        
    X = clean_df.drop(columns=[target_col])

    # First split: Separate Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: From remaining 85%, extract Validation set (15% of total => 0.15 / 0.85 ≈ 0.17647)
    val_ratio_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, stratify=y_temp, random_state=random_state
    )

    logger.info(f"Split sizes: Train={len(X_train)} ({len(X_train)/len(df):.1%}), "
                f"Val={len(X_val)} ({len(X_val)/len(df):.1%}), "
                f"Test={len(X_test)} ({len(X_test)/len(df):.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def run_preprocessing_pipeline(
    raw_path: str = "data/raw/heart.csv",
    output_dir: str = "data/processed",
    models_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute end-to-end preprocessing pipeline and save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    logger.info(f"Starting preprocessing pipeline for dataset with {len(df)} records.")

    # 1. Outlier analysis
    outlier_info = detect_outliers_iqr(df, ["Weight_(kg)", "BMI"])

    # 2. Stratified train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(
        df, target_col="Heart_Disease", test_size=0.15, val_size=0.15, random_state=random_state
    )

    # 3. Fit Preprocessor exclusively on X_train
    preprocessor = HeartDiseasePreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # 4. Save preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    # 5. Assemble processed dataframes and save
    train_df = X_train_proc.copy()
    train_df["Heart_Disease"] = y_train.values

    val_df = X_val_proc.copy()
    val_df["Heart_Disease"] = y_val.values

    test_df = X_test_proc.copy()
    test_df["Heart_Disease"] = y_test.values

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Processed splits saved to {output_dir}")

    # 6. Save feature metadata
    metadata = {
        "raw_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "target_col": "Heart_Disease",
        "target_classes": {"0": "No", "1": "Yes"},
        "train_class_distribution": {
            "0 (No)": int((y_train == 0).sum()),
            "1 (Yes)": int((y_train == 1).sum()),
            "prevalence_pct": round(float((y_train == 1).mean() * 100), 2)
        },
        "val_class_distribution": {
            "0 (No)": int((y_val == 0).sum()),
            "1 (Yes)": int((y_val == 1).sum()),
            "prevalence_pct": round(float((y_val == 1).mean() * 100), 2)
        },
        "test_class_distribution": {
            "0 (No)": int((y_test == 0).sum()),
            "1 (Yes)": int((y_test == 1).sum()),
            "prevalence_pct": round(float((y_test == 1).mean() * 100), 2)
        },
        "feature_names": preprocessor.feature_names,
        "num_features": len(preprocessor.feature_names),
        "outlier_analysis": outlier_info
    }
    
    metadata_path = os.path.join(output_dir, "feature_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Preprocessing completed. Feature count: {len(preprocessor.feature_names)}")
    return metadata

if __name__ == "__main__":
    meta = run_preprocessing_pipeline()
    print("Preprocessing completed successfully.")
    print(f"Engineered feature count: {meta['num_features']}")
