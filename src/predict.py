"""
Inference and Risk Prediction Module.

Provides safe artifact loading and inference for patient records,
returning predicted classification, calibrated probability, risk categorization,
and educational clinical disclaimers.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import tensorflow as tf
from src.data_preprocessing import HeartDiseasePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.keras")
DEFAULT_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
DEFAULT_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

class HeartDiseasePredictor:
    """Production inference wrapper for Heart Disease Deep Learning Model."""
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metadata_path = metadata_path
        
        self.preprocessor = None
        self.model = None
        self.metadata = {}
        self.optimal_threshold = 0.48
        
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Safely load trained model, preprocessor, and metadata."""
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {self.preprocessor_path}")
        self.preprocessor = HeartDiseasePreprocessor.load(self.preprocessor_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained Keras model not found at: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.48))
            
        logger.info("Predictor artifacts loaded successfully.")

    def predict(
        self,
        patient_data: Union[Dict[str, Any], pd.DataFrame],
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Predict heart disease risk for one or more patient records.
        
        Args:
            patient_data: Single dictionary or DataFrame containing required features.
            threshold: Optional custom decision threshold. Defaults to calibrated threshold.
            
        Returns:
            Dict[str, Any]: Prediction results including probability, class, risk category, and disclaimer.
        """
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()
        else:
            raise TypeError("patient_data must be a dict or pandas DataFrame")

        active_thresh = threshold if threshold is not None else self.optimal_threshold

        # Transform inputs using training pipeline
        X_proc = self.preprocessor.transform(df)
        
        # Raw probability output from sigmoid neuron
        probs = self.model.predict(X_proc.values, verbose=0).flatten()
        prob = float(probs[0])
        pred_class = int(prob >= active_thresh)

        # Categorize risk spectrum
        if prob < 0.25:
            risk_tier = "Low Relative Risk"
            recommendation = "Maintain routine preventative healthy lifestyle habits."
        elif prob < 0.50:
            risk_tier = "Moderate Relative Risk"
            recommendation = "Consider reviewing cardiovascular risk factors (blood pressure, lipid profile, activity level)."
        else:
            risk_tier = "High Relative Risk"
            recommendation = "Advise comprehensive medical screening and consultation with a healthcare professional."

        # Analyze qualitative risk amplifiers
        key_factors = []
        rec = df.iloc[0].to_dict()
        if rec.get("General_Health") in ["Poor", "Fair"]:
            key_factors.append(f"Self-reported health is {rec.get('General_Health')}")
        if rec.get("Diabetes") == "Yes":
            key_factors.append("Diagnosed with diabetes")
        if rec.get("Smoking_History") == "Yes":
            key_factors.append("Positive smoking history")
        if rec.get("Age_Category") in ["65-69", "70-74", "75-79", "80+"]:
            key_factors.append(f"Age category ({rec.get('Age_Category')}) carries elevated baseline demographic incidence")
        if rec.get("BMI", 0) >= 30.0:
            key_factors.append(f"BMI ({rec.get('BMI')}) is in the obese range (>= 30)")
        if rec.get("Exercise") == "No":
            key_factors.append("Absence of regular physical exercise")

        result = {
            "prediction_class": pred_class,
            "prediction_label": "Heart Disease Risk Detected" if pred_class == 1 else "No Heart Disease Detected",
            "model_probability": round(prob, 4),
            "model_probability_percentage": round(prob * 100, 2),
            "decision_threshold_used": round(active_thresh, 3),
            "risk_tier": risk_tier,
            "clinical_recommendation": recommendation,
            "notable_contributing_factors": key_factors if key_factors else ["No major high-risk indicators highlighted"],
            "disclaimer": (
                "EDUCATIONAL DISCLAIMER: This model prediction is an experimental computational estimate "
                "generated by a neural network trained on behavioral survey records for academic demonstration. "
                "It does NOT constitute medical advice, diagnostic opinion, or personalized clinical prognostic evaluation. "
                "Always consult a licensed physician or cardiologist for cardiovascular health guidance."
            )
        }
        return result

def predict_heart_disease(patient_record: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
    """Convenience function for external callers."""
    predictor = HeartDiseasePredictor()
    return predictor.predict(patient_record, threshold=threshold)

if __name__ == "__main__":
    sample_low_risk = {
        "General_Health": "Excellent",
        "Exercise": "Yes",
        "Depression": "No",
        "Diabetes": "No",
        "Sex": "Female",
        "Age_Category": "30-34",
        "Weight_(kg)": 62.0,
        "BMI": 22.5,
        "Smoking_History": "No",
        "Alcohol_Consumption": "No"
    }

    sample_high_risk = {
        "General_Health": "Poor",
        "Exercise": "No",
        "Depression": "Yes",
        "Diabetes": "Yes",
        "Sex": "Male",
        "Age_Category": "75-79",
        "Weight_(kg)": 105.0,
        "BMI": 36.2,
        "Smoking_History": "Yes",
        "Alcohol_Consumption": "Yes"
    }

    print("\n--- Low Risk Patient Test ---")
    res_low = predict_heart_disease(sample_low_risk)
    print(f"Prediction: {res_low['prediction_label']} ({res_low['model_probability_percentage']}%)")
    print(f"Risk Tier: {res_low['risk_tier']}")

    print("\n--- High Risk Patient Test ---")
    res_high = predict_heart_disease(sample_high_risk)
    print(f"Prediction: {res_high['prediction_label']} ({res_high['model_probability_percentage']}%)")
    print(f"Risk Tier: {res_high['risk_tier']}")
    print("Contributing factors:", res_high["notable_contributing_factors"])
