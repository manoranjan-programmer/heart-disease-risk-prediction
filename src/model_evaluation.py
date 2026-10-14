"""
Model Evaluation and Threshold Calibration Module.

Trains the final selected MLP architecture, optimizes classification threshold
on validation data, and performs rigorous evaluation on the untouched test set.
Generates publication-quality diagnostic plots (Loss, Accuracy/AUC, Confusion Matrix, ROC).
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
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

from src.model_training import set_seed, build_mlp_model, calculate_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal classification decision threshold maximizing validation F1-score.
    Ensures balanced clinical sensitivity without excessive false alarms.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = -1.0
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            
    logger.info(f"Threshold Optimization on Validation Set: Best Threshold={best_thresh:.3f} (Val F1={best_f1:.4f})")
    return float(best_thresh), float(best_f1)

def plot_training_loss_curves(history: tf.keras.callbacks.History, output_dir: str) -> str:
    """1. Training Loss vs Epoch."""
    filepath = os.path.join(output_dir, "11_training_loss_curves.png")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, history.history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2.2, linestyle="--")
    
    min_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(min_val_loss) + 1
    ax.scatter([best_epoch], [min_val_loss], color="#2ca02c", s=100, zorder=5, label=f"Min Val Loss ({min_val_loss:.4f} @ Ep {best_epoch})")
    
    ax.set_title("Multi-Layer Perceptron: Training Loss vs Epoch\n(Binary Cross-Entropy with L2 Weight Decay)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_training_accuracy_curves(history: tf.keras.callbacks.History, output_dir: str) -> str:
    """2. Training Accuracy & ROC-AUC vs Epoch."""
    filepath = os.path.join(output_dir, "12_training_accuracy_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    epochs = range(1, len(history.history["accuracy"]) + 1)
    
    # 1. Accuracy
    axes[0].plot(epochs, history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2.0)
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.0, linestyle="--")
    axes[0].set_title("Training vs Validation Accuracy Across Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    
    # 2. ROC-AUC
    axes[1].plot(epochs, history.history["auc"], label="Training ROC-AUC", color="#2ca02c", linewidth=2.0)
    axes[1].plot(epochs, history.history["val_auc"], label="Validation ROC-AUC", color="#d62728", linewidth=2.0, linestyle="--")
    axes[1].set_title("Training vs Validation ROC-AUC Discrimination", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> str:
    """3. Confusion Matrix with Counts, Percentages, and Clinical Annotations."""
    filepath = os.path.join(output_dir, "13_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    tn, fp, fn, tp = cm.ravel()
    total = len(cm.ravel())
    
    cm_labels = np.array([
        [f"True Negative (TN)\n{tn}\n({tn/(tn+fp):.1%})", f"False Positive (FP)\n{fp}\n({fp/(tn+fp):.1%})"],
        [f"False Negative (FN)\n{fn}\n({fn/(fn+tp):.1%})", f"True Positive (TP)\n{tp}\n({tp/(fn+tp):.1%})"]
    ])
    
    sns.heatmap(cm, annot=cm_labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["No Heart Disease (0)", "Heart Disease (1)"],
                yticklabels=["No Heart Disease (0)", "Heart Disease (1)"],
                annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax)
    
    ax.set_title("Test Set Confusion Matrix\n(Evaluation on Untouched Test Split)", pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", labelpad=10, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", labelpad=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def plot_roc_and_pr_curves(
    y_test: np.ndarray, y_probs: np.ndarray, output_dir: str
) -> str:
    """4. Receiver Operating Characteristic (ROC) and Precision-Recall Curves."""
    filepath = os.path.join(output_dir, "14_roc_curve.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_score = roc_auc_score(y_test, y_probs)
    
    axes[0].plot(fpr, tpr, color="#d95f02", linewidth=2.5, label=f"MLP Test ROC (AUC = {roc_score:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    axes[0].set_title("Test Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_vals, precision_vals)
    baseline_pr = float(np.sum(y_test == 1) / len(y_test))
    
    axes[1].plot(recall_vals, precision_vals, color="#2b5c8f", linewidth=2.5, label=f"MLP Test PR (AUC = {pr_auc:.4f})")
    axes[1].axhline(baseline_pr, color="#7f7f7f", linestyle="--", linewidth=1.5, label=f"Prevalence Baseline ({baseline_pr:.2f})")
    axes[1].set_title("Test Precision-Recall Curve (Imbalanced Benchmark)", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def run_model_evaluation(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv",
    best_params_path: str = "models/best_hyperparameters.json",
    model_save_path: str = "models/final_model.keras",
    figures_dir: str = "outputs/figures",
    metrics_dir: str = "outputs/metrics"
) -> Dict[str, Any]:
    """Execute end-to-end model training, validation calibration, and test evaluation."""
    set_seed(42)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["Heart_Disease"]).values
    y_train = train_df["Heart_Disease"].values
    X_val = val_df.drop(columns=["Heart_Disease"]).values
    y_val = val_df["Heart_Disease"].values
    X_test = test_df.drop(columns=["Heart_Disease"]).values
    y_test = test_df["Heart_Disease"].values
    
    # Load optimal hyperparameters
    if os.path.exists(best_params_path):
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_meta = json.load(f)
        params = best_meta["parameters"]
    else:
        params = {"hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 32}
        
    logger.info(f"Training final MLP with selected parameters: {params}")
    model = build_mlp_model(
        input_dim=X_train.shape[1],
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"]
    )
    
    class_weights = calculate_class_weights(y_train)
    
    cb_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=params["batch_size"],
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )
    
    # Load best checkpointed weights
    model = tf.keras.models.load_model(model_save_path)
    logger.info(f"Loaded optimal checkpointed model from {model_save_path}")
    
    # Validation threshold optimization
    val_probs = model.predict(X_val, verbose=0).flatten()
    opt_threshold, val_f1 = find_optimal_threshold(y_val, val_probs)
    
    # Evaluate on UNTOUCHED TEST SET
    logger.info("Evaluating final model on untouched test set...")
    test_probs = model.predict(X_test, verbose=0).flatten()
    test_preds = (test_probs >= opt_threshold).astype(int)
    test_preds_default = (test_probs >= 0.5).astype(int)
    
    # Metrics calculation
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Classification Report
    cls_report_dict = classification_report(y_test, test_preds, target_names=["No Heart Disease", "Heart Disease"], output_dict=True)
    cls_report_df = pd.DataFrame(cls_report_dict).transpose()
    cls_report_csv = os.path.join(metrics_dir, "classification_report.csv")
    cls_report_df.to_csv(cls_report_csv)
    
    test_metrics = {
        "dataset_split": "Untouched Test Set (150 samples)",
        "selected_configuration": params,
        "optimal_threshold": round(opt_threshold, 3),
        "default_threshold": 0.5,
        "metrics_at_optimal_threshold": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall_sensitivity": round(float(test_rec), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_roc_auc), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "metrics_at_default_0.5_threshold": {
            "accuracy": round(float(accuracy_score(y_test, test_preds_default)), 4),
            "precision": round(float(precision_score(y_test, test_preds_default, zero_division=0)), 4),
            "recall_sensitivity": round(float(recall_score(y_test, test_preds_default, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, test_preds_default, zero_division=0)), 4)
        }
    }
    
    # Save test metrics JSON
    metrics_json_path = os.path.join(metrics_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Test metrics saved to: {metrics_json_path}")
    
    # Generate Evaluation Figures
    fig11 = plot_training_loss_curves(history, figures_dir)
    fig12 = plot_training_accuracy_curves(history, figures_dir)
    fig13 = plot_confusion_matrix(cm, figures_dir)
    fig14 = plot_roc_and_pr_curves(y_test, test_probs, figures_dir)
    
    # Save complete model metadata
    model_metadata = {
        "model_architecture": "Multi-Layer Perceptron (TensorFlow/Keras)",
        "input_features": X_train.shape[1],
        "hidden_layers": params["hidden_units"],
        "dropout_rate": params["dropout_rate"],
        "optimizer": "Adam",
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "optimal_threshold": opt_threshold,
        "test_performance": test_metrics["metrics_at_optimal_threshold"],
        "training_epochs_completed": len(history.history["loss"])
    }
    meta_save_path = os.path.join(os.path.dirname(model_save_path), "model_metadata.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=4)
        
    logger.info(f"Model metadata saved to {meta_save_path}")
    logger.info("Evaluation pipeline finished successfully.")
    return test_metrics

if __name__ == "__main__":
    metrics = run_model_evaluation()
    print("Test Evaluation Metrics:")
    print(json.dumps(metrics["metrics_at_optimal_threshold"], indent=2))
