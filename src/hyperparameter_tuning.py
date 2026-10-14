"""
Hyperparameter Tuning Module.

Performs systematic grid evaluation across architectures, learning rates,
dropout rates, and batch sizes using strictly validation data.
Logs metrics, visualizes parameter trade-offs, and preserves test set isolation.
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
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from src.model_training import set_seed, build_mlp_model, calculate_class_weights


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def get_tuning_configurations() -> List[Dict[str, Any]]:
    """Define a structured matrix of hyperparameter combinations for systematic evaluation."""
    return [
        {"config_id": "Config_1_Baseline", "hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 32},
        {"config_id": "Config_2_Wider", "hidden_units": [128, 64], "dropout_rate": 0.3, "learning_rate": 0.001, "batch_size": 32},
        {"config_id": "Config_3_Deeper", "hidden_units": [64, 32, 16], "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 32},
        {"config_id": "Config_4_Compact", "hidden_units": [32, 16], "dropout_rate": 0.1, "learning_rate": 0.001, "batch_size": 32},
        {"config_id": "Config_5_LowLR", "hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.0005, "batch_size": 32},
        {"config_id": "Config_6_HighLR", "hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.002, "batch_size": 32},
        {"config_id": "Config_7_HighDropout", "hidden_units": [128, 64], "dropout_rate": 0.4, "learning_rate": 0.001, "batch_size": 32},
        {"config_id": "Config_8_SmallBatch", "hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 16},
        {"config_id": "Config_9_LargeBatch", "hidden_units": [64, 32], "dropout_rate": 0.2, "learning_rate": 0.001, "batch_size": 64},
        {"config_id": "Config_10_Regularized", "hidden_units": [64, 32, 16], "dropout_rate": 0.3, "learning_rate": 0.0005, "batch_size": 32}
    ]

def evaluate_configuration(
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
    epochs: int = 50
) -> Dict[str, Any]:
    """Train and evaluate a single hyperparameter configuration on validation set."""
    set_seed(42)
    input_dim = X_train.shape[1]
    
    model = build_mlp_model(
        input_dim=input_dim,
        hidden_units=config["hidden_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"]
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=config["batch_size"],
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )
    
    val_eval = model.evaluate(X_val, y_val, return_dict=True, verbose=0)
    
    # Calculate exact sklearn metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    
    val_auc = float(roc_auc_score(y_val, y_val_prob))
    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_prec = float(precision_score(y_val, y_val_pred, zero_division=0))
    val_rec = float(recall_score(y_val, y_val_pred, zero_division=0))
    val_f1 = float(f1_score(y_val, y_val_pred, zero_division=0))
    
    result = {
        "config_id": config["config_id"],
        "hidden_units": str(config["hidden_units"]),
        "dropout_rate": config["dropout_rate"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "val_loss": round(float(val_eval.get("loss", 0.0)), 4),
        "val_accuracy": round(val_acc, 4),
        "val_precision": round(val_prec, 4),
        "val_recall": round(val_rec, 4),
        "val_auc": round(val_auc, 4),
        "val_f1": round(val_f1, 4),
        "epochs_trained": len(history.history["loss"])
    }
    return result

def plot_tuning_comparison(results_df: pd.DataFrame, output_dir: str) -> str:
    """Generate high-resolution chart comparing all evaluated configurations."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "10_hyperparameter_tuning_comparison.png")
    
    sorted_df = results_df.sort_values(by="val_auc", ascending=False).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    
    # 1. Validation ROC-AUC comparison
    colors = ["#2b5c8f" if i > 0 else "#2ca02c" for i in range(len(sorted_df))]
    bars = axes[0].barh(sorted_df["config_id"], sorted_df["val_auc"], color=colors, edgecolor="black", height=0.6)
    axes[0].set_title("Validation ROC-AUC Across Configurations\n(Green = Best Selected Architecture)", fontweight="bold")
    axes[0].set_xlabel("Validation ROC-AUC Score")
    axes[0].set_xlim(0.6, 0.9)
    axes[0].invert_yaxis()
    
    for bar in bars:
        w = bar.get_width()
        axes[0].annotate(f"{w:.4f}",
                         xy=(w, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=9, fontweight="bold")
        
    # 2. Multi-Metric Radar/Bar Comparison for Top 4 configs
    top_4 = sorted_df.head(4).copy()
    metrics_to_plot = ["val_auc", "val_recall", "val_precision", "val_accuracy"]
    tidy_df = pd.melt(top_4, id_vars=["config_id"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    sns.barplot(x="Metric", y="Score", hue="config_id", data=tidy_df, ax=axes[1], palette="Set2", edgecolor="black")
    axes[1].set_title("Multi-Metric Profile of Top 4 Configurations", fontweight="bold")
    axes[1].set_ylabel("Validation Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(title="Configuration", loc="lower right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath

def run_hyperparameter_tuning(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    output_metrics_dir: str = "outputs/metrics",
    output_figures_dir: str = "outputs/figures",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Execute hyperparameter sweep and select optimal configuration."""
    set_seed(42)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df.drop(columns=["Heart_Disease"]).values
    y_train = train_df["Heart_Disease"].values
    X_val = val_df.drop(columns=["Heart_Disease"]).values
    y_val = val_df["Heart_Disease"].values
    
    class_weights = calculate_class_weights(y_train)
    configs = get_tuning_configurations()
    
    logger.info(f"Initiating tuning search across {len(configs)} configurations...")
    records = []
    for cfg in configs:
        logger.info(f"Evaluating {cfg['config_id']}: units={cfg['hidden_units']}, dropout={cfg['dropout_rate']}, lr={cfg['learning_rate']}, batch={cfg['batch_size']}")
        res = evaluate_configuration(cfg, X_train, y_train, X_val, y_val, class_weights)
        records.append(res)
        
    results_df = pd.DataFrame(records)
    
    # Save CSV
    os.makedirs(output_metrics_dir, exist_ok=True)
    csv_path = os.path.join(output_metrics_dir, "hyperparameter_tuning_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Tuning results written to: {csv_path}")
    
    # Plot comparison
    plot_path = plot_tuning_comparison(results_df, output_figures_dir)
    logger.info(f"Tuning comparison plot saved to: {plot_path}")
    
    # Identify best configuration by highest validation AUC (and lowest val loss as tiebreaker)
    best_row = results_df.sort_values(by=["val_auc", "val_loss"], ascending=[False, True]).iloc[0]
    best_config_id = best_row["config_id"]
    best_config = next(c for c in configs if c["config_id"] == best_config_id)
    
    best_metadata = {
        "best_config_id": best_config_id,
        "parameters": best_config,
        "validation_performance": {
            "val_auc": float(best_row["val_auc"]),
            "val_loss": float(best_row["val_loss"]),
            "val_accuracy": float(best_row["val_accuracy"]),
            "val_precision": float(best_row["val_precision"]),
            "val_recall": float(best_row["val_recall"]),
            "val_f1": float(best_row["val_f1"]),
            "epochs_trained": int(best_row["epochs_trained"])
        },
        "selection_rationale": (
            f"Configuration '{best_config_id}' achieved the optimal balance of discrimination capability "
            f"(Validation ROC-AUC: {best_row['val_auc']:.4f}) and clinical Sensitivity (Recall: {best_row['val_recall']:.4f}) "
            f"while maintaining low validation cross-entropy loss ({best_row['val_loss']:.4f}). "
            f"The untouched test set was completely excluded from this selection procedure."
        )
    }
    
    os.makedirs(models_dir, exist_ok=True)
    best_meta_path = os.path.join(models_dir, "best_hyperparameters.json")
    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(best_metadata, f, indent=4)
        
    logger.info(f"Optimal configuration selected: {best_config_id} (val_auc={best_row['val_auc']})")
    return best_metadata

if __name__ == "__main__":
    run_hyperparameter_tuning()
    print("Hyperparameter tuning completed successfully.")
