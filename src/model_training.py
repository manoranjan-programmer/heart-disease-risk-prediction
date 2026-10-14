"""
Deep Learning Model Training Module.

Constructs, compiles, and trains a Multi-Layer Perceptron (MLP) architecture
using TensorFlow/Keras with regularized dense layers, dropout, class weighting,
and callback-driven checkpointing.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set global seeds for deterministic reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

def build_mlp_model(
    input_dim: int,
    hidden_units: list = [64, 32],
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build a regularized Multi-Layer Perceptron (MLP) for binary classification.
    
    Architecture:
    - Input: input_dim features
    - Dense layers with ReLU and L2 weight decay
    - Dropout for variance reduction
    - Dense(1, activation='sigmoid') for probability output
    """
    model = keras.Sequential(name="HeartDisease_MLP")
    model.add(layers.Input(shape=(input_dim,)))
    
    for idx, units in enumerate(hidden_units):
        model.add(layers.Dense(
            units=units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{idx+1}_{units}"
        ))
        model.add(layers.BatchNormalization(name=f"batchnorm_{idx+1}"))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(rate=dropout_rate, name=f"dropout_{idx+1}_{dropout_rate}"))
            
    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC")
        ]
    )
    return model

def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calculate balanced class weights to address class imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed Class Weights: {weight_dict}")
    return weight_dict

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    epochs: int = 100,
    model_save_path: str = "models/best_model.keras",
    patience: int = 15,
    use_class_weights: bool = True
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train MLP model with early stopping, checkpointing, and class weights."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    class_weights = calculate_class_weights(y_train) if use_class_weights else None
    
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    logger.info(f"Beginning training for {epochs} max epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )
    logger.info("Training completed successfully.")
    return model, history

if __name__ == "__main__":
    set_seed(42)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    X_tr = train_df.drop(columns=["Heart_Disease"]).values
    y_tr = train_df["Heart_Disease"].values
    X_va = val_df.drop(columns=["Heart_Disease"]).values
    y_va = val_df["Heart_Disease"].values
    
    model = build_mlp_model(input_dim=X_tr.shape[1], hidden_units=[64, 32], dropout_rate=0.2)
    model.summary()
    trained_model, history = train_model(model, X_tr, y_tr, X_va, y_va, epochs=10)
    print("Test run completed.")
