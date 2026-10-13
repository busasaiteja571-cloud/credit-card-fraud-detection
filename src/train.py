# train.py - the full training flow: load -> split -> tune threshold -> train -> evaluate -> save

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
)

from src import config
from src.preprocess import load_data, split_data
from src.model import build_pipeline


def imbalance_ratio(y):
    """How many normal rows there are for every fraud row."""
    return float((y == 0).sum() / (y == 1).sum())


def find_best_threshold(y_true, y_prob):
    """Return the probability threshold that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision/recall have one extra value at the end; drop it to match thresholds
    precision, recall = precision[:-1], recall[:-1]

    # F1 at every threshold (1e-12 avoids dividing by zero)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, y_prob, threshold):
    """Compute the final metrics using the chosen threshold."""
    # Fraud (1) whenever the probability reaches the threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "frauds_caught": int(tp),
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
    }


def main():
    # 1) Load and split the raw data
    print("1/5 Loading and splitting data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 2) Hold out part of the TRAINING data to choose the threshold (never use the test set)
    print("2/5 Creating validation set...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VAL_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )

    # 3) Train on the smaller set and find the best threshold on validation data
    print("3/5 Finding the best threshold...")
    tuning_pipeline = build_pipeline(imbalance_ratio(y_tr))
    tuning_pipeline.fit(X_tr, y_tr)
    val_prob = tuning_pipeline.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_prob)
    print(f"    Best threshold: {threshold:.4f}")

    # 4) Train the FINAL pipeline on all the training data
    print("4/5 Training the final pipeline...")
    pipeline = build_pipeline(imbalance_ratio(y_train))
    pipeline.fit(X_train, y_train)

    # 5) Evaluate once on the untouched test set, then save everything
    print("5/5 Evaluating and saving...")
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_prob, threshold)

    # Create the models folder if it doesn't exist
    config.MODELS_DIR.mkdir(exist_ok=True)

    # Save the pipeline and the threshold together in one file
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, config.PIPELINE_PATH)

    # Save the scores as a readable JSON file
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal test results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


# This runs main() only when we execute the file directly (not when importing it)
if __name__ == "__main__":
    main()