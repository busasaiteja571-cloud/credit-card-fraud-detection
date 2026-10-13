# predict.py - load the saved pipeline and predict fraud on new transactions

import sys
import joblib
import pandas as pd

from src import config


def load_bundle():
    """Load the saved pipeline and its threshold."""
    return joblib.load(config.PIPELINE_PATH)


def predict(df):
    """Add fraud_probability and is_fraud columns to a DataFrame of RAW transactions."""
    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    # Remove the answer column if present (it is not an input feature)
    features = df.drop(columns=[config.TARGET], errors="ignore")

    # The pipeline scales Time/Amount itself, then predicts the probability of fraud
    prob = pipeline.predict_proba(features)[:, 1]

    result = df.copy()
    result["fraud_probability"] = prob.round(4)
    result["is_fraud"] = (prob >= threshold).astype(int)   # 1 if probability >= threshold
    return result


def demo():
    """Try the pipeline on 5 known frauds and 5 known normal transactions."""
    df = pd.read_csv(config.DATA_PATH)
    sample = pd.concat([
        df[df[config.TARGET] == 1].head(5),
        df[df[config.TARGET] == 0].head(5),
    ])

    result = predict(sample)

    # Show actual vs predicted side by side
    print(result[[config.TARGET, "fraud_probability", "is_fraud"]].to_string())


if __name__ == "__main__":
    # If a CSV path is given: python -m src.predict my_transactions.csv
    if len(sys.argv) > 1:
        new_data = pd.read_csv(sys.argv[1])
        print(predict(new_data)[["fraud_probability", "is_fraud"]].to_string())
    else:
        demo()