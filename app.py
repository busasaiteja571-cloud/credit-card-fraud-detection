# app.py - a small web app to try the fraud detection model in the browser

import json
import pandas as pd
import streamlit as st

from src import config
from src.predict import predict

# Basic page settings: the browser tab title and wide layout
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

# The 30 input columns the model expects
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_data   # cache so the 150 MB CSV is only read once, not on every click
def load_dataset():
    return pd.read_csv(config.DATA_PATH)


# ---------- Page header ----------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "An XGBoost pipeline trained on 284,807 real (anonymized) card transactions, "
    "where only about 0.17% are fraud."
)

# ---------- Sidebar: show the model's saved test scores ----------
with st.sidebar:
    st.header("Model performance (test set)")
    if config.METRICS_PATH.exists():
        # Read the metrics.json file saved by train.py
        metrics = json.loads(config.METRICS_PATH.read_text())
        st.metric("PR-AUC", metrics["pr_auc"])
        st.metric("Precision", metrics["precision"])
        st.metric("Recall", metrics["recall"])
        st.metric("F1", metrics["f1"])
        st.caption(f"Decision threshold: {metrics['threshold']}")
    else:
        st.warning("Run `python -m src.train` first.")

# ---------- Two tabs: sample demo and CSV upload ----------
tab_demo, tab_upload = st.tabs(["Try a sample", "Upload a CSV"])

# ----- Tab 1: pick a random transaction from the dataset -----
with tab_demo:
    if not config.DATA_PATH.exists():
        st.info("The sample dataset isn't included in this deployment. Upload a CSV in the other tab to try predictions.")
    else:
        choice = st.radio("Pick a transaction type", ["Known fraud", "Known normal"], horizontal=True)

        if st.button("Pick a random transaction and predict"):
            df = load_dataset()

            # 1 = fraud, 0 = normal
            label = 1 if choice == "Known fraud" else 0

            # Take one random row of the chosen type
            row = df[df[config.TARGET] == label].sample(1)

            # Run the saved pipeline on that row
            result = predict(row)
            prob = float(result["fraud_probability"].iloc[0])
            flagged = int(result["is_fraud"].iloc[0])

            st.metric("Fraud probability", f"{prob:.2%}")
            if flagged:
                st.error("Flagged as FRAUD")
            else:
                st.success("Looks normal")

            st.caption(
                f"Actual label in the dataset: {'Fraud' if label else 'Normal'}. "
                "Note: this row may have been in the training data, so this is a demo, not a fair test."
            )
            st.dataframe(row)

# ----- Tab 2: upload your own CSV -----
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a CSV with the columns Time, V1 to V28, Amount", type="csv"
    )

    if uploaded is not None:
        data = pd.read_csv(uploaded)

        # Check that all required columns are present
        missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Keep only the expected columns, in the right order
            result = predict(data[EXPECTED_COLUMNS])

            flagged = result[result["is_fraud"] == 1]
            st.write(f"**{len(flagged)}** of **{len(result)}** transactions flagged as fraud.")

            # Show the flagged ones, riskiest first
            st.dataframe(flagged.sort_values("fraud_probability", ascending=False))

            # Let the user download all results as a CSV file
            st.download_button(
                "Download all predictions",
                result.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )