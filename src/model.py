# model.py - builds the pipeline: scaler + XGBoost chained together

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src import config


def build_pipeline(scale_pos_weight):
    """Create an untrained pipeline.

    scale_pos_weight = (number of normal rows) / (number of fraud rows),
    which tells XGBoost to treat fraud mistakes as much more costly.
    """

    # Step 1: scale only Time and Amount; pass every other column through unchanged
    preprocessor = ColumnTransformer(
        transformers=[("scale", RobustScaler(), config.COLS_TO_SCALE)],
        remainder="passthrough",            # keep V1..V28 as they are
        verbose_feature_names_out=False,    # keep the original column names
    )
    # Return a DataFrame (with column names) instead of a plain array
    preprocessor.set_output(transform="pandas")

    # Step 2: the classifier (same settings as Stage 5)
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )

    # Chain them: data flows through the scaler first, then into XGBoost
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])