# config.py - all paths and settings live here, so we change them in ONE place

import os
from pathlib import Path

# BASE_DIR = the project root folder.
# __file__ is this file (src/config.py); .parent.parent goes up two levels to the project root
BASE_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

# Allow runtime overrides for cloud/container deployments
DATA_PATH = Path(os.environ.get("DATA_PATH", BASE_DIR / "data" / "creditcard.csv"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
PIPELINE_PATH = Path(os.environ.get("PIPELINE_PATH", MODELS_DIR / "fraud_pipeline.pkl"))
METRICS_PATH = Path(os.environ.get("METRICS_PATH", MODELS_DIR / "metrics.json"))

# Name of the column we want to predict
TARGET = "Class"

# Columns that need scaling (V1 to V28 are already scaled)
COLS_TO_SCALE = ["Time", "Amount"]

# Settings reused everywhere so results are repeatable
RANDOM_STATE = 42
TEST_SIZE = 0.2   # 20% of data for the final test
VAL_SIZE = 0.2    # 20% of training data for choosing the threshold