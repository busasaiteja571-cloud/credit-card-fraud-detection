# config.py - all paths and settings live here, so we change them in ONE place

from pathlib import Path

# BASE_DIR = the project root folder.
# __file__ is this file (src/config.py); .parent.parent goes up two levels to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Where the raw dataset is
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

# Where trained files are saved
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "fraud_pipeline.pkl"   # model + threshold bundle
METRICS_PATH = MODELS_DIR / "metrics.json"          # final test scores

# Name of the column we want to predict
TARGET = "Class"

# Columns that need scaling (V1 to V28 are already scaled)
COLS_TO_SCALE = ["Time", "Amount"]

# Settings reused everywhere so results are repeatable
RANDOM_STATE = 42
TEST_SIZE = 0.2   # 20% of data for the final test
VAL_SIZE = 0.2    # 20% of training data for choosing the threshold