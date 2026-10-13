# preprocess.py - loading, cleaning and splitting the data

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path=config.DATA_PATH):
    """Read the CSV and remove duplicate rows."""
    df = pd.read_csv(path)

    # Remove exact duplicate transactions (same as Stage 3)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df):
    """Split into train and test sets, keeping the same fraud ratio in both."""
    # X = input features, y = target (0 = normal, 1 = fraud)
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    # Same settings as Stage 3, so the split is identical to the notebooks
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )