import logging
import pandas as pd
import numpy as np
from typing import Tuple

from zenml import step


@step
def model_train(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Create features and split into X and y."""
    try:
        # Select features for the model
        features = [
            "response_time_hours",
            "creation_month",
            "creation_day",
            "creation_dayofweek",
            "has_comment",
            "comment_length",
        ]

        X = df[features].values
        y = df["review_score"].values

        logging.info(f"Selected features: {features}")
        logging.info(f"Feature matrix shape: {
                     X.shape}, Target vector shape: {y.shape}")

        # Keep feature names for later use
        feature_names = features

        return X, y
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        raise e
