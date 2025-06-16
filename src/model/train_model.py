import logging
from sklearn.linear_model import LinearRegression
import numpy as np
from zenml import step


@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    try:
        """Train a linear regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e
