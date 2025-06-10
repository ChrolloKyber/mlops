import logging
import pandas as pd
from zenml import step
from ..model.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> RegressorMixin:
    try:
        trained_model = LinearRegressionModel.train(X_train, Y_train)
        logging.info("Model successfully trained")
        return trained_model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e
