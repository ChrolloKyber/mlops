import logging
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from zenml import step


@step
def evaluate_model(
    model: LinearRegression,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate the model's performance."""
    try:
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        logging.info(
            f"Training metrics: MSE = {train_mse:.4f}, RMSE = {
                train_rmse:.4f}, R² = {train_r2:.4f}"
        )
        logging.info(
            f"Testing metrics: MSE = {test_mse:.4f}, RMSE = {
                test_rmse:.4f}, R² = {test_r2:.4f}"
        )

        # Return evaluation metrics
        metrics = {
            "train_mse": float(train_mse),
            "train_rmse": float(train_rmse),
            "train_r2": float(train_r2),
            "test_mse": float(test_mse),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        }

        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e
