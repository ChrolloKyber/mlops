from sklearn.linear_model import LinearRegression
import numpy as np
from zenml import step


@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model
