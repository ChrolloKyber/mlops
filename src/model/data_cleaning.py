import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from zenml import step


@step
def train_test_split_step(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {
          X_train.shape}, Testing data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
