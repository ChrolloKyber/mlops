import logging
from sklearn.linear_model import LinearRegression
from zenml import step


@step
def save_model(model: LinearRegression) -> None:
    try:
        """Save the trained model to a file."""
        import pickle
        path = "dist/review_score_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
        return
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        raise e
