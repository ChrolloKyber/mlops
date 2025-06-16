from sklearn.linear_model import LinearRegression
from zenml import step


@step
def save_model(model: LinearRegression) -> None:
    """Save the trained model to a file."""
    import pickle
    with open("review_score_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to 'review_score_model.pkl'")
    return
