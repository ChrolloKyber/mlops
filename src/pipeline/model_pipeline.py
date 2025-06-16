from zenml.pipelines import pipeline

from ..steps.ingest_data import ingest_data
from ..steps.clean_data import clean_data
from ..steps.model_train import model_train
from ..steps.model_evaluate import evaluate_model
from ..model.train_model import train_model
from ..model.data_cleaning import train_test_split_step
from ..model.save_model import save_model


@pipeline(enable_cache=False)
def review_score_prediction_pipeline(data_path: str):
    """ZenML pipeline for review score prediction."""
    df = ingest_data(data_path)
    clean_df = clean_data(df)
    X, y = model_train(clean_df)
    X_train, X_test, y_train, y_test = train_test_split_step(X, y)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    save_model(model)

    return model, metrics
