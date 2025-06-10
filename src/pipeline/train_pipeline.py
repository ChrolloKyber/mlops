from zenml import pipeline
from ..steps.ingest_data import ingest_data
from ..steps.clean_data import clean_data
from ..steps.model_train import train_model
from ..steps.model_evaluate import evaluate_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, Y_train, Y_test = clean_data(df)
    model = train_model(X_train, X_test, Y_train, Y_test)
    r2_score, rmse = evaluate_model(model, X_test, Y_test)
