import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from ..model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin


@step
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, Y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rmse"],
]:
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(Y_test, prediction)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(Y_test, prediction)
        r2_score = R2Score()
        r2 = r2_score.calculate_score(Y_test, prediction)
        return r2, rmse
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        raise e
