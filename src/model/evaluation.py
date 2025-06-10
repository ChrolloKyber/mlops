import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Square Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse

        except Exception as e:
            logging.error(f"Failed to calculate Mean Square Error score: {e}")
            raise e


class R2Score(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"MSE: {r2}")
            return r2

        except Exception as e:
            logging.error(f"Failed to calculate R2 score: {e}")
            raise e


class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Square Error")
            mse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"Root MSE: {mse}")
            return mse

        except Exception as e:
            logging.error(f"Failed to calculate Root Mean Square Error score: {e}")
            raise e
