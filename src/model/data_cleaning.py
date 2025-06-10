import logging
from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        pass


class DataPreProcessingStrategy(DataStrategy):
    @override
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "review_creation_date",
                    "review_creation_date",
                    "review_answer_timestamp",
                    "order_id",
                ],
                axis=1,
            )
            data["review_score"].fillna(data["review_score"].median(), inplace=True)
            data["review_comment_title"].fillna("No review", inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    @override
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        try:
            X = data.drop(["review_score"], axis=1)
            Y = data["review_score"]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.strategy = strategy
        self.data = data

    def handle_data(self) -> None:
        try:
            return self.strategy.handle_data(data=self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
