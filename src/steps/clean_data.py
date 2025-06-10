import logging

import pandas as pd
from zenml import step

from src.model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreProcessingStrategy,
)
from typing import Tuple
from typing_extensions import Annotated


@step
def clean_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"],
]:
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, Y_train, Y_test = data_cleaning.handle_data()
        logging.info("Data cleanup completed")
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error(f"Error cleaning up data: {e}")
        raise e
