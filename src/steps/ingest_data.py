import logging
import pandas as pd
from zenml import step


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Ingests the review data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {data_path}")
        raise e
