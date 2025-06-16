import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from zenml import step


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the data for modeling."""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Convert date columns to datetime objects
    df_clean["review_creation_date"] = pd.to_datetime(
        df_clean["review_creation_date"], errors="coerce"
    )
    df_clean["review_answer_timestamp"] = pd.to_datetime(
        df_clean["review_answer_timestamp"], errors="coerce"
    )

    # Calculate response time in hours
    df_clean["response_time_hours"] = (
        df_clean["review_answer_timestamp"] - df_clean["review_creation_date"]
    ).dt.total_seconds() / 3600

    # Extract features from dates
    df_clean["creation_month"] = df_clean["review_creation_date"].dt.month
    df_clean["creation_day"] = df_clean["review_creation_date"].dt.day
    df_clean["creation_dayofweek"] = df_clean["review_creation_date"].dt.dayofweek

    # Handle missing comment data
    df_clean["has_comment"] = df_clean["review_comment_message"].notna().astype(int)
    df_clean["comment_length"] = (
        df_clean["review_comment_message"].fillna("").apply(len)
    )

    # Drop rows with missing values in critical columns
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=["review_score", "response_time_hours"])
    dropped_rows = initial_rows - df_clean.shape[0]
    print(
        f"Dropped {dropped_rows} rows with missing values in critical columns")

    print(f"Data cleaned successfully. Shape: {df_clean.shape}")
    return df_clean
