import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divides it into train and test split.

    Args:
        df: pd.DataFrame: The data to clean and divide.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The cleaned and divided data.
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        process_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning and division complete.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e
