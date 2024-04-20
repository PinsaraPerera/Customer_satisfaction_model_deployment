import logging
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train the model.
    Args:
        X_train: pd.DataFrame: The training data.
        X_test: pd.DataFrame: The testing data.
        y_train: pd.Series: The target data.
        y_test: pd.Series: The target data.
    Returns:
        model: RegressorMixin
    """

    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info("Model training complete.")
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e
