import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple


@step
def evaluate_model(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    '''
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    '''
    
    try:
        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_scores(y_test, prediction)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        
        return r2_score, rmse 

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e

