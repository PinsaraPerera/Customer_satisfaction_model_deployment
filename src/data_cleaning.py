import logging 
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    '''
    Abstract class for data strategies.
    '''
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    '''
    Data pre-processing strategy.
    '''
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Pre-process the data.
        
        Args:
            data: pd.DataFrame: The data to pre-process.
        Returns:
            pd.DataFrame: The pre-processed data.
        '''
        
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_purchase_timestamp",
            ], axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data

        except Exception as e:
            logging.error(f"Error while pre-processing data: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    '''
    Data divide into train and test split.
    '''
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        '''
        Divide the data into features and target.
        
        Args:
            data: pd.DataFrame: The data to divide.
        Returns:
            Union[pd.DataFrame, pd.Series]: The features and target.
        '''
        
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error while dividing data: {e}")
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        '''
        Args:
            data: pd.DataFrame: The data to clean.
            strategy: DataStrategy: The strategy to use for cleaning.
        '''
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        '''
        Handle the data.
        '''
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error while handling data: {e}")
            raise e

