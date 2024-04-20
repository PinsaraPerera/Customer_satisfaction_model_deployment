import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    '''
    Abstract class for model development.
    '''

    @abstractmethod
    def train(self, X_train, y_train):
        '''
        Train the model.
        Args:
            X_train: pd.DataFrame: The training data.
            y_train: pd.Series: The target data.
        Returns:
            None
        '''
        pass

class LinearRegressionModel(Model):
    '''
    Linear Regression model.
    '''

    def train(self, X_train, y_train, **kwargs):
        '''
        Train the model.
        Args:
            X_train: pd.DataFrame: The training data.
            y_train: pd.Series: The target data.
        Returns:
            None
        '''
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training complete.")
            return model
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e