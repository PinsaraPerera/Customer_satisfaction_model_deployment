import logging
import pickle
import pandas as pd
from zenml import step
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

class LoadModel:
    '''Class to load model from a specified path.'''
    def __init__(self, model_path: str):
        '''
        Args:
            model_path: Path to the pickled model file.
        '''
        self.model_path = model_path

    def load(self):
        '''Load the model from the file.'''
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error("Failed to load model: {}".format(e))
            raise

@step
def load_model_step(data_path: str = "saved_model/model.pickle") -> RegressorMixin:
    '''ZenML step to load a model from a given path.'''
    loader = LoadModel(data_path)
    return loader.load()


