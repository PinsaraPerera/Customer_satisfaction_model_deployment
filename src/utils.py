import logging
import pickle
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy


def store_model(model, filename="saved_model/model.pickle"):
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
            logging.info("Model saved successfully")
    except Exception as e:
        logging.error("model cannot save as pickle")
        raise e


def load_model(filename="saved_model/model.pickle"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
            logging.info("Model loaded successfully")
    except Exception as e:
        logging.error("model loading failed")
        raise e
