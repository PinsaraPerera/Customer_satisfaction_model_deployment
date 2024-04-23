import logging
import pickle
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e

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
