import logging
import pandas as pd 
from zenml import step 

@step
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data.
    
    Args:
        df: pd.DataFrame: The data to clean.
    Returns:
        pd.DataFrame: The cleaned data.
    '''
    
    logging.info("Cleaning the data")
    # Dropping rows with missing values
    df = df.dropna()
    return df