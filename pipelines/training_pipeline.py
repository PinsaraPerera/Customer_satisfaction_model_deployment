from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df=df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model=model, x_test=X_test, y_test=y_test)
