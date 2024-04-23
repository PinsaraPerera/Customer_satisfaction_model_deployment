from zenml import pipeline
from zenml.config.schedule import Schedule
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from .utils import store_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def train_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df=df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model=model, x_test=X_test, y_test=y_test)

    

# def run_train_pipeline(data_path: str, scheduler=None):
#     train_pipeline.with_options(data_path=data_path, schedule=scheduler)
