from zenml import pipeline
from steps.load_model import load_model_step
from rich import print



@pipeline(enable_cache=False)
def prediction_pipeline():
    model = load_model_step()
    print("this is model prediction")

