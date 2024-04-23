import click
from pipelines.training_pipeline import train_pipeline
from pipelines.prediction_pipeline import prediction_pipeline
from zenml.config.schedule import Schedule
from zenml.client import Client
from typing import cast

TRAIN = "train"
PREDICT = "predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([TRAIN, PREDICT]),
    default=TRAIN,
    help="Optionally you can choose to only run the training "
    "pipeline to train a model (`train`), or to "
    "only run a prediction against the trained model "
    "(`predict`). By default training pipeline will be run "
    "(`deploy_and_predict`).",
)

def run_pipeline(config: str):
    train_model = config == TRAIN
    predict = config == PREDICT

    if train_model:

        # scheduler = Schedule(cron_expression='0 0 * * *')
        scheduler = Schedule(cron_expression="* * * * *")
        sheduled_pipeline = train_pipeline.with_options(schedule=scheduler)

        sheduled_pipeline(data_path='data/olist_customers_dataset.csv')

        print(Client().active_stack.experiment_tracker.get_tracking_uri())

    if predict:
        prediction_pipeline()


if __name__ == '__main__':
    run_pipeline()
    

    
    