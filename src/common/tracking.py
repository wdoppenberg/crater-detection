import mlflow


def setup(uri="http://localhost:5000/", experiment_name="crater-detection"):
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
    except ConnectionRefusedError as err:
        print("MLflow server could not be found:\n", err)
