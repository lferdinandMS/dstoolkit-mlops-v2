"""
This module is designed to train a machine learning model.

The module performs the following key steps:
1. Reading and combining data from specified training data files.
2. Splitting the combined data into training and testing datasets.
3. Training a Linear Regression model using the training dataset.
4. Using MLflow for logging and tracking experiments.
5. Saving the trained model and its metadata to specified paths.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import json

import pkg_resources, mlflow
print("mlflow:", mlflow.__version__)
try:
    print("azureml-mlflow:", pkg_resources.get_distribution("azureml-mlflow").version)
except Exception as e:
    print("azureml-mlflow not found or from image:", e)

def main(training_data, test_data, model_output, model_metadata):
    """
    Read training data, split data and initiate training.

    Parameters:
      training_data (str): training data folder
      test_data (str): test data folder
      model_output (str): a folder to store model files
      model_metadata (str): a file to store information about thr model
    """
    print("Hello training world...")

    lines = [
        f"Training data path: {training_data}",
        f"Test data path: {test_data}",
        f"Model output path: {model_output}",
        f"Model metadata path: {model_metadata}",
    ]

    for line in lines:
        print(line)

    print("mounted_path files: ")
    arr = os.listdir(training_data)
    print(arr)

    df_list = []
    for filename in arr:
        print("reading file: %s ..." % filename)
        input_df = pd.read_csv((Path(training_data) / filename))
        df_list.append(input_df)

    train_data = df_list[0]
    print(train_data.columns)

    train_x, test_x, trainy, testy = split(train_data)
    write_test_data(test_x, testy, test_data)
    train_model(train_x, trainy, model_output, model_metadata)


def split(train_data):
    """
    Split the input data into training and testing sets.

    Parameters:
    train_data (DataFrame): The input data.

    Returns:
    trainX (DataFrame): The training data.
    testX (DataFrame): The testing data.
    trainy (Series): The training labels.
    testy (Series): The testing labels.
    """
    # Split the data into input(X) and output(y)
    y = train_data["cost"]
    x = train_data[
        [
            "distance",
            "dropoff_latitude",
            "dropoff_longitude",
            "passengers",
            "pickup_latitude",
            "pickup_longitude",
            "store_forward",
            "vendor",
            "pickup_weekday",
            "pickup_month",
            "pickup_monthday",
            "pickup_hour",
            "pickup_minute",
            "pickup_second",
            "dropoff_weekday",
            "dropoff_month",
            "dropoff_monthday",
            "dropoff_hour",
            "dropoff_minute",
            "dropoff_second",
        ]
    ]

    # Split the data into train and test sets
    train_x, test_x, trainy, testy = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    print(train_x.shape)
    print(train_x.columns)

    return train_x, test_x, trainy, testy


def train_model(train_x, trainy, model_output, model_metadata):
    """
    Train a Linear Regression model and save the model and its metadata.

    Parameters:
    trainX (DataFrame): The training data.
    trainy (Series): The training labels.

    Returns:
    None
    """
    mlflow.autolog()
    # Train a Linear Regression Model with the train set
    with mlflow.start_run() as run:
        model = LinearRegression().fit(train_x, trainy)
        print(model.score(train_x, trainy))

        # Log the model to MLflow under the standard artifact path "model"
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Persist run metadata for the register step
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_data = {"run_id": run_id, "run_uri": model_uri}
        with open(model_metadata, "w") as json_file:
            json.dump(model_data, json_file, indent=4)

        # Optionally also dump a pickle alongside for local debugging
        pickle.dump(model, open((Path(model_output) / "model.sav"), "wb"))


def write_test_data(test_x, testy, test_data_path):
    """
    Write the testing data to a CSV file.

    Parameters:
    testX (DataFrame): The testing data.
    testy (Series): The testing labels.

    Returns:
    None
    """
    test_x["cost"] = testy
    print(test_x.shape)
    test_x.to_csv((Path(test_data_path) / "test_data.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--model_metadata", type=str, help="Path of model metadata")

    args = parser.parse_args()

    training_data = args.training_data
    test_data = args.test_data
    model_output = args.model_output
    model_metadata = args.model_metadata

    main(training_data, test_data, model_output, model_metadata)
