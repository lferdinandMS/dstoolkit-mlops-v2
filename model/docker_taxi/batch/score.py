"""This module provides the functionality for initializing and running a machine learning model."""
import os
import joblib
import pandas as pd
from typing import List
import subprocess
import json


def init():
    """
    Initialize the service instance on startup.

    You can write the logic here to perform init operations like caching the model in memory.
    """
    global model

    # Log identity information for diagnostics
    print("="*80)
    print("BATCH SCORING IDENTITY DIAGNOSTICS")
    print("="*80)
    
    try:
        # Get the identity token metadata to see which identity is being used
        result = subprocess.run(
            ["curl", "-H", "Metadata:true", "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            token_info = json.loads(result.stdout)
            print(f"Running with managed identity - Client ID: {token_info.get('client_id', 'N/A')}")
            print(f"Resource: {token_info.get('resource', 'N/A')}")
        else:
            print("Could not retrieve identity token metadata")
    except Exception as e:
        print(f"Error checking identity: {e}")
    
    # Log environment info
    print(f"AZUREML_MODEL_DIR: {os.getenv('AZUREML_MODEL_DIR', 'Not set')}")
    print(f"AZUREML_RUN_ID: {os.getenv('AZUREML_RUN_ID', 'Not set')}")
    print(f"Working directory: {os.getcwd()}")
    print("="*80)

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model", "model.pkl")

    # deserialize the model file back into a sklearn model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Init complete")


def run(mini_batch: List[str]) -> pd.DataFrame:
    """
    Execure inferencing logic on a request.

    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back.
    """
    
    import os
    print(os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"))

    results = []

    print("Request received")

    for raw_data in mini_batch:
        print(f"File name: {raw_data}")
        data = pd.read_csv(raw_data)

        result = model.predict(data.to_numpy())
        print(f"predicted results: {result}")

        print("Item has been proccessed")

        # You need to implement a better way to combine results from the model depends on your desired output
        results.append("Item has been processed")

    return pd.DataFrame(results)
