"""
This module is designed to register data assets in an Azure Machine Learning environment.

It utilizes the Azure AI MLClient from the Azure Machine Learning SDK to interact with Azure resources.

The script reads a configuration file to identify and register datasets in Azure Machine Learning.
It supports operations like creating or updating
data assets and retrieving the latest version of these assets.
"""
import argparse
import json
import os
from azure.identity import DefaultAzureCredential, ClientAssertionCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from mlops.common.config_utils import MLOpsConfig


def get_token():
    """Read the OIDC token from the file."""
    token_file = os.getenv("AZURE_FEDERATED_TOKEN_FILE")
    if token_file and os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()
    return None


def main():
    """Register all datasets from the config file."""
    # Enable identity-based datastore authentication
    os.environ["AZURE_STORAGE_AUTH_MODE"] = "login"

    config = MLOpsConfig()

    # Use workload identity if available, otherwise fall back to DefaultAzureCredential
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")

    if tenant_id and client_id and os.getenv("AZURE_FEDERATED_TOKEN_FILE"):
        credential = ClientAssertionCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            func=get_token,
        )
    else:
        credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential,
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
    )

    parser = argparse.ArgumentParser("register data assets")

    parser.add_argument(
        "--data_config_path", type=str, help="data config file path", required=True
    )

    args = parser.parse_args()

    data_config_path = args.data_config_path

    config_file = open(data_config_path)
    data_config = json.load(config_file)

    for elem in data_config["datasets"]:
        data_path = elem["DATA_PATH"]
        dataset_desc = elem["DATASET_DESC"]
        dataset_name = elem["DATASET_NAME"]

        aml_dataset = Data(
            path=data_path,
            type=AssetTypes.URI_FOLDER,
            description=dataset_desc,
            name=dataset_name,
        )

        # Use identity-based authentication by setting datastore credential
        ml_client.data.create_or_update(aml_dataset)

        aml_dataset_unlabeled = ml_client.data.get(
            name=dataset_name, label="latest"
        )

        print(aml_dataset_unlabeled.id)


if __name__ == "__main__":
    main()
