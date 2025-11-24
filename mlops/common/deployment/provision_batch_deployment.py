"""
This script automates the deployment of machine learning models in Azure Machine Learning.

It supports both batch deployment scenario.
"""
import argparse
import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    BatchRetrySettings,
    CodeConfiguration,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.core.exceptions import ResourceExistsError
from mlops.common.config_utils import MLOpsConfig
from mlops.common.naming_utils import generate_model_name
from mlops.common.get_compute import get_compute


def wait_for_endpoint_ready(ml_client, endpoint_name, max_wait=600):
    """Wait for endpoint to be ready for operations."""
    print(f"Checking if endpoint {endpoint_name} is ready for operations...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            endpoint = ml_client.batch_endpoints.get(endpoint_name)
            print(f"Endpoint state: {endpoint.provisioning_state}")
            
            if endpoint.provisioning_state == "Succeeded":
                print(f"Endpoint {endpoint_name} is ready")
                return True
            elif endpoint.provisioning_state in ["Failed", "Canceled"]:
                raise Exception(f"Endpoint in {endpoint.provisioning_state} state")
            else:
                print(f"Endpoint still provisioning ({endpoint.provisioning_state}). Waiting 30 seconds...")
                time.sleep(30)
        except Exception as e:
            if "not found" in str(e).lower() or "ResourceNotFound" in str(e):
                print(f"Endpoint {endpoint_name} does not exist yet - ready to create")
                return True
            raise
    
    raise TimeoutError(f"Endpoint not ready after {max_wait} seconds")


def deploy_with_retry(ml_client, deployment, max_retries=3, initial_delay=60):
    """Deploy with retry logic for concurrent operation conflicts."""
    for attempt in range(max_retries):
        try:
            print(f"Deployment attempt {attempt + 1}/{max_retries}...")
            poller = ml_client.begin_create_or_update(deployment)
            result = poller.result()
            print("Deployment completed successfully")
            return result
        except ResourceExistsError as e:
            if "Already running method" in str(e) and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"Conflict detected: Another operation is in progress.")
                print(f"Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                time.sleep(delay)
            else:
                print(f"Deployment failed after {attempt + 1} attempts")
                raise
        except Exception as e:
            print(f"Unexpected error during deployment: {str(e)}")
            raise
    
    raise Exception("Deployment failed after all retry attempts")


def main():
    """Automate the deployment of machine learning models in Azure Machine Learning."""
    parser = argparse.ArgumentParser("provision_deployment")
    parser.add_argument(
        "--model_type", type=str, help="registered model type to be deployed", required=True
    )
    parser.add_argument(
        "--environment_name",
        type=str,
        help="env name (dev, test, prod) for deployment",
        required=True,
    )
    parser.add_argument(
        "--run_id", type=str, help="AML run id for model generation", required=True
    )
    args = parser.parse_args()

    model_type = args.model_type
    run_id = args.run_id
    env_type = args.environment_name

    config = MLOpsConfig(environment=env_type)

    ml_client = MLClient(
        DefaultAzureCredential(),
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
    )

    deployment_config = config.get_deployment_config(deployment_name=f"{model_type}_batch")

    published_model_name = generate_model_name(model_type)
    
    print(f"Looking for model: {published_model_name}")

    try:
        model_refs = ml_client.models.list(published_model_name)
        model_list = list(model_refs)
        
        if not model_list:
            print(f"ERROR: No models found with name '{published_model_name}'")
            print("Available models:")
            for model in ml_client.models.list():
                print(f"  - {model.name} (version {model.version})")
            raise ValueError(f"Model '{published_model_name}' not found. Please check model name and ensure training completed successfully.")
        
        latest_version = max(model.version for model in model_list)
        print(f"Found model version: {latest_version}")
        model = ml_client.models.get(published_model_name, latest_version)
    except Exception as e:
        print(f"Error retrieving model '{published_model_name}': {str(e)}")
        raise

    get_compute(
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
        deployment_config["batch_cluster_name"],
        deployment_config["batch_cluster_size"],
        deployment_config["batch_cluster_region"],
    )

    environment = Environment(
        name="prs-env",
        conda_file=deployment_config["deployment_conda_path"],
        image=deployment_config["deployment_base_image"],
    )

    deployment = ModelBatchDeployment(
        name=deployment_config["deployment_name"],
        description="model with batch endpoint",
        endpoint_name=deployment_config["endpoint_name"],
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=deployment_config["score_dir"], scoring_script=deployment_config["score_file_name"]
        ),
        compute=deployment_config["batch_cluster_name"],
        settings=ModelBatchDeploymentSettings(
            instance_count=deployment_config["cluster_instance_count"],
            max_concurrency_per_instance=deployment_config["max_concurrency_per_instance"],
            mini_batch_size=deployment_config["mini_batch_size"],
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name=deployment_config["output_file_name"],
            retry_settings=BatchRetrySettings(
                max_retries=deployment_config["max_retries"],
                timeout=deployment_config["retry_timeout"],
            ),
            logging_level="info",
        ),
        tags={
            "build_id": config.environment_configuration["build_reference"],
            "run_id": run_id,
        },
    )

    # Wait for endpoint to be ready before deploying
    wait_for_endpoint_ready(ml_client, deployment_config["endpoint_name"])
    
    # Deploy with retry logic
    deploy_with_retry(ml_client, deployment)

    # Update default deployment with retry logic
    print("Updating default deployment...")
    endpoint = ml_client.batch_endpoints.get(deployment_config["endpoint_name"])
    endpoint.defaults.deployment_name = deployment.name
    deploy_with_retry(ml_client, endpoint)
    print(f"The default deployment is {endpoint.defaults.deployment_name}")


if __name__ == "__main__":
    main()
