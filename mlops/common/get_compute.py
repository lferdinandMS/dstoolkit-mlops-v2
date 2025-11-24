"""
This module provides functionality for managing Azure Machine Learning compute resources.

It includes functions to get an existing compute target or create a new one in an Azure
Machine Learning workspace. The module uses the Azure Machine Learning SDK for Python to
interact with Azure resources. It is designed to be run as a standalone script with command-line
arguments for specifying the details of the Azure Machine Learning compute target.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, IdentityConfiguration
import subprocess
import time
import json


def _check_role_assignment(principal_id, storage_id):
    """Check if the principal has Storage Blob Data Contributor role on the storage account."""
    try:
        cmd = [
            "az",
            "role",
            "assignment",
            "list",
            "--assignee",
            principal_id,
            "--scope",
            storage_id,
            "--role",
            "Storage Blob Data Contributor",
            "--output",
            "json",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assignments = json.loads(result.stdout)
        return len(assignments) > 0
    except Exception as e:
        print(f"Warning: Could not check role assignment: {e}")
        return False


def _assign_storage_role(client, workspace_name, compute_object):
    """Assign Storage Blob Data Contributor role to the compute identity."""
    if not (compute_object.identity and compute_object.identity.principal_id):
        print("WARNING: Compute does not have a managed identity. Cannot assign storage role.")
        return

    print(f"Ensuring RBAC for compute identity {compute_object.identity.principal_id}...")
    try:
        ws = client.workspaces.get(workspace_name)
        storage_id = ws.storage_account

        print(f"Assigning 'Storage Blob Data Contributor' role to principal {compute_object.identity.principal_id}")
        print(f"Storage account scope: {storage_id}")

        # Check if role already exists
        if _check_role_assignment(compute_object.identity.principal_id, storage_id):
            print("Role assignment already exists and is confirmed - skipping assignment.")
            return

        cmd = [
            "az",
            "role",
            "assignment",
            "create",
            "--assignee",
            compute_object.identity.principal_id,
            "--role",
            "Storage Blob Data Contributor",
            "--scope",
            storage_id,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Role assignment successful.")
        print(f"Output: {result.stdout}")
        print("Waiting 120 seconds for RBAC role assignment to propagate...")
        time.sleep(120)
        print("RBAC propagation wait complete.")
        
        # Verify the role assignment
        if _check_role_assignment(compute_object.identity.principal_id, storage_id):
            print("Role assignment verified successfully.")
        else:
            print("WARNING: Role assignment could not be verified. May need more time to propagate.")
    except subprocess.CalledProcessError as e:
        if "RoleAssignmentExists" in e.stderr:
            print("Role assignment already exists - this is OK.")
            print("Waiting 30 seconds to ensure RBAC is fully propagated...")
            time.sleep(30)
        else:
            print(f"ERROR: Failed to assign role: {e.stderr}")
            print(f"Command output: {e.stdout}")
            raise Exception(f"Failed to assign Storage Blob Data Contributor role: {e.stderr}")
    except Exception as e:
        print(f"ERROR: Could not assign role: {e}")
        raise


def _get_or_create_compute_target(
    client,
    cluster_name,
    cluster_size,
    cluster_region,
    min_instances,
    max_instances,
    idle_time_before_scale_down,
):
    """Get existing compute or create new one."""
    try:
        compute_object = client.compute.get(cluster_name)
        print(f"Found existing compute target {cluster_name}, so using it.")
        # Ensure identity is enabled even for existing clusters
        if not compute_object.identity:
            print(f"Enabling SystemAssigned identity for {cluster_name}...")
            compute_object.identity = IdentityConfiguration(type="SystemAssigned")
            client.compute.begin_create_or_update(compute_object).result()
            print(f"Identity enabled for {cluster_name}.")
        return compute_object
    except Exception:
        print(f"{cluster_name} is not found! Trying to create a new one.")
        compute_object = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size=cluster_size,
            location=cluster_region,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time_before_scale_down,
            identity=IdentityConfiguration(type="SystemAssigned"),
        )
        return client.compute.begin_create_or_update(compute_object).result()


def get_compute(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    cluster_name: str,
    cluster_size: str,
    cluster_region: str,
    min_instances: int = 0,
    max_instances: int = 4,
    idle_time_before_scale_down: int = 600,
):
    """Get an existing compute or create a new one."""
    try:
        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        compute_object = _get_or_create_compute_target(
            client,
            cluster_name,
            cluster_size,
            cluster_region,
            min_instances,
            max_instances,
            idle_time_before_scale_down,
        )

        _assign_storage_role(client, workspace_name, compute_object)

        return compute_object

    except Exception as ex:
        print(
            "An error occurred while trying to create or update the Azure ML environment. "
            "Please check your credentials, subscription details, and workspace configuration, and try again. "
            f"Error details: {ex}"
        )
        raise
