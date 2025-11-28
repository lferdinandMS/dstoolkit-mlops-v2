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


def _check_role_assignment(principal_id, scope, role_name):
    """Check if the principal has the specified role on the given scope."""
    try:
        cmd = [
            "az",
            "role",
            "assignment",
            "list",
            "--assignee",
            principal_id,
            "--scope",
            scope,
            "--role",
            role_name,
            "--output",
            "json",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        assignments = json.loads(result.stdout)
        return len(assignments) > 0
    except Exception as e:
        print(f"Warning: Could not check role assignment: {e}")
        return False


def _ensure_role_assignment(principal_id, role_name, scope, wait_seconds=120):
    """Ensure the principal holds the specified role, assigning if necessary."""
    print(f"Assigning '{role_name}' role to principal {principal_id}")
    print(f"Scope: {scope}")

    if _check_role_assignment(principal_id, scope, role_name):
        print(f"{role_name} assignment already exists and is confirmed.")
        return

    cmd = [
        "az",
        "role",
        "assignment",
        "create",
        "--assignee",
        principal_id,
        "--role",
        role_name,
        "--scope",
        scope,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{role_name} assignment successful.")
        if result.stdout:
            print(result.stdout)
        if wait_seconds:
            print(f"Waiting {wait_seconds} seconds for RBAC propagation...")
            time.sleep(wait_seconds)
    except subprocess.CalledProcessError as e:
        if "RoleAssignmentExists" in e.stderr:
            print(f"{role_name} assignment already exists - this is OK.")
            if wait_seconds:
                print("Waiting 30 seconds to ensure RBAC is fully propagated...")
                time.sleep(30)
        else:
            raise Exception(f"Failed to assign {role_name} role: {e.stderr}")


def _assign_storage_role(client, workspace_name, compute_object):
    """Assign Storage Blob Data Contributor role to the compute identity."""
    if not (compute_object.identity and compute_object.identity.principal_id):
        error_msg = (
            f"ERROR: Compute '{compute_object.name}' does not have a managed identity with principal_id. "
            "Cannot assign storage role. This will cause authentication failures during batch deployments."
        )
        print(error_msg)
        raise ValueError(error_msg)

    print(f"Ensuring RBAC for compute identity {compute_object.identity.principal_id}...")
    try:
        ws = client.workspaces.get(workspace_name)
        storage_id = ws.storage_account

        storage_scope = storage_id

        # Assign storage-related roles needed for batch orchestration
        principal_id = compute_object.identity.principal_id
        _ensure_role_assignment(principal_id, "Storage Blob Data Contributor", storage_scope)
        _ensure_role_assignment(principal_id, "Storage Table Data Contributor", storage_scope, wait_seconds=30)
        _ensure_role_assignment(principal_id, "Storage Queue Data Contributor", storage_scope, wait_seconds=30)

        # Assign AzureML Data Scientist role on workspace for model access
        workspace_id = (
            f"/subscriptions/{client.subscription_id}/"
            f"resourceGroups/{client.resource_group_name}/"
            "providers/Microsoft.MachineLearningServices/workspaces/"
            f"{workspace_name}"
        )
        print("")
        _ensure_role_assignment(principal_id, "AzureML Data Scientist", workspace_id)

    except Exception as e:
        print(f"ERROR: Could not assign roles: {e}")
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
        if not compute_object.identity or not compute_object.identity.principal_id:
            print(f"Enabling SystemAssigned identity for {cluster_name}...")
            compute_object.identity = IdentityConfiguration(type="SystemAssigned")
            compute_object = client.compute.begin_create_or_update(compute_object).result()
            print(f"Identity enabled for {cluster_name}.")

            # Wait a moment for identity to be fully provisioned
            print("Waiting 30 seconds for identity provisioning...")
            time.sleep(30)

            # Refresh compute object to get the principal_id
            compute_object = client.compute.get(cluster_name)
            if compute_object.identity and compute_object.identity.principal_id:
                print(f"Identity principal ID: {compute_object.identity.principal_id}")
            else:
                print("WARNING: Identity was enabled but principal_id is not yet available.")
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

        # Workspace-level RBAC is now handled during infrastructure provisioning
        # to avoid conflicting role assignments.
        print("Workspace-level RBAC is managed by infrastructure provisioning; skipping inline setup.")

        return compute_object

    except Exception as ex:
        print(
            "An error occurred while trying to create or update the Azure ML environment. "
            "Please check your credentials, subscription details, and workspace configuration, and try again. "
            f"Error details: {ex}"
        )
        raise
