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

        # Assign Storage Blob Data Contributor role on storage account
        print(f"Assigning 'Storage Blob Data Contributor' role to principal {compute_object.identity.principal_id}")
        print(f"Storage account scope: {storage_id}")

        if _check_role_assignment(compute_object.identity.principal_id, storage_id, "Storage Blob Data Contributor"):
            print("Storage role assignment already exists and is confirmed.")
        else:
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
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("Storage role assignment successful.")
                print(f"Output: {result.stdout}")
                print("Waiting 120 seconds for RBAC role assignment to propagate...")
                time.sleep(120)
                
                if _check_role_assignment(compute_object.identity.principal_id, storage_id, "Storage Blob Data Contributor"):
                    print("Storage role assignment verified successfully.")
                else:
                    print("WARNING: Storage role assignment could not be verified. May need more time to propagate.")
            except subprocess.CalledProcessError as e:
                if "RoleAssignmentExists" in e.stderr:
                    print("Storage role assignment already exists - this is OK.")
                    print("Waiting 30 seconds to ensure RBAC is fully propagated...")
                    time.sleep(30)
                else:
                    print(f"ERROR: Failed to assign storage role: {e.stderr}")
                    raise Exception(f"Failed to assign Storage Blob Data Contributor role: {e.stderr}")

        # Assign AzureML Data Scientist role on workspace for model access
        workspace_id = f"/subscriptions/{client.subscription_id}/resourceGroups/{client.resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}"
        print(f"\nAssigning 'AzureML Data Scientist' role to principal {compute_object.identity.principal_id}")
        print(f"Workspace scope: {workspace_id}")
        
        if _check_role_assignment(compute_object.identity.principal_id, workspace_id, "AzureML Data Scientist"):
            print("Workspace role assignment already exists and is confirmed.")
        else:
            cmd_workspace = [
                "az",
                "role",
                "assignment",
                "create",
                "--assignee",
                compute_object.identity.principal_id,
                "--role",
                "AzureML Data Scientist",
                "--scope",
                workspace_id,
            ]
            try:
                result = subprocess.run(cmd_workspace, check=True, capture_output=True, text=True)
                print("Workspace role assignment successful.")
                print(f"Output: {result.stdout}")
                print("Waiting 120 seconds for workspace RBAC to propagate...")
                time.sleep(120)
                
                if _check_role_assignment(compute_object.identity.principal_id, workspace_id, "AzureML Data Scientist"):
                    print("Workspace role assignment verified successfully.")
                else:
                    print("WARNING: Workspace role assignment could not be verified. May need more time to propagate.")
            except subprocess.CalledProcessError as e:
                if "RoleAssignmentExists" in e.stderr:
                    print("Workspace role assignment already exists - this is OK.")
                    print("Waiting 30 seconds to ensure RBAC is fully propagated...")
                    time.sleep(30)
                else:
                    print(f"ERROR: Failed to assign workspace role: {e.stderr}")
                    raise Exception(f"Failed to assign AzureML Data Scientist role: {e.stderr}")
                    
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
        
        # Also assign storage role to workspace identity (used by batch jobs)
        print("\nEnsuring workspace identity has storage access...")
        ws = client.workspaces.get(workspace_name)
        if ws.identity and ws.identity.principal_id:
            print(f"Workspace identity principal ID: {ws.identity.principal_id}")
            storage_id = ws.storage_account
            
            if _check_role_assignment(ws.identity.principal_id, storage_id, "Storage Blob Data Contributor"):
                print("Workspace storage role assignment already exists and is confirmed.")
            else:
                print(f"Assigning 'Storage Blob Data Contributor' role to workspace identity {ws.identity.principal_id}")
                cmd = [
                    "az",
                    "role",
                    "assignment",
                    "create",
                    "--assignee",
                    ws.identity.principal_id,
                    "--role",
                    "Storage Blob Data Contributor",
                    "--scope",
                    storage_id,
                ]
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print("Workspace storage role assignment successful.")
                    print("Waiting 120 seconds for RBAC to propagate...")
                    time.sleep(120)
                    
                    if _check_role_assignment(ws.identity.principal_id, storage_id, "Storage Blob Data Contributor"):
                        print("Workspace storage role assignment verified successfully.")
                    else:
                        print("WARNING: Workspace storage role assignment could not be verified.")
                except subprocess.CalledProcessError as e:
                    if "RoleAssignmentExists" in e.stderr:
                        print("Workspace storage role assignment already exists - this is OK.")
                    else:
                        print(f"ERROR: Failed to assign workspace storage role: {e.stderr}")
                        raise Exception(f"Failed to assign Storage Blob Data Contributor role to workspace: {e.stderr}")
        else:
            print("WARNING: Workspace does not have a managed identity!")

        return compute_object

    except Exception as ex:
        print(
            "An error occurred while trying to create or update the Azure ML environment. "
            "Please check your credentials, subscription details, and workspace configuration, and try again. "
            f"Error details: {ex}"
        )
        raise
