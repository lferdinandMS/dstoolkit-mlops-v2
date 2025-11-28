data "azurerm_client_config" "current" {} 
resource "azurerm_resource_group" "rg" {
  location            = var.location
  name = var.rg_name
  
}

resource "azurerm_application_insights" "aml_appins" {
  name                = "${var.appinsights_name}"
  location            = var.location
  resource_group_name = var.rg_name
  application_type    = "web"
  depends_on = [azurerm_resource_group.rg]
}

resource "azurerm_key_vault" "akv" {
  name                = "${var.keyvault_name}"
  location            = var.location
  resource_group_name = var.rg_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"
  depends_on = [azurerm_resource_group.rg]
}

resource "azurerm_storage_account" "stacc" {
  name                     = "${var.storage_acct}"
  location                 = var.location
  resource_group_name      = var.rg_name
  account_tier             = "Standard"
  account_replication_type = "LRS"
  # Policy enforces this to be false. We must use Identity-based access for datastores.
  shared_access_key_enabled = false
  depends_on = [azurerm_resource_group.rg]
}

resource "azurerm_storage_container" "workspace_blob" {
  name                 = "workspaceblobstore"
  storage_account_id   = azurerm_storage_account.stacc.id
  container_access_type = "private"
}

resource "azurerm_storage_share" "workspace_file" {
  name               = "workspacefilestore"
  storage_account_id = azurerm_storage_account.stacc.id
  quota              = 5120
}

resource "azurerm_storage_container" "workspace_artifact" {
  name                  = "workspaceartifactstore"
  storage_account_id    = azurerm_storage_account.stacc.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "workspace_working_dir" {
  name                  = "workspaceworkingdirectory"
  storage_account_id    = azurerm_storage_account.stacc.id
  container_access_type = "private"
}

resource "azurerm_container_registry" "acr" {
  name                          = "${var.container_registry_name}"
  location                      = var.location
  resource_group_name           = var.rg_name
  sku                           = "Basic"
  admin_enabled                 = true
  depends_on = [azurerm_resource_group.rg]
  }

  resource "azurerm_machine_learning_workspace" "adl_mlw" {
  name                          = "${var.workspace_name}"
  location                      = var.location
  resource_group_name           = var.rg_name
  application_insights_id       = azurerm_application_insights.aml_appins.id
  key_vault_id                  = azurerm_key_vault.akv.id
  storage_account_id            = azurerm_storage_account.stacc.id
  container_registry_id         = azurerm_container_registry.acr.id
  public_network_access_enabled = true
  v1_legacy_mode_enabled        = false
  depends_on = [azurerm_resource_group.rg]
  
  identity {
    type = "SystemAssigned"
  }
  
  # Network isolation disabled - workspace accessible from public internet
  # For production, consider enabling managed_network with appropriate isolation_mode
}

resource "azurerm_machine_learning_datastore_blobstorage" "workspace_blob" {
  name                     = "workspaceblobstore"
  workspace_id             = azurerm_machine_learning_workspace.adl_mlw.id
  storage_container_id     = azurerm_storage_container.workspace_blob.id
  service_data_auth_identity = "WorkspaceSystemAssignedIdentity"
  is_default               = true
  depends_on               = [azurerm_machine_learning_workspace.adl_mlw]
}

resource "azurerm_machine_learning_datastore_fileshare" "workspace_file" {
  name                 = "workspacefilestore"
  workspace_id         = azurerm_machine_learning_workspace.adl_mlw.id
  storage_fileshare_id = azurerm_storage_share.workspace_file.id
  service_data_identity = "WorkspaceSystemAssignedIdentity"
  depends_on           = [azurerm_machine_learning_workspace.adl_mlw]
}

resource "azurerm_machine_learning_datastore_blobstorage" "workspace_artifact" {
  name                     = "workspaceartifactstore"
  workspace_id             = azurerm_machine_learning_workspace.adl_mlw.id
  storage_container_id     = azurerm_storage_container.workspace_artifact.id
  service_data_auth_identity = "WorkspaceSystemAssignedIdentity"
  depends_on               = [azurerm_machine_learning_workspace.adl_mlw]
}

resource "azurerm_machine_learning_datastore_blobstorage" "workspace_working_dir" {
  name                     = "workspaceworkingdirectory"
  workspace_id             = azurerm_machine_learning_workspace.adl_mlw.id
  storage_container_id     = azurerm_storage_container.workspace_working_dir.id
  service_data_auth_identity = "WorkspaceSystemAssignedIdentity"
  depends_on               = [azurerm_machine_learning_workspace.adl_mlw]
}

# Role assignments ensure workspace and compute identities have the data-plane access they need

resource "azurerm_machine_learning_compute_cluster" "training" {
  name                           = var.training_cluster_name
  location                       = var.location
  machine_learning_workspace_id  = azurerm_machine_learning_workspace.adl_mlw.id
  vm_size                        = var.training_cluster_vm_size
  vm_priority                    = "Dedicated"

  scale_settings {
    min_node_count                = tonumber(var.training_cluster_min_nodes)
    max_node_count                = tonumber(var.training_cluster_max_nodes)
    scale_down_nodes_after_idle_duration = "PT${tonumber(var.training_cluster_idle_seconds)}S"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_machine_learning_compute_cluster" "batch" {
  name                           = var.batch_cluster_name
  location                       = var.location
  machine_learning_workspace_id  = azurerm_machine_learning_workspace.adl_mlw.id
  vm_size                        = var.batch_cluster_vm_size
  vm_priority                    = "Dedicated"

  scale_settings {
    min_node_count                = tonumber(var.batch_cluster_min_nodes)
    max_node_count                = tonumber(var.batch_cluster_max_nodes)
    scale_down_nodes_after_idle_duration = "PT${tonumber(var.batch_cluster_idle_seconds)}S"
  }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_role_assignment" "workspace_data_scientist" {
  scope                = azurerm_machine_learning_workspace.adl_mlw.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azurerm_machine_learning_workspace.adl_mlw.identity[0].principal_id
}

resource "azurerm_role_assignment" "training_blob" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

resource "azurerm_role_assignment" "training_table" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Table Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

resource "azurerm_role_assignment" "training_queue" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Queue Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

resource "azurerm_role_assignment" "training_data_scientist" {
  scope                = azurerm_machine_learning_workspace.adl_mlw.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azurerm_machine_learning_compute_cluster.training.identity[0].principal_id
}

resource "azurerm_role_assignment" "batch_blob" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.batch.identity[0].principal_id
}

resource "azurerm_role_assignment" "batch_table" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Table Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.batch.identity[0].principal_id
}

resource "azurerm_role_assignment" "batch_queue" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Queue Data Contributor"
  principal_id         = azurerm_machine_learning_compute_cluster.batch.identity[0].principal_id
}

resource "azurerm_role_assignment" "batch_data_scientist" {
  scope                = azurerm_machine_learning_workspace.adl_mlw.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azurerm_machine_learning_compute_cluster.batch.identity[0].principal_id
}