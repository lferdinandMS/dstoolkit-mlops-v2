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
  # Re-enabled shared keys to allow Azure ML pipeline data streaming (SAS generation required for mount).
  shared_access_key_enabled = true
  depends_on = [azurerm_resource_group.rg]
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

# Ensure the workspace managed identity can access the storage account used by AML datastores.
resource "azurerm_role_assignment" "workspace_storage_blob_data_contributor" {
  scope                = azurerm_storage_account.stacc.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_workspace.adl_mlw.identity[0].principal_id
  depends_on           = [azurerm_machine_learning_workspace.adl_mlw, azurerm_storage_account.stacc]
}

# Role assignments are handled idempotently in the CI workflow to avoid conflicts