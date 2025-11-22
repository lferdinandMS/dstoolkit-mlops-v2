# Configure the Microsoft Azure Provider
  terraform {
    backend "azurerm" {      
      use_oidc             = true  # Can also be set via `ARM_USE_OIDC` environment variable.}

    } 
}
  provider "azurerm" {
    
    use_oidc = true
    resource_provider_registrations = "none"
    features {
      machine_learning {
        purge_soft_deleted_workspace_on_destroy = true
      }
    }
    
  }