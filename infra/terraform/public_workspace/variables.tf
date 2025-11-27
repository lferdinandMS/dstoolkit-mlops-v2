# variable "subscription_id" {
#   type    = string
#   default = ""
# }
# variable "tenant_id" {
#   type    = string
#   default = ""
# }
# variable "client_id" {
#   type    = string
#   default = ""
# }


##############################
## Resource Group Variables
##############################
variable "rg_name" {
  type    = string
  default = "rg-terraform"
}
variable "tfstate_rg_name" {
  type    = string
  default = "rg-tfstate-terraform"
}

variable "storage_acct" {
  type    = string
  default = "stterraform"
}

variable "tfstate_storage_acct" {
  type    = string
  default = "sttfstateterraform"
}

variable "keyvault_name" {
  type    = string
  default = "kvterraform"
}

variable "appinsights_name" {
  type    = string
  default = "appiterraform"
}

variable "container_registry_name" {
  type    = string
  default = "crterraform"
}

variable "workspace_name" {
  type    = string
  default = "amlterraform"
}


variable "location" {
  type    = string
  default = "eastus"
}

variable "service_principal_object_id" {
  type        = string
  description = "The Object ID of the service principal used for deployments"
}

variable "training_cluster_name" {
  type    = string
  default = "cpucluster"
}

variable "training_cluster_vm_size" {
  type    = string
  default = "STANDARD_DS3_V2"
}

variable "training_cluster_min_nodes" {
  type    = number
  default = 0
}

variable "training_cluster_max_nodes" {
  type    = number
  default = 4
}

variable "training_cluster_idle_seconds" {
  type    = number
  default = 600
}

variable "batch_cluster_name" {
  type    = string
  default = "batchcluster"
}

variable "batch_cluster_vm_size" {
  type    = string
  default = "STANDARD_DS3_V2"
}

variable "batch_cluster_min_nodes" {
  type    = number
  default = 0
}

variable "batch_cluster_max_nodes" {
  type    = number
  default = 4
}

variable "batch_cluster_idle_seconds" {
  type    = number
  default = 600
}
