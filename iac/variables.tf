variable "account" {
  default = "037548347524"
}

variable "instance_count" {
  default = "3"
}
variable "instance_type" {
  default = "t2.nano"
}

variable "region" {
  default = "eu-west-1"
}

variable "vpc_cidr" {
  description = "CIDR range of VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_key" {
  default = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDCaqXPSWV5xSt8BM/e1oVEiUA4cF/NBQA7ZQvndbbx5z4mkIfKlDI3HIuxKNEYAWS0BjpxYLzOKEHQFlIkWqopc9h/LBI33VLLRM5ESnekp2PxHile3RiEJaXeu4mCswHffMpjuETJapTxrBMPZRL6rnDf/uEenoxmzMSIYYpa/ZPHEBKMg6MYDgaQ0EO6Ce007kMEJzgApjdczTys6N/HKDBxOmERNGbKf5POExYiWyiHr3bMUbOQ/DsbGCnXTxJjyZCVDn1DIdfJ+J82fCc8hTv3NqFUVNmMz12Pd5uM+VbcxBA5z32gq8CfZ5OHhD2w6kO1hKwgrA9BcpJbwCov joshharrison@joshs-mbp.lan"
}