# altpnt — Oracle Cloud (OCI) デプロイ用 Terraform スタック
# Resource Manager（OCI コンソール）にこのディレクトリの zip をアップロードして Apply する想定。
# ローカル実行する場合は ~/.oci/config を用意し、terraform.tfvars で tenancy_ocid 等を渡す。

terraform {
  required_version = ">= 1.2"
  required_providers {
    oci = { source = "oracle/oci", version = ">= 5.0" }
  }
}

provider "oci" {
  region = var.region
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

locals {
  is_flex = length(regexall("Flex$", var.shape)) > 0
  ad_name = data.oci_identity_availability_domains.ads.availability_domains[var.ad_index].name
  user_data = templatefile("${path.module}/cloud-init.yaml", {
    repo_url          = var.repo_url
    repo_ref          = var.repo_ref
    anthropic_api_key = var.anthropic_api_key
    openai_api_key    = var.openai_api_key
    cloud_bind        = var.expose_api ? "0.0.0.0" : "127.0.0.1"
  })
}

# ------------------------------------------------------------------ ネットワーク
resource "oci_core_vcn" "vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.42.0.0/16"]
  display_name   = "${var.name}-vcn"
  dns_label      = "altpnt"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "${var.name}-igw"
}

resource "oci_core_route_table" "rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "${var.name}-rt"
  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_security_list" "sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.vcn.id
  display_name   = "${var.name}-sl"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  ingress_security_rules {
    description = "SSH"
    protocol    = "6"
    source      = var.allowed_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }

  dynamic "ingress_security_rules" {
    for_each = var.expose_api ? [1] : []
    content {
      description = "altpnt cloud API"
      protocol    = "6"
      source      = var.allowed_cidr
      tcp_options {
        min = 8080
        max = 8080
      }
    }
  }
}

resource "oci_core_subnet" "public" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  cidr_block        = "10.42.1.0/24"
  display_name      = "${var.name}-public"
  dns_label         = "pub"
  route_table_id    = oci_core_route_table.rt.id
  security_list_ids = [oci_core_security_list.sl.id]
}

# ------------------------------------------------------------------ インスタンス
resource "oci_core_instance" "vm" {
  compartment_id      = var.compartment_ocid
  availability_domain = local.ad_name
  display_name        = var.name
  shape               = var.shape

  dynamic "shape_config" {
    for_each = local.is_flex ? [1] : []
    content {
      ocpus         = var.ocpus
      memory_in_gbs = var.memory_gb
    }
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
    hostname_label   = var.name
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data           = base64encode(local.user_data)
  }
}
