# Resource Manager は tenancy_ocid / region / compartment_ocid を自動で埋める

variable "tenancy_ocid" { type = string }
variable "region" { type = string }
variable "compartment_ocid" { type = string }

variable "name" {
  type    = string
  default = "altpnt"
}

variable "ssh_public_key" {
  type        = string
  description = "インスタンスに登録する SSH 公開鍵（ssh-ed25519 ... / ssh-rsa ...）"
}

variable "allowed_cidr" {
  type        = string
  description = "SSH（と公開時の API）を許可する接続元 CIDR。自宅/職場の IP/32 を推奨"
  default     = "0.0.0.0/0"
}

variable "shape" {
  type        = string
  description = "VM.Standard.A1.Flex（Arm, Always Free 枠あり）/ VM.Standard.E4.Flex（x86）など"
  default     = "VM.Standard.A1.Flex"
}

variable "ocpus" {
  type    = number
  default = 2
}

variable "memory_gb" {
  type    = number
  default = 12
}

variable "boot_volume_gb" {
  type    = number
  default = 50
}

variable "ad_index" {
  type        = number
  description = "可用性ドメインの番号（A1 の容量不足エラー時は 1, 2 を試す）"
  default     = 0
}

variable "repo_url" {
  type    = string
  default = "https://github.com/UeEmon/altpnt.git"
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "expose_api" {
  type        = bool
  description = "true で cloud API(8080) を allowed_cidr に公開。false なら SSH トンネル経由のみ"
  default     = false
}

variable "anthropic_api_key" {
  type        = string
  description = "任意。LLM 状況解析用。インスタンスメタデータに残るため、後から SSH で .env に設定する方が安全"
  default     = ""
  sensitive   = true
}

variable "openai_api_key" {
  type      = string
  default   = ""
  sensitive = true
}
