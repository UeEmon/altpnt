output "public_ip" {
  value = oci_core_instance.vm.public_ip
}

output "ssh" {
  value = "ssh ubuntu@${oci_core_instance.vm.public_ip}"
}

output "api_tunnel" {
  description = "手元で実行後、http://localhost:8080/docs を開く"
  value       = "ssh -N -L 8080:127.0.0.1:8080 -L 4000:127.0.0.1:4000 ubuntu@${oci_core_instance.vm.public_ip}"
}

output "setup_log" {
  value = "ssh ubuntu@${oci_core_instance.vm.public_ip} 'tail -f /var/log/altpnt-setup.log'"
}
