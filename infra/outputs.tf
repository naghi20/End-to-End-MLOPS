output "ec2_public_ip" {
  value = aws_instance.mlops_host.public_ip
}

output "mlflow_url" {
  value = "http://${aws_instance.mlops_host.public_ip}:5000"
}

output "api_url" {
  value = "http://${aws_instance.mlops_host.public_ip}:8000"
}

output "s3_bucket" {
  value = aws_s3_bucket.mlops.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.churn_api.repository_url
}
