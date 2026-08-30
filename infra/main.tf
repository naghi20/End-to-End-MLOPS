terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "mlops" {
  bucket        = "${var.project_name}-${var.environment}-artifacts-naghi"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "mlops" {
  bucket = aws_s3_bucket.mlops.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_ecr_repository" "churn_api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ec2_policy" {
  name = "${var.project_name}-ec2-policy"
  role = aws_iam_role.ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
        Resource = [aws_s3_bucket.mlops.arn, "${aws_s3_bucket.mlops.arn}/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData", "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

resource "aws_security_group" "mlops_sg" {
  name        = "${var.project_name}-sg"
  description = "MLOps lab host"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }
  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "Serving API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "mlops_host" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]

  user_data = <<-EOT
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io docker-compose-plugin awscli
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu

    mkdir -p /opt/mlflow
    docker run -d --restart always --name mlflow \
      -p 5000:5000 \
      -v /opt/mlflow:/mlflow \
      -e AWS_DEFAULT_REGION=${var.aws_region} \
      python:3.11-slim bash -c "pip install mlflow boto3 && \
        mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db \
        --default-artifact-root s3://${aws_s3_bucket.mlops.bucket}/mlflow-artifacts \
        --host 0.0.0.0 --port 5000"
  EOT

  tags = {
    Name = "${var.project_name}-mlops-host"
  }
}
