resource "aws_ecr_repository" "ga_repo" {
  name = "ga-repo"
}

resource "aws_s3_bucket" "ga_bucket" {
  bucket = "ga-44e2849d-f075"
  acl = "private"

  tags = {
    Name = "Genetic Algorithm bucket"
  }

  versioning { enabled = true}
}

# Allow ssh access to ec2 instances
# generate key with: ssh-keygen -t rsa -b 2048
resource "aws_key_pair" "ga-ec2_key" {
  key_name   = "ga-ec2-key"
  public_key = var.public_key
}

resource "aws_instance" "ga_instance" {
    instance_type = var.instance_type
    # Amazon-linux ami
    ami = "ami-0be9259c3f48b4026"
    security_groups = [aws_security_group.instances.name]
    count         = var.instance_count
    key_name= "ga-ec2-key"
    iam_instance_profile = aws_iam_instance_profile.ec2_profile.id



    connection {
      type        = "ssh"
      host        = self.public_ip
      user        = "ubuntu"
      private_key = file("~/.ssh/ga_ec2_rsa")
      timeout     = "4m"
   }

    # hello world web app
    user_data       = <<-EOF
                #! /bin/bash
                sudo yum update -y
                sudo yum install -y jq
                sudo amazon-linux-extras install -y docker
                sudo service docker start
                sudo usermod -a -G docker ec2-user
                sudo chkconfig docker on
                aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin "${var.account}.dkr.ecr.${var.region}.amazonaws.com"
                docker pull "${var.account}.dkr.ecr.${var.region}.amazonaws.com/ga-repo:latest"
                docker run -e PARALLEL=1 --rm "${var.account}.dkr.ecr.${var.region}.amazonaws.com/ga-repo:latest"
              EOF
}

# `data` references data item which already exist in aws
data "aws_vpc" "default_vpc" {
    default =  true
}

data "aws_subnet_ids" "default_subnet" {
    vpc_id = data.aws_vpc.default_vpc.id
}

resource "aws_security_group" "instances" {
  name = "instance-security-group"
}

resource "aws_security_group_rule" "allow_http_inbound" {
    type = "ingress"
    security_group_id = aws_security_group.instances.id

    from_port = 8080
    to_port = 8080
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "allow_ssh_inbound" {
    type = "ingress"
    security_group_id = aws_security_group.instances.id
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "allow_https_inbound" {
    type = "ingress"
    security_group_id = aws_security_group.instances.id
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "allow_https_outbound" {
    type = "egress"
    security_group_id = aws_security_group.instances.id
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
}

resource "aws_iam_role" "ec2_iam_role" {
    name = "ec2-iam-role"
    assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}


# S3 access on EC2
resource "aws_iam_instance_profile" "ec2_profile" {
    name = "ec2-profile"
    role = "ec2-iam-role"
}

resource "aws_iam_role_policy" "ec2_iam_role_policy" {
  name = "ec2-iam-role-policy"
  role = aws_iam_role.ec2_iam_role.id
  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::*"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": ["arn:aws:s3:::*"]
    },
        {
        "Effect": "Allow",
        "Action": [
            "ecr:GetAuthorizationToken",
            "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchCheckLayerAvailability",
            "ecr:BatchImportUpstreamImage",
            "ecr:DescribeImageScanFindings",
            "ecr:DescribeImages",
            "ecr:ListImages"
            ],
        "Resource": "*"
    }
  ]
}
EOF
}

data "aws_iam_policy_document" "ecs" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      identifiers = ["ec2.amazonaws.com"]
      type        = "Service"
    }
  }
}

resource "aws_iam_role" "ecs" {
  assume_role_policy = data.aws_iam_policy_document.ecs.json
  name               = "ecs-instance-role"

}

resource "aws_iam_role_policy_attachment" "ecs" {
  role       = aws_iam_role.ecs.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs" {
  name = "ecs-instance-profile"
  role = aws_iam_role.ecs.name
}