sudo yum update -y
sudo yum install -y jq
sudo amazon-linux-extras install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo chkconfig docker on
aws ecr get-login-password --region eu-west-1 | sudo docker login --username AWS --password-stdin "037548347524.dkr.ecr.eu-west-1.amazonaws.com"
sudo docker pull "037548347524.dkr.ecr.eu-west-1.amazonaws.com/ga-repo:latest"

docker run -e PARALLEL=1 \
-e BUCKET=ga-44e2849d-f075 \
-e POPULATION_SIZE=6 \
-e GENERATIONS=3 \
-e BUCKET=ga-44e2849d-f075 \
-e PYTHONUNBUFFERED=1 \
--rm "037548347524.dkr.ecr.eu-west-1.amazonaws.com/ga-repo:latest"

# remove docker images
# docker rmi -f $(docker image ls -a -q)