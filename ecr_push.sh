# NB Requires aws-cli > 2.x.x
# Retrieve an authentication token and authenticate Docker client
ACCOUNT_ID=$(aws sts get-caller-identity | jq -r ".Account")

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com"

# Build Docker image
docker build -t ga-repo .

# tag your image so you can push the image to this repository:
docker tag ga-repo:latest "${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com/ga-repo:latest"

# push image to AWS repository:
docker push "${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com/ga-repo:latest"
