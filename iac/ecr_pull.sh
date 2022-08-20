# NB Requires aws-cli > 2.x.x
# Retrieve an authentication token and authenticate Docker client
ACCOUNT_ID=$(aws sts get-caller-identity | jq -r ".Account")

# push image to AWS repository:
docker pull "${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com/ga-repo:latest"