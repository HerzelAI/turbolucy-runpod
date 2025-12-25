#!/bin/bash
# TurboLucy RunPod Deployment Script

# CONFIGURATION
DOCKER_USER="arieliking" # Update if different
IMAGE_NAME="turbolucy-runpod"
TAG="v1.8"

echo "🚀 Building TurboLucy for RunPod Serverless..."

# Build the image
docker build -t $DOCKER_USER/$IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    echo "✅ Build successful. Pushing to Docker Hub..."
    docker push $DOCKER_USER/$IMAGE_NAME:$TAG
    echo "🎉 Successfully pushed to $DOCKER_USER/$IMAGE_NAME:$TAG"
else
    echo "❌ Build failed. Check the logs above."
    exit 1
fi
