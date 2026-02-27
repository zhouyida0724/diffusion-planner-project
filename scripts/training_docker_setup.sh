#!/bin/bash
# training_docker_setup.sh - Training Docker setup and container management
# Usage:
#   ./training_docker_setup.sh pull    # Pull Docker image
#   ./training_docker_setup.sh run     # Run container with port forwarding
#   ./training_docker_setup.sh start   # Start existing container
#   ./training_docker_setup.sh stop    # Stop container
#   ./training_docker_setup.sh enter   # Enter container shell
#   ./training_docker_setup.sh init    # Initialize/install dependencies
#   ./training_docker_setup.sh rebuild # Rebuild image from container
#   ./training_docker_setup.sh push    # Push to Docker Hub

set -e

IMAGE_NAME="zhouyida/diffusion-planner-training"
CONTAINER_NAME="diffusion-planner-training"
HOST_DATA_DIR="/home/zhouyida/.openclaw/workspace/diffusion-planner-project/data"
HOST_WORKSPACE="/home/zhouyida/.openclaw/workspace"

# Install training dependencies inside container
install_deps() {
    echo "Installing training dependencies..."
    docker exec $CONTAINER_NAME bash -c '
        pip3 install --upgrade pip --quiet
        
        pip3 install --quiet torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
        
        pip3 install --quiet \
                pytorch_lightning==2.0.1 \
                timm==1.0.10 \
                numpy \
                tqdm \
                pyyaml \
                matplotlib \
                scipy \
                tensorboard
    '
    echo "Training dependencies installed."
}

case "${1:-run}" in
    pull)
        echo "Pulling Docker image..."
        docker pull $IMAGE_NAME:latest
        ;;
        
    run)
        echo "Running new container..."
        # Remove existing container if any
        docker rm -f $CONTAINER_NAME 2>/dev/null || true
        
        docker run -d \
            --name $CONTAINER_NAME \
            -p 5006:5006 \
            -p 2000:2000 \
            -v $HOST_DATA_DIR:/workspace/data \
            -v $HOST_WORKSPACE/diffusion-planner-project:/workspace \
            -w /workspace \
            $IMAGE_NAME \
            tail -f /dev/null
        
        echo "Container '$CONTAINER_NAME' started."
        
        # Auto-install deps on first run
        install_deps
        
        echo "Use './training_docker_setup.sh enter' to enter."
        ;;
        
        
    start)
        echo "Starting container..."
        docker start $CONTAINER_NAME
        ;;
        
    stop)
        echo "Stopping container..."
        docker stop $CONTAINER_NAME
        ;;
        
    enter)
        echo "Entering container..."
        docker exec -it $CONTAINER_NAME bash
        ;;
        
    exec)
        echo "Entering container..."
        docker exec -it $CONTAINER_NAME bash
        ;;
        
    init)
        echo "Installing training dependencies..."
        install_deps
        ;;
        
    rebuild)
        echo "Rebuilding image from current container..."
        docker commit $CONTAINER_NAME $IMAGE_NAME:latest
        echo "Image updated: $IMAGE_NAME:latest"
        ;;
        
    push)
        echo "Pushing to Docker Hub..."
        docker push $IMAGE_NAME:latest
        ;;
        
    *)
        echo "Usage: $0 {pull|run|start|stop|enter|exec|init|rebuild|push}"
        exit 1
        ;;
esac
