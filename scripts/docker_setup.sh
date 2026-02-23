#!/bin/bash
# docker_setup.sh - Docker setup and container management
# Usage:
#   ./docker_setup.sh pull    # Pull Docker image
#   ./docker_setup.sh run     # Run container with port forwarding
#   ./docker_setup.sh start   # Start existing container
#   ./docker_setup.sh stop    # Stop container
#   ./docker_setup.sh enter   # Enter container shell
#   ./docker_setup.sh init    # Initialize/install dependencies
#   ./docker_setup.sh rebuild # Rebuild image from container
#   ./docker_setup.sh push    # Push to Docker Hub

set -e

IMAGE_NAME="zhouyida/nuplan-simulation"
CONTAINER_NAME="nuplan-simulation"
HOST_DATA_DIR="/home/zhouyida/.openclaw/workspace/diffusion-planner-project/data"
HOST_WORKSPACE="/home/zhouyida/.openclaw/workspace"

# Install dependencies inside container
install_deps() {
    echo "Installing dependencies..."
    docker exec $CONTAINER_NAME bash -c '
        if ! python3 -c "import rasterio" 2>/dev/null; then
            pip3 install --quiet \
                shapely \
                geopandas \
                botocore \
                s3transfer \
                boto3 \
                aioboto3 \
                aiofiles \
                s3fs \
                rasterio \
                retry \
                pyquaternion \
                pytest \
                psutil \
                pyyaml \
                hydra-core \
                omegaconf \
                msgpack \
                ujson \
                tensorboard \
                pyarrow \
                guppy3 \
                coloredlogs \
                pytorch-lightning==2.0.0 \
                bokeh==2.4.2 \
                nest-asyncio \
                selenium \
                aioboto3 \
                ray \
                2>/dev/null || true
        fi
    '
    echo "Dependencies installed."
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
            -v $HOST_WORKSPACE/carla:/workspace/carla \
            -w /workspace \
            $IMAGE_NAME \
            tail -f /dev/null
        
        echo "Container '$CONTAINER_NAME' started."
        
        # Auto-install deps on first run
        install_deps
        
        echo "Use './docker_setup.sh enter' to enter."
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
