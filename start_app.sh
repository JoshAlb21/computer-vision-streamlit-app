#!/bin/bash

# Docker container name
CONTAINER_NAME="tachinidae_analyzer"

# Function to check if the specified port is being used by a Docker container
check_and_stop_container_on_port() {
    local port=$1
    local container_id=$(docker ps --filter "publish=$port" --format "{{.ID}}")

    if [ ! -z "$container_id" ]; then
        echo "Port $port is used by container $container_id. Stopping this container..."
        docker stop $container_id
        return 0 # Indicate that a container was stopped
    else
        return 1 # No container to stop
    fi
}

# Function to run the Docker container
run_docker_container() {
    echo "Running the Docker container..."
    docker run --name $CONTAINER_NAME -p 8501:8501 tachinidae_analyzer_image &
}

# Function to open the Streamlit app in the browser
open_streamlit_app() {
    echo "Opening the Streamlit app in the browser..."
    open http://localhost:8501
}

# Main script execution
if check_and_stop_container_on_port 8501; then
    sleep 2 # Wait for the container to fully stop
fi

run_docker_container
sleep 10 # Wait for the server to start
open_streamlit_app
