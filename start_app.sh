#!/bin/bash

# Function to check if Docker is running
is_docker_running() {
    docker info > /dev/null 2>&1
    return $?
}

# Docker container name
CONTAINER_NAME="computer_vision_streamlit_app"

# Function to check if a Docker container with the specified name is running or exists
check_container_exists() {
    docker ps -a | grep -q $CONTAINER_NAME
    return $?
}

# Function to prompt the user and handle the existing container
handle_existing_container() {
    echo "A container with the name '$CONTAINER_NAME' already exists."
    read -p "Do you want to remove the existing container and start a new one? (y/n): " user_choice

    if [[ $user_choice == "y" ]]; then
        docker rm -f $CONTAINER_NAME
        run_docker_container
        echo "Opening the Streamlit app in the new container..."
        open_streamlit_app
    else
        echo "Opening the Streamlit app in the existing container..."
        open_streamlit_app
    fi
}

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
    docker run -d --name $CONTAINER_NAME -p 8501:8501 computer_vision_streamlit_app_image
    # Wait briefly for the container to start
    sleep 5
    echo "Waited 5 seconds."
}

# Function to check if the container is running
is_container_running() {
    docker ps | grep -q $CONTAINER_NAME
    return $?
}

# Function to open the Streamlit app in the browser
open_streamlit_app() {
    echo "Opening the Streamlit app in the browser..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:8501
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8501
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        start http://localhost:8501
    else
        echo "Unable to detect OS type. Please manually open http://localhost:8501 in your browser."
    fi
}

# Main script execution
if ! is_docker_running; then
    echo "Docker is not running. Please start Docker and then rerun this script."
    exit 1
fi

if check_and_stop_container_on_port 8501; then
    sleep 2 # Wait for the container to fully stop
fi

if check_container_exists; then
    handle_existing_container
else
    run_docker_container
    if is_container_running; then
        echo "Container is running."
        open_streamlit_app
    else
        echo "Failed to start the container. Please check Docker logs."
    fi
fi
