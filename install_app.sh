#!/bin/bash

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null
    then
        echo "Docker could not be found. Please install Docker from https://docs.docker.com/get-docker/"
        open https://docs.docker.com/get-docker/
        exit 1
    fi
}

# Function to build the Docker image
build_docker_image() {
    echo "Building the Docker image from the Dockerfile..."
    # Replace 'computer_vision_streamlit_app_image' with your preferred Docker image name
    docker build -t computer_vision_streamlit_app_image .
}

# Function to run the Docker container
run_docker_container() {
    echo "Running the Docker container..."
    # Replace 'computer_vision_streamlit_app_image' with your Docker image name used in the build step
    docker run -p 8501:8501 computer_vision_streamlit_app_image
}

# Function to open the Streamlit app in the browser
open_streamlit_app() {
    echo "Opening the Streamlit app in the browser..."
    open http://localhost:8501
}

# Main script execution
check_docker
build_docker_image
run_docker_container &
sleep 2 # Wait for the server to start
open_streamlit_app
