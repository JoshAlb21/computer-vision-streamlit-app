@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Check for Docker
docker -v >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Docker is not installed. Please install Docker from https://docs.docker.com/docker-for-windows/install/
    start https://docs.docker.com/docker-for-windows/install/
    exit /b
)

:: Build the Docker image
echo Building the Docker image from the Dockerfile...
docker build -t tachinidae_analyzer_image .

:: Run the Docker container
echo Running the Docker container...
start "" http://localhost:8501
docker run -p 8501:8501 tachinidae_analyzer_image
