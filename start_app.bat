@echo off

:: Run the Docker container
docker run -p 8501:8501 tachinidae_analyzer_image
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to run the Docker container.
    echo Please ensure the Docker image is built. If not, run 'install_app.bat' first.
    exit /b
)

:: Open the browser if the container starts successfully
start "" http://localhost:8501
