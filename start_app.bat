@echo off
REM Check if Docker is running
docker info > NUL 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Please start Docker and then rerun this script.
    exit /b 1
)

REM Docker container name
set CONTAINER_NAME=tachinidae_analyzer

REM Check if a Docker container with the specified name is running or exists
docker ps -a | findstr /C:%CONTAINER_NAME% > NUL
if %errorlevel% neq 0 (
    call :run_docker_container
    call :is_container_running
    if %errorlevel% eq 0 (
        echo Container is running.
        call :open_streamlit_app
    ) else (
        echo Failed to start the container. Please check Docker logs.
    )
    goto :EOF
)

REM Handle the existing container
echo A container with the name '%CONTAINER_NAME%' already exists.
set /p user_choice=Do you want to remove the existing container and start a new one? (y/n): 

if /I "%user_choice%"=="y" (
    docker rm -f %CONTAINER_NAME%
    call :run_docker_container
    echo Opening the Streamlit app in the new container...
    call :open_streamlit_app
) else (
    echo Opening the Streamlit app in the existing container...
    call :open_streamlit_app
)
goto :EOF

:run_docker_container
echo Running the Docker container...
docker run -d --name %CONTAINER_NAME% -p 8501:8501 tachinidae_analyzer_image
REM Wait briefly for the container to start
timeout /t 5 /nobreak
echo Waited 5 seconds.
goto :EOF

:is_container_running
docker ps | findstr /C:%CONTAINER_NAME% > NUL
exit /b %errorlevel%

:open_streamlit_app
echo Opening the Streamlit app in the browser...
start http://localhost:8501
goto :EOF
