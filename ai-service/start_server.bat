@echo off
REM ============================================================================
REM Aurora Melody AI Server Startup Script (Windows)
REM ============================================================================
REM
REM The server loads configuration from (priority order):
REM   1. Command-line arguments (if provided)
REM   2. Environment variables / .env file
REM   3. config.yaml
REM
REM Quick Start:
REM   1. Copy config.yaml.example to config.yaml (or edit config.yaml)
REM   2. Update the model paths in config.yaml
REM   3. Run: start_server.bat
REM
REM Or with arguments:
REM   start_server.bat --model_path model.safetensors --config_path config.json --vocab_path vocab.pkl
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo =============================================
echo   Aurora Melody AI Server
echo =============================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Change to script directory
cd /d "%~dp0"

REM Check if config.yaml exists
if exist "config.yaml" (
    echo Config: config.yaml found
) else (
    echo Note: config.yaml not found - using defaults or CLI args
)

REM Check if .env exists
if exist ".env" (
    echo Env:    .env found
)

echo.
echo Starting server...
echo.

REM Pass all arguments to the Python script
python melody_server.py %*

pause
