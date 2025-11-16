@echo off
REM Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
REM
REM NVIDIA CORPORATION and its licensors retain all intellectual property
REM and proprietary rights in and to this software, related documentation
REM and any modifications thereto.  Any use, reproduction, disclosure or
REM distribution of this software and related documentation without an express
REM license agreement from NVIDIA CORPORATION is strictly prohibited.

setlocal

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Use the Python from the environment, or default to python
if "%PYTHON%"=="" set PYTHON=python

REM Check if lc_agent is installed
%PYTHON% -c "import lc_agent" 2>nul
if errorlevel 1 (
    echo [ERROR] lc_agent is not installed in the current Python environment
    echo.
    echo Please run the installation script first:
    echo   dev-install.bat
    echo.
    echo Or install it manually:
    echo   cd source\modules\lc_agent ^&^& pip install -e .
    exit /b 1
)

REM Run the CLI with all arguments passed through
%PYTHON% "%SCRIPT_DIR%lc_agent_cli.py" %*

endlocal
