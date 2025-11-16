@echo off
REM Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
REM
REM NVIDIA CORPORATION and its licensors retain all intellectual property
REM and proprietary rights in and to this software, related documentation
REM and any modifications thereto.  Any use, reproduction, disclosure or
REM distribution of this software and related documentation without an express
REM license agreement from NVIDIA CORPORATION is strictly prohibited.

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Default Python command
if "%PYTHON%"=="" set PYTHON=python

REM Install mode (editable or regular)
set INSTALL_MODE=%1
if "%INSTALL_MODE%"=="" set INSTALL_MODE=editable

echo ================================================
echo LC Agent Development Installation Script
echo ================================================
echo.
echo Python executable: %PYTHON%
echo Install mode: %INSTALL_MODE%
echo.

REM Verify Python is available
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python executable not found: %PYTHON%
    echo Please specify Python path: set PYTHON=C:\path\to\python.exe
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('%PYTHON% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: !PYTHON_VERSION!

REM Warn if not in virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo [WARN] Not running in a virtual environment!
    echo It's recommended to create and activate a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    set /p CONTINUE="Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" exit /b 1
)

echo.
echo Installing LC Agent modules...
echo.

REM Upgrade pip first
echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] Failed to upgrade pip, continuing anyway...
)

REM Install lc_agent first
echo.
echo Step 1/2: Installing lc_agent ^(core framework^)...
cd "%SCRIPT_DIR%\source\modules\lc_agent"

if "%INSTALL_MODE%"=="editable" (
    echo Running: pip install -e .
    %PYTHON% -m pip install -e . --verbose
) else if "%INSTALL_MODE%"=="-e" (
    echo Running: pip install -e .
    %PYTHON% -m pip install -e . --verbose
) else (
    echo Running: pip install .
    %PYTHON% -m pip install . --verbose
)

if errorlevel 1 (
    echo.
    echo [ERROR] Installation of lc_agent failed!
    echo Please check the error messages above.
    exit /b 1
)

REM Install lc_agent_cli
echo.
echo Step 2/2: Installing lc_agent_cli ^(CLI with NVIDIA model support^)...
cd "%SCRIPT_DIR%\source\modules\lc_agent_cli"

if "%INSTALL_MODE%"=="editable" (
    echo Running: pip install -e .
    %PYTHON% -m pip install -e . --verbose
) else if "%INSTALL_MODE%"=="-e" (
    echo Running: pip install -e .
    %PYTHON% -m pip install -e . --verbose
) else (
    echo Running: pip install .
    %PYTHON% -m pip install . --verbose
)

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed!
    echo Please check the error messages above.
    exit /b 1
)

echo.
echo ================================================
echo Installation complete!
echo ================================================
echo.
echo You can now use the LC Agent CLI:
echo   lc-agent                      # Interactive mode
echo   lc-agent --query "your text"  # Single query
echo   run-lc-agent.bat              # Using wrapper script
echo.
echo Or import lc_agent in your Python scripts:
echo   from lc_agent import RunnableNetwork, RunnableNode
echo.
echo Set your NVIDIA API key:
echo   set NVIDIA_API_KEY=your_key_here
echo.
echo For more information, see claude.md in the repository root.

endlocal
