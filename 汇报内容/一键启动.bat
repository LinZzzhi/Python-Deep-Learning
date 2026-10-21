@echo off
setlocal
cd /d "%~dp0"
title Python Project Presentation
set "PY=%~dp0..\media-player\.venv\Scripts\python.exe"
if exist "%PY%" goto launch
set "PY=python"
%PY% -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if not errorlevel 1 goto launch
set "PY=py"
%PY% -3 -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if errorlevel 1 goto missing
%PY% -3 "%~dp0launch.py" --background
goto result
:launch
"%PY%" "%~dp0launch.py" --background
:result
if errorlevel 1 (
 echo Startup failed. Read runtime\launcher.log for details.
 pause
 exit /b 1
)
exit /b 0
:missing
echo Python 3.10 or later is required. Please install Python and add it to PATH.
pause
exit /b 1
