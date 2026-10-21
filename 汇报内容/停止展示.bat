@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\media-player\.venv\Scripts\python.exe"
if exist "%PY%" goto stop
set "PY=python"
python -V >nul 2>&1
if not errorlevel 1 goto stop
py -3 "%~dp0launch.py" --stop
goto result
:stop
"%PY%" "%~dp0launch.py" --stop
:result
if errorlevel 1 pause
