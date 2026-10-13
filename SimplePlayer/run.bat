@echo off
rem ============================================================
rem  SimplePlayer launcher
rem  Runs player.py with the virtual environment of this folder.
rem  You can also drag a media file onto this .bat to open it.
rem ============================================================
setlocal
set "PY=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%PY%" (
    echo [ERROR] Virtual environment not found:
    echo         %PY%
    echo.
    echo Create it first:
    echo         python -m venv .venv
    echo         .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

start "" "%PY%" "%~dp0player.py" %*
endlocal
