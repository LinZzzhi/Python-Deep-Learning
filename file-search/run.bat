@echo off
chcp 936 >nul 2>&1
cd /d "%~dp0"
title ffind 快文件查找器 - 一键启动
setlocal
set "PY="
set "PYVER="
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

echo.
echo  ==================================================
echo    ffind 快文件查找器  (多进程索引 + SQLite 查询)
echo  ==================================================
echo.

rem ============ 1/3  查找可用的 Python 解释器 ============
echo  [1/3] 检查 Python 解释器 ...

py -3 -V >nul 2>&1
if not errorlevel 1 goto :use_py
python -V >nul 2>&1
if not errorlevel 1 goto :use_python
python3 -V >nul 2>&1
if not errorlevel 1 goto :use_python3
goto :err_no_python

:use_py
set "PY=py -3"
goto :got_python

:use_python
set "PY=python"
goto :got_python

:use_python3
set "PY=python3"
goto :got_python

:got_python
for /f "delims=" %%v in ('%PY% -V 2^>^&1') do set "PYVER=%%v"
echo        找到 %PYVER%  (%PY%)

rem ============ 2/3  检查 Python 版本 >= 3.10 ============
%PY% -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if errorlevel 1 goto :err_old_python
echo        版本检查通过 (需要 3.10 或更高)

rem ============ 3/3  检查项目文件与依赖 ============
if not exist "%~dp0ffinder\__init__.py" goto :err_files
if not exist "%~dp0ffinder\cli.py" goto :err_files
if not exist "%~dp0ffinder\index.py" goto :err_files
if not exist "%~dp0ffinder\scan.py" goto :err_files
if not exist "%~dp0ffinder\server.py" goto :err_files
if not exist "%~dp0ffinder\web\index.html" goto :err_files

%PY% -c "import ffinder.cli,ffinder.index,ffinder.scan,ffinder.server" >nul 2>&1
if errorlevel 1 goto :err_import
echo        项目文件完整, 依赖检查通过 (纯标准库, 无第三方依赖)

rem ============ 启动 ============
echo.
echo  正在启动 Web 界面: http://127.0.0.1:8765
echo  关闭本窗口或按 Ctrl+C 即可停止服务。
echo.
%PY% -m ffinder serve --open
set "RC=%ERRORLEVEL%"
echo.
echo  服务已退出 (退出码 %RC%)
endlocal
pause
exit /b %RC%

rem ============ 缺少依赖时的终止分支 ============
:err_no_python
echo.
echo  [错误] 未找到 Python 解释器。
echo         本项目需要 Python 3.10 或更高版本。
echo         请安装 Python 并勾选 Add to PATH, 然后重新双击本脚本。
echo.
echo  任务已终止, 未启动任何服务。
endlocal
pause
exit /b 1

:err_old_python
echo.
echo  [错误] Python 版本过低: %PYVER%
echo         本项目需要 Python 3.10 或更高版本。
echo         请升级 Python 后重新双击本脚本。
echo.
echo  任务已终止, 未启动任何服务。
endlocal
pause
exit /b 2

:err_files
echo.
echo  [错误] 项目文件不完整, 缺少 ffinder 包的源码或 web 资源。
echo         请确认 run.bat 与 ffinder 目录位于同一层:
echo         %~dp0ffinder\  (需含 __init__.py cli.py index.py scan.py server.py web\index.html)
echo.
echo  任务已终止, 未启动任何服务。
endlocal
pause
exit /b 3

:err_import
echo.
echo  [错误] ffinder 包无法正常导入, 依赖检查未通过。
echo         以下为详细错误:
echo.
%PY% -c "import ffinder.cli,ffinder.index,ffinder.scan,ffinder.server"
echo.
echo  任务已终止, 未启动任何服务。
endlocal
pause
exit /b 4
