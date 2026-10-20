@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  echo Python environment missing. See the setup guide in docs.
  pause
  exit /b 1
)
".venv-web\Scripts\python.exe" -B -X utf8 "python\start_ct_studio_demo.py" %*
if errorlevel 1 (
  echo.
  echo Startup check did not pass. Read the message above; existing jobs were not stopped.
  pause
  exit /b 1
)
