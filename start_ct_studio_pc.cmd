@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  echo Python environment missing. See the setup guide in docs.
  pause
  exit /b 1
)
".venv-web\Scripts\python.exe" -B -X utf8 "python\start_ct_studio_demo.py" --desktop %*
if errorlevel 1 (
  echo.
  echo PC startup check did not pass. Existing services and results were preserved.
  pause
  exit /b 1
)
