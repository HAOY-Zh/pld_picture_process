@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  py -3.11 -m venv .venv-web
  if errorlevel 1 goto failed
)
rem CPU Torch is sufficient: FPGA executes the neural networks.
".venv-web\Scripts\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv-web\Scripts\python.exe" -m pip install -r python\requirements-current.txt
if errorlevel 1 goto failed
if not exist "data\ct_studio_current_20261027\host_config.json" (
  ".venv-web\Scripts\python.exe" -B -X utf8 python\ct_studio_current_release.py install
  if errorlevel 1 goto failed
)
".venv-web\Scripts\python.exe" -B -X utf8 python\ct_studio_current_release.py verify
if errorlevel 1 goto failed
echo.
echo Setup complete. PC browsing: start_ct_studio_preview.cmd
echo FPGA mode: pair the generated board package, then start_ct_studio_current.cmd
popd
exit /b 0
:failed
echo Setup failed. See docs\CT_Studio_latest_20261027.md
popd
pause
exit /b 1
