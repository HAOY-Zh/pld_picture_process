@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  echo Run setup_ct_studio_current.cmd first.
  popd
  exit /b 1
)
if not exist "data\ct_studio_current_20261027\host_config.json" (
  echo Current runtime missing. Run setup_ct_studio_current.cmd first.
  popd
  exit /b 1
)
".venv-web\Scripts\python.exe" -B -u -X utf8 python\start_ct_studio_demo.py --config data\ct_studio_current_20261027\host_config.json --port 8099 %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
