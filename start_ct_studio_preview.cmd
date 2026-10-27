@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  echo Run setup_ct_studio_current.cmd first.
  popd
  exit /b 1
)
".venv-web\Scripts\python.exe" -B -u -X utf8 python\start_ct_studio_demo.py --desktop --port 8099 %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
