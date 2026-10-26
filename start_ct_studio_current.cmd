@echo off
setlocal
pushd "%~dp0"
".venv-web\Scripts\python.exe" -B -u -X utf8 python\start_ct_studio_demo.py --config data\ct_studio_current_20261026\host_config.json --port 8099 %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
