@echo off
setlocal

REM Windows-safe local launcher. Do not use plain "python -m uvicorn" here:
REM it may pick Python 3.13 from PATH and crash FastAPI/Pydantic.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1" %*
