@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_local.ps1" %*
