@echo off
REM VTES OBS Deploy Wrapper - Run from Developer Command Prompt for VS 2022
REM Usage: deploy.bat [options]
REM Options: -Configuration Release -StartServer -SkipBuild -SkipObsRestart

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0\deploy-windows.ps1" %*