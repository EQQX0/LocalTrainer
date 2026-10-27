@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment is missing. Running setup.bat first ...
  call setup.bat /nopause || (pause & exit /b 1)
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 (
  echo.
  echo LocalTrainer closed with an error. The messages above show what went wrong.
  pause
)
