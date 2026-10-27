@echo off
setlocal
cd /d "%~dp0"
echo.
echo  LocalTrainer setup
echo  ------------------
echo  Creates .venv, installs PyTorch with CUDA 12.8 (about 3 GB) and the other packages.
echo.

set "PY="
py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
if not defined PY py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Install Python 3.12 or 3.13 from https://www.python.org/downloads/
  goto :fail
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment with %PY% ...
  %PY% -m venv .venv || goto :fail
)

".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
echo.
echo Installing PyTorch (CUDA 12.8 build for RTX 50-series and older cards) ...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
echo.
echo Installing LocalTrainer requirements ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
echo.
".venv\Scripts\python.exe" -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok, '-', torch.cuda.get_device_name(0) if ok else 'training will use the CPU')"
echo.
echo Setup finished. Start LocalTrainer with run.bat
if /i not "%~1"=="/nopause" pause
exit /b 0

:fail
echo.
echo Setup failed. Read the messages above, fix the problem and run setup.bat again.
if /i not "%~1"=="/nopause" pause
exit /b 1
