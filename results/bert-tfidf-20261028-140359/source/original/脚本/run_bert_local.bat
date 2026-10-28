@echo off
title MacBERT Full Fine-tuning - local CPU - formal run
chcp 65001 > nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo   MacBERT-base Full Fine-tuning -- local CPU, FORMAL run
echo   Expected total time: about 26 hours on this laptop
echo     8 hyper-param configs + 5-point data curve + 1 control
echo   RESUMABLE: stop it any time, then run this file again
echo   and it continues where it stopped.
echo   Output dir: %~dp0src\results\results_BERT_FULL
echo   Log file  : %~dp0src\results\results_BERT_FULL\run_log.txt
echo ============================================================
echo.

REM Which Python to use.
REM This machine has two interpreters and they are NOT equivalent:
REM   D:\nlp_venv\Scripts\python.exe -> torch 2.10.0+cpu + transformers 5.20.2
REM   The default C:\...\Python311 interpreter CANNOT import torch:
REM     Windows Smart App Control blocks PyTorch unsigned DLLs,
REM     WinError 4551 "application control policy blocked this file"
REM   The venv also pins torchvision==0.25.0+cpu because transformers 5.x
REM     hard-imports torchvision and the D:\python torchvision 0.27.1
REM     requires torch 2.12.1 (we have 2.10.0) -> RuntimeError at import.
REM On a clean machine just run `pip install -r src\requirements.txt`
REM and delete the two set PY lines below.
set "PY=D:\nlp_venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not defined BERT_SMOKE set "BERT_SMOKE=0"
set "BERT_OUT_ROOT=%~dp0src\results"
set "BERT_CKPT_DIR=%~dp0"
if "%BERT_SMOKE%"=="0" (
  set "BERT_LOG=%BERT_OUT_ROOT%\results_BERT_FULL\run_log.txt"
) else (
  set "BERT_LOG=%BERT_OUT_ROOT%\results_BERT_SMOKE\run_log.txt"
)

echo [0/4] Checking that no other training run is active ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\check_no_other_run.ps1"
if errorlevel 1 (
    echo ERROR: another training run is already active. Aborting.
    pause
    exit /b 2
)

echo [1/4] Checking the interpreter and torch import ...
"%PY%" -c "import sys,torch,transformers; print('  python  :',sys.executable); print('  torch   :',torch.__version__); print('  transf  :',transformers.__version__)"
if errorlevel 1 (
    echo ERROR: this interpreter cannot import torch.
    echo        Edit the set PY line in this file and try again.
    pause
    exit /b 3
)

echo.
echo [2/4] Generating runnable script from train_bert.md ...
"%PY%" _extract_md_code.py train_bert.md _bert_run.py
if errorlevel 1 (
    echo ERROR: extraction failed, aborting.
    pause
    exit /b 1
)

echo.
echo [3/4] Training starts now. Do not close this window.
echo   interpreter: %PY%
echo   log file   : %BERT_LOG%
echo.
"%PY%" -u _bert_run.py

echo.
echo [4/4] Finished. exit code = %errorlevel%
echo   Results: %BERT_OUT_ROOT%\results_BERT_FULL
echo   Weights: %BERT_CKPT_DIR%bert_FULL.pt
echo   Log    : %BERT_LOG%
echo ============================================================
pause
