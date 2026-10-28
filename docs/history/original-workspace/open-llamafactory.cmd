@echo off
setlocal
set "WSL_UTF8=1"
wsl -d Ubuntu -u lgh -- bash -c "cd /home/lgh/LLaMA-Factory && source /home/lgh/.venvs/llamafactory/bin/activate && export USE_MODELSCOPE_HUB=1 && exec bash --norc -i"
if errorlevel 1 (
    echo.
    echo Failed to open the LLaMA-Factory environment.
    pause
)
endlocal
