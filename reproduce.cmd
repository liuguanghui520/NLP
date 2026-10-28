@echo off
setlocal
python "%~dp0reproduce.py" %*
exit /b %errorlevel%
