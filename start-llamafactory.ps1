$env:WSL_UTF8 = '1'
wsl -d Ubuntu -u lgh -- bash -lc 'cd /home/lgh/LLaMA-Factory && source /home/lgh/.venvs/llamafactory/bin/activate && GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui'
