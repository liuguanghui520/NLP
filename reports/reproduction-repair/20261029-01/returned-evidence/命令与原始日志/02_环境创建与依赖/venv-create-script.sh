set -u
if [ -e /home/yilin/.venvs/nlp-reproduction-20261029 ]; then echo 'target_exists=1'; exit 2; fi
/home/yilin/.local/bin/uv venv --python 3.11.17 /home/yilin/.venvs/nlp-reproduction-20261029
code=$?
echo "venv_create_exit_code=$code"
/home/yilin/.venvs/nlp-reproduction-20261029/bin/python --version
echo "python_version_exit_code=$?"
exit $code
