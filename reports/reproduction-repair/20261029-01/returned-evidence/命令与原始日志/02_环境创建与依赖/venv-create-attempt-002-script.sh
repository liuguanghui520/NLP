set -u
if [ -e /home/yilin/.venvs/nlp-reproduction-20261029-a02 ]; then echo target_exists; exit 2; fi
/home/yilin/.local/bin/uv venv --python 3.11.17 /home/yilin/.venvs/nlp-reproduction-20261029-a02
status=$?
echo "venv_create_exit_code=$status"
/home/yilin/.venvs/nlp-reproduction-20261029-a02/bin/python --version
version_status=$?
echo "python_version_exit_code=$version_status"
if [ "$status" -ne 0 ]; then exit "$status"; fi
exit "$version_status"
