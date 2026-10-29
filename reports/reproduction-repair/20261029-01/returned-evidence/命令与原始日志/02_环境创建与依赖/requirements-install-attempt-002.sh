#!/usr/bin/env bash
set -euo pipefail

/home/yilin/.local/bin/uv pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --python /home/yilin/.venvs/nlp-reproduction-20261029-a02/bin/python \
  --requirement /mnt/d/NLPP/复现证据/独立工作副本/提交包复现副本/src/project/requirements.integration.lock
