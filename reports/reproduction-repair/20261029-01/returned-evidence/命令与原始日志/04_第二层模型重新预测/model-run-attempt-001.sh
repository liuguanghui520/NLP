#!/usr/bin/env bash
set -euo pipefail

source /home/yilin/.venvs/nlp-reproduction-20261029-a02/bin/activate
cd "/mnt/d/NLPP/复现证据/独立工作副本/提交包复现副本"
python src/run.py \
  --models \
  --paths src/project/configs/reproduce-models.local.json \
  --work-dir "/mnt/d/NLPP/reproduction-runs/L2-attempt-001-20261029T1648"
