#!/usr/bin/env bash
# Student-operated local API for the completed 0.6B Optuna exercise.
set -uo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
factory_root=/home/lgh/LLaMA-Factory
adapter_path="$factory_root/project-workspaces/qwen06-optuna/qwen06-optuna-20261025-175431-0a7f8f/trial-001/adapter"
if [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; then
    echo "Missing adapter: $adapter_path"
    exit 2
fi
source /home/lgh/.venvs/llamafactory/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export API_HOST=127.0.0.1
export API_PORT=8000
export API_MODEL_NAME=qwen3-0.6b-lora
# Local demonstration credential only; unrelated to any cloud API key.
export API_KEY=qwen06-local-demo
export API_VERBOSE=0
run_id="qwen06-api-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$project_root/results/qwen06-api"
run_root="$project_root/results/qwen06-api/$run_id"
mkdir "$run_root" || exit 2
date -Iseconds > "$run_root/started.txt"
printf '%s\n' "$adapter_path" > "$run_root/adapter_path.txt"
printf '%s\n' 'API_HOST=127.0.0.1' 'API_PORT=8000' 'API_MODEL_NAME=qwen3-0.6b-lora' 'template=qwen3' 'enable_thinking=False' 'infer_dtype=bfloat16' > "$run_root/settings.txt"
cp -- "${BASH_SOURCE[0]}" "$run_root/launch_snapshot.sh"
cd "$factory_root" || exit 2
echo "Serving local 0.6B rank16 adapter; log directory: $run_root"
llamafactory-cli api \
    --model_name_or_path /home/lgh/models/Qwen3-0.6B \
    --adapter_name_or_path "$adapter_path" \
    --finetuning_type lora \
    --template qwen3 \
    --enable_thinking False \
    --infer_backend huggingface \
    --infer_dtype bfloat16 \
    --trust_remote_code False \
    --do_sample False \
    --max_new_tokens 8 \
    --repetition_penalty 1.0 2>&1 | tee "$run_root/server.log"
server_exit=${PIPESTATUS[0]}
printf '%s\n' "$server_exit" > "$run_root/exit_code.txt"
date -Iseconds > "$run_root/ended.txt"
exit "$server_exit"
