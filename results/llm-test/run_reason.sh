#!/bin/bash
# gpt-oss-20b reasoning first (effort low, up to 4,096 new tokens, vLLM) on the reasoning cells of llm_test.py.
# Results: /content/reason_results.tgz.
set -eo pipefail
cd /content/work
pip install -q vllm 2>&1 | tail -2
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
python3 -u llm_test.py reason openai/gpt-oss-20b reason_gptoss20_4096.npz --budget 4096 2>&1 | grep --line-buffered -v -e Warning -e Fetching
tar czf /content/reason_results.tgz reason_gptoss20_4096.npz
