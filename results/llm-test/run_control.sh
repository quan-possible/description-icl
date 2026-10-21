#!/bin/bash
# The specificity control (CONTROL in llm_test.py: the stated value and the numbers fixed, only the stated sd changes),
# 2,000 tasks, every n, on the six reported models with the readout Table 2 uses (and gpt-oss-20b under both).
# run_control.sh [written]: the distribution readout's three models, or the written readout's four. One model failing
# does not stop the others. Results: /content/control_results.tgz.
cd /content/work
run() { python3 -u llm_test.py score "$@" --control --tasks 2000 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching || echo "FAILED: $*"; }
gone() { rm -rf ~/.cache/huggingface/hub/models--$1; }
if [ "$1" = written ]; then
  run openai/gpt-oss-20b control_written_gptoss20.npz --written
  gone openai--gpt-oss-20b
  run Qwen/Qwen3.5-9B control_written_qwen9.npz --written
  gone Qwen--Qwen3.5-9B
  run google/gemma-4-12B-it control_written_gemma12.npz --written
  gone google--gemma-4-12B-it
  run google/gemma-4-31B-it control_written_gemma31.npz --written --batch 25
else
  run openai/gpt-oss-20b control_gptoss20.npz
  gone openai--gpt-oss-20b
  run microsoft/phi-4 control_phi4.npz
  gone microsoft--phi-4
  run allenai/OLMo-2-0325-32B-Instruct control_olmo32.npz --batch 25
fi
tar czf /content/control_results.tgz control_*.npz
echo ALL DONE
