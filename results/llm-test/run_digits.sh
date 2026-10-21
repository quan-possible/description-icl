#!/bin/bash
# The distribution readout for the three models whose tokenizers write numbers digit by digit (--digits: Qwen3.5 9B,
# Gemma 4 12B, Gemma 4 31B), on the main conditions and the specificity control, 2,000 tasks, every n; and the prompt
# that also states the loss (--objective) on the ablation's conditions and n = 1, 2, 4, for every model with its
# distribution readout. run_digits.sh small | large | objective. One model failing does not stop the others.
# Results: /content/digits_results.tgz.
cd /content/work
run() { python3 -u llm_test.py score "$@" --tasks 2000 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching || echo "FAILED: $*"; }
gone() { rm -rf ~/.cache/huggingface/hub/models--$1; }
digits() {  # repo tag batch
  run $1 digits_$2.npz --digits --batch $3
  run $1 control_digits_$2.npz --digits --control --batch $3
  run $1 objective_$2.npz --digits --objective --ablation --batch $3
}
case $1 in
  small)
    digits Qwen/Qwen3.5-9B qwen9 8
    gone Qwen--Qwen3.5-9B
    digits google/gemma-4-12B-it gemma12 50 ;;
  large)
    digits google/gemma-4-31B-it gemma31 20 ;;
  objective)
    run openai/gpt-oss-20b objective_gptoss20.npz --objective --ablation
    gone openai--gpt-oss-20b
    run microsoft/phi-4 objective_phi4.npz --objective --ablation
    gone microsoft--phi-4
    run allenai/OLMo-2-0325-32B-Instruct objective_olmo32.npz --objective --ablation --batch 25 ;;
esac
tar czf /content/digits_results.tgz *_*.npz
echo ALL DONE
