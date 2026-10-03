#!/bin/bash
# The written-answer readout (the number the model writes) on every model, 2,000 tasks, all 19 conditions,
# plus the paraphrase ablation (three other wordings of the description sentence, on 4 conditions and n = 1, 2, 4).
# One model failing does not stop the others.
cd /content/work
run() { python3 -u llm_test.py score "$@" --written --tasks 2000 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching || echo "FAILED: $*"; }
model() {  # repo tag [batch]
  local b=${3:-100}
  run $1 written_$2.npz --batch $b
  for k in 1 2 3; do run $1 written_$2_p$k.npz --batch $b --paraphrase $k --ablation; done
}
model openai/gpt-oss-20b gptoss20
model Qwen/Qwen3.5-9B qwen9
model microsoft/phi-4 phi4
model google/gemma-4-12B-it gemma12
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi-4 ~/.cache/huggingface/hub/models--google--gemma-4-12B-it ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B
model google/gemma-4-31B-it gemma31 25
rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-31B-it
model allenai/OLMo-2-0325-32B-Instruct olmo32 25
echo ALL DONE
