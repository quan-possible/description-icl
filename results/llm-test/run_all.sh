#!/bin/bash
# The whole language-model test on one Colab GPU (G4 for the 32B model; 2,000 tasks each).
# OLMo 2 7B and 13B Instruct are scored too; they rarely answer with a number and are reported only for that.
set -eo pipefail
cd /content/work
run() { python3 -u llm_test.py score "$@" --tasks 2000 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching; }
run openai/gpt-oss-20b full_gptoss20.npz
run microsoft/phi-4 full_phi4.npz
run openbmb/MiniCPM5-2B full_minicpm2.npz
run allenai/OLMo-2-1124-7B-Instruct full_olmo7.npz
run allenai/OLMo-2-1124-13B-Instruct full_olmo13.npz
rm -rf ~/.cache/huggingface/hub/models--allenai--OLMo-2-1124-13B-Instruct
python3 -u llm_test.py score allenai/OLMo-2-0325-32B-Instruct full_olmo32.npz --tasks 2000 --batch 25 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching  # smaller batches: the 64-number prompts do not fit at 100
