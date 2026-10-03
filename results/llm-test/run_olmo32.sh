#!/bin/bash
set -eo pipefail
cd /content/work
python3 -u llm_test.py score allenai/OLMo-2-0325-32B-Instruct full_olmo32.npz --tasks 2000 --batch 25 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching
