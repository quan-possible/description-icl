#!/bin/bash
# The reasoning condition: gpt-oss-20b thinks before answering; 100 tasks, n = 1 and 4, seven conditions at noise 60.
set -eo pipefail
cd /content/work
python3 -u llm_test.py reason openai/gpt-oss-20b reason_gptoss20.npz --tasks 100 2>&1 | grep --line-buffered -v -e Warning -e Loading -e Fetching
