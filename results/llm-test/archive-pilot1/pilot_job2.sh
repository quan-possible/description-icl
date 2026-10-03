#!/bin/bash
# The settled prompt on the pilot's two models: everything stated, numbers so far in the user's turn, "Predict the next number."
set -eo pipefail
cd /content/work
run() { python3 -u pilot.py score "$@" --heads stated --chat 2>&1 | grep --line-buffered -v -e Warning -e Loading; }
run ibm-granite/granite-4.2-8b pilot_granite8_stated.npz
run openai/gpt-oss-20b pilot_gptoss20_stated.npz
