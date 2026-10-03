#!/bin/bash
# The instruction pilot on one Colab GPU: small current models, plain text and chat template.
set -eo pipefail
cd /content/work
run() { python3 -u pilot.py score "$@" 2>&1 | grep --line-buffered -v -e Warning -e Loading; }
run ibm-granite/granite-4.2-8b pilot_granite8.npz
run ibm-granite/granite-4.2-8b pilot_granite8_chat.npz --chat
run openai/gpt-oss-20b pilot_gptoss20.npz
run openai/gpt-oss-20b pilot_gptoss20_chat.npz --chat
