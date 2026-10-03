#!/bin/bash
# Both versions of one model size on a Colab GPU. Usage: run_size.sh REPO TAG BATCH
# (REPO-Instruct is the instruction-tuned version). The base weights are deleted
# before the second download to stay inside the disk.
set -eo pipefail
cd /content/work
python3 -u score_model.py $1 base$2.npz --tasks 2000 --batch $3 2>&1 | grep --line-buffered -v -e Warning -e Loading
rm -rf ~/.cache/huggingface/hub/models--${1//\//--}
python3 -u score_model.py $1-Instruct instruct$2.npz --tasks 2000 --batch $3 2>&1 | grep --line-buffered -v -e Warning -e Loading
