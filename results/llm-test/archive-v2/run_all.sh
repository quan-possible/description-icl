#!/bin/bash
# The whole test on one Colab GPU: a float16-against-float32 check, then both models.
set -eo pipefail
cd /content/work
M=allenai/OLMo-2-1124-7B
python3 score_model.py $M check16.npz --tasks 32 --batch 16 --conditions none,valid10,invalid3 2>&1 | grep -v -e Warning -e Loading
python3 score_model.py $M check32.npz --tasks 32 --batch 8 --dtype float32 --conditions none,valid10,invalid3 2>&1 | grep -v -e Warning -e Loading
python3 score_model.py $M base.npz --tasks 2000 --batch 16 2>&1 | grep -v -e Warning -e Loading
python3 score_model.py $M-Instruct instruct.npz --tasks 2000 --batch 16 2>&1 | grep -v -e Warning -e Loading
