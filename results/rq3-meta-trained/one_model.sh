#!/bin/bash
# Train and evaluate one model on a Colab VM, then bundle its outputs.
# Usage: one_model.sh <p> <r>    (seed 0, 40k steps, bf16 autocast)
set -eo pipefail
cd /content/work
export PYTHONPATH=/content/work/src
mkdir -p tmp/rq3
R=results/rq3-meta-trained
name="p$1_r$2_d5_s0"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)  model: $name"
python3 $R/train.py --p $1 --r $2 --seed 0 --steps 40000 --amp 2>&1 | tee tmp/rq3/$name.log
python3 $R/evaluate.py tmp/rq3/$name.pt | tee -a tmp/rq3/$name.log
[[ $1 == 1.0 ]] && python3 $R/evaluate.py tmp/rq3/$name.pt --p-test 0.9 | tee -a tmp/rq3/$name.log
tar czf /content/${name}_results.tgz $R/${name}_*.csv tmp/rq3/$name.log
tar czf /content/${name}_ckpt.tgz tmp/rq3/$name.pt
ls -la /content/${name}_*.tgz
