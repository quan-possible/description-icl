#!/bin/bash
# Train and evaluate models on a Colab VM, one after another, then bundle
# their outputs. Usage: one_model.sh <p> <r> [<p> <r> ...]
# Seed SEED (default 0), 40k steps, bf16 autocast, prompts with 0 to 20 examples
# (K = 21). Every model is also evaluated at the other reliabilities of the grid.
set -eo pipefail
cd /content/work
export PYTHONPATH=/content/work/src
mkdir -p tmp/rq3
R=results/rq3-meta-trained
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
while (( $# )); do
  p=$1; r=$2; shift 2
  s=${SEED:-0}
  name="p${p}_r${r}_d5_s${s}"
  echo "model: $name"
  python3 $R/train.py --p $p --r $r --seed $s --steps 40000 --K 21 --amp 2>&1 | tee tmp/rq3/$name.log
  python3 $R/evaluate.py tmp/rq3/$name.pt | tee -a tmp/rq3/$name.log
  for pt in 1.0 0.99 0.9; do [[ $pt == $p ]] || python3 $R/evaluate.py tmp/rq3/$name.pt --p-test $pt | tee -a tmp/rq3/$name.log; done
  tar czf /content/${name}_results.tgz $R/${name}_*.csv tmp/rq3/$name.log
  tar czf /content/${name}_ckpt.tgz tmp/rq3/$name.pt
done
ls -la /content/*_results.tgz /content/*_ckpt.tgz
