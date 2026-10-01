#!/bin/bash
# Continue a trained model for 40k more steps on fresh prompts (data seed 1,
# Adam restarted) and evaluate it, to check whether its gap to the exact
# learner is under-training. Usage: continue_model.sh <p> <r>
set -eo pipefail
cd /content/work
export PYTHONPATH=/content/work/src
R=results/rq3-meta-trained
base="p$1_r$2_d5_s0"; name="${base}_cont"
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)  continuing: $base"
python3 $R/train.py --p $1 --r $2 --seed 1 --steps 40000 --amp --init tmp/rq3/$base.pt --name $name 2>&1 | tee tmp/rq3/$name.log
python3 $R/evaluate.py tmp/rq3/$name.pt | tee -a tmp/rq3/$name.log
tar czf /content/${name}_results.tgz $R/${name}_*.csv tmp/rq3/$name.log
tar czf /content/${name}_ckpt.tgz tmp/rq3/$name.pt
ls -la /content/${name}_*.tgz
