#!/bin/bash
# The whole RQ3 run on one Colab VM: pilot, step choice, grid, bundle outputs.
set -eo pipefail
cd /content/work
export PY=python3 PYTHONPATH=/content/work/src AMP=--amp
mkdir -p tmp/rq3
R=results/rq3-meta-trained
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# pilot with snapshots every 5k steps
$PY $R/train.py --p 1.0 --r 0.05 --seed 0 --steps 40000 --snapshots --amp 2>&1 | tee tmp/rq3/pilot.log
for f in tmp/rq3/p1.0_r0.05_d5_s0_step*.pt; do
  mv "$f" "tmp/rq3/pilot_${f##*_}"
  $PY $R/evaluate.py "tmp/rq3/pilot_${f##*_}" --S 20000 > /dev/null
done
rm tmp/rq3/p1.0_r0.05_d5_s0.pt
STEPS=$($PY $R/choose_steps.py $R/pilot_step*_regret.csv)
echo "chosen steps: $STEPS" | tee -a tmp/rq3/pilot.log
cat $R/pilot.csv
# the grid at the chosen step count
zsh $R/grid.sh $STEPS 2>/dev/null || bash $R/grid.sh $STEPS
# bundle everything to bring home
tar czf /content/rq3_results.tgz $R/*.csv tmp/rq3/*.log tmp/rq3/*.done
tar czf /content/rq3_checkpoints.tgz tmp/rq3/p*_d5_s?.pt
ls -la /content/rq3_*.tgz
