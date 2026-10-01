#!/bin/zsh
# Train the four models, one seed each (decision 20), and evaluate them;
# p = 1 models are also evaluated at p = 0.9. SEEDS="0 1 2" adds seeds.
# Usage: results/rq3-meta-trained/grid.sh <steps>
#   (run inside tmux: tmux new -d -s rq3grid "results/rq3-meta-trained/grid.sh 30000")
set -e
cd "$(dirname "$0")/../.."
STEPS=${1:?steps}
PY=${PY:-"uv run python"}   # on Colab: PY=python3
AMP=${AMP:-}                  # on a CUDA GPU: AMP=--amp (bf16 autocast)
R=results/rq3-meta-trained
# The pilot (p = 1, r = 0.05, seed 0) snapshot at STEPS is the same model the
# grid would train for that configuration: same seed, same sampling order,
# stopped at the same step. Reuse it.
if [[ -f tmp/rq3/pilot_step$STEPS.pt && ! -f tmp/rq3/p1.0_r0.05_d5_s0.pt ]]; then
  cp tmp/rq3/pilot_step$STEPS.pt tmp/rq3/p1.0_r0.05_d5_s0.pt
  echo "reused pilot snapshot at step $STEPS as p1.0_r0.05_d5_s0" | tee -a tmp/rq3/grid.log
fi
run() {  # p r seed
  name="p$1_r$2_d5_s$3"
  [[ -f tmp/rq3/$name.done ]] && return
  [[ -f tmp/rq3/$name.pt ]] || $PY $R/train.py --p $1 --r $2 --seed $3 --steps $STEPS $AMP 2>&1 | tee -a tmp/rq3/grid.log
  $PY $R/evaluate.py tmp/rq3/$name.pt >> tmp/rq3/grid.log 2>&1
  [[ $1 == 1.0 ]] && $PY $R/evaluate.py tmp/rq3/$name.pt --p-test 0.9 >> tmp/rq3/grid.log 2>&1
  touch tmp/rq3/$name.done
}
for s in ${SEEDS:-0}; do
  run 1.0 0.05 $s   # precise, reliable (+ evaluated at p = 0.9)
  run 1.0 0.5 $s    # coarse, reliable
  run 0.9 0.05 $s   # trained at p = 0.9
  run 0.9 0.5 $s
  echo "seed $s done $(date)" | tee -a tmp/rq3/grid.log
done
echo "grid done $(date)" | tee -a tmp/rq3/grid.log
