#!/bin/zsh
# Train the grid in the paper's readout order, one model after another, and
# evaluate each one. Usage: results/rq3-meta-trained/grid.sh <steps>
#   (run inside tmux: tmux new -d -s rq3grid "results/rq3-meta-trained/grid.sh 30000")
set -e
cd "$(dirname "$0")/../.."
STEPS=${1:?steps}
PY=${PY:-"uv run python"}   # on Colab: PY=python3
AMP=${AMP:-}                  # on a CUDA GPU: AMP=--amp (bf16 autocast)
run() {  # p r seed
  name="p$1_r$2_d5_s$3"
  [[ -f tmp/rq3/$name.done ]] && return
  $PY results/rq3-meta-trained/train.py --p $1 --r $2 --seed $3 --steps $STEPS $AMP 2>&1 | tee -a tmp/rq3/grid.log
  $PY results/rq3-meta-trained/evaluate.py tmp/rq3/$name.pt >> tmp/rq3/grid.log 2>&1
  [[ $1 == 1.0 ]] && $PY results/rq3-meta-trained/evaluate.py tmp/rq3/$name.pt --p-test 0.9 >> tmp/rq3/grid.log 2>&1
  touch tmp/rq3/$name.done
}
for s in 0 1 2; do run 1.0 0.05 $s; done   # precise, reliable (+ evaluated at p = 0.9)
for s in 0 1 2; do run 1.0 0.5 $s; done    # coarse, reliable
for s in 0 1 2; do run 0.9 0.05 $s; done   # trained at p = 0.9
for s in 0 1 2; do run 0.9 0.5 $s; done
echo "grid done $(date)" | tee -a tmp/rq3/grid.log
