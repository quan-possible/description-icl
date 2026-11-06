#!/bin/bash
# The partial-description and dimension networks (2026-11-03): r = 0.01, p = 1, 40k steps, bf16 autocast, everything
# else as one_model.sh; each is evaluated at its training reliability only (p = 1).
#   partial: d = 5, the description gives m in directions 1-3 only (--k 3), 0 to 20 examples (K = 21)
#   dim:     d = 10, full description, 0 to 40 examples (K = 41)
# Usage: run_partial_dim.sh STEPS "KIND SEED" ["KIND SEED" ...]: the models one after another (two at once on one
# L4 ran each 3x slower). Results: /content/nets_results.tgz (CSVs, logs, checkpoints), refreshed after each model.
set -o pipefail
cd /content/work
export PYTHONPATH=/content/work/src
R=results/rq3-meta-trained
mkdir -p tmp/rq3
steps=$1; shift
echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for spec in "$@"; do
  set -- $spec
  case $1 in
    partial) args="--d 5 --k 3 --K 21"; name="p1.0_r0.01_d5_k3_s$2" ;;
    dim) args="--d 10 --K 41"; name="p1.0_r0.01_d10_s$2" ;;
  esac
  echo "model: $name"
  python3 $R/train.py --p 1.0 --r 0.01 --seed $2 --steps $steps --amp $args 2>&1 | tee tmp/rq3/$name.log \
    && python3 $R/evaluate.py tmp/rq3/$name.pt 2>&1 | tee -a tmp/rq3/$name.log || echo "FAILED: $name"
  tar czf /content/nets_results.tgz $R/p1.0_r0.01_d{5_k3,10}_s*_{regret,ess,trust}.csv tmp/rq3/*.log tmp/rq3/*.pt 2>/dev/null
done
ls -la /content/nets_results.tgz
