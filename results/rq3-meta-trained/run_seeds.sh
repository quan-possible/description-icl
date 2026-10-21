#!/bin/bash
# More training seeds for the networks of Figure 3 and Table 1, two one_model.sh queues at once on one VM.
# Usage: run_seeds.sh "SEED p r [p r ...]" "SEED p r [p r ...]". Results: /content/seeds_results.tgz.
cd /content/work
for q in "$@"; do
  set -- $q
  s=$1; shift
  SEED=$s bash results/rq3-meta-trained/one_model.sh "$@" > /content/queue_s${s}_p${1}_r${2}.log 2>&1 &
done
wait
tar czf /content/seeds_results.tgz /content/*_results.tgz /content/queue_*.log
ls -la /content/*_results.tgz
