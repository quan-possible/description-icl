#!/bin/bash
# Run the exact-learner computations on a Colab CPU VM and bundle the CSVs.
set -eo pipefail
cd /content/work
export PYTHONPATH=/content/work/src
python3 -c "import scipy, numpy; print('scipy', scipy.__version__)"
python3 results/rq1-single-query-gap/run.py 2>&1 | tail -3
python3 results/rq2-reliability/run.py 2>&1 | tail -3
python3 results/rq2-reliability/worth.py 2>&1 | tail -8
tar czf /content/exact_results.tgz results/rq1-single-query-gap/ess.csv results/rq2-reliability/ess_map.csv results/rq2-reliability/trust.csv results/rq2-reliability/tipping.csv results/rq2-reliability/worth.csv
ls -la /content/exact_results.tgz
