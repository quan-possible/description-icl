#!/bin/bash
# What remains: OLMo 2 32B with small batches, then the reasoning condition on gpt-oss-20b.
set -eo pipefail
cd /content/work
bash run_olmo32.sh
bash run_reason.sh
