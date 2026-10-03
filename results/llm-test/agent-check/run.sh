#!/bin/zsh
# run.sh [effort]: all 600 prompts, six at a time (ask.sh skips those already answered).
cd "$(dirname "$0")"
export EFFORT=${1:-low}
seq 0 599 | xargs -P 6 -n 1 ./ask.sh
echo "done $EFFORT"
