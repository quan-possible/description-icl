#!/bin/zsh
# run.sh [effort]: every prompt of $PROMPTS (default prompts.json), six at a time (ask.sh skips those already answered).
cd "$(dirname "$0")"
export EFFORT=${1:-low} PROMPTS=${PROMPTS:-prompts.json}
n=$(python3 -c "import json; print(len(json.load(open('$PROMPTS'))))")
seq 0 $((n-1)) | xargs -P 6 -n 1 ./ask.sh
echo "done $PROMPTS $EFFORT"
