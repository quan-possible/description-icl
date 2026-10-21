#!/bin/zsh
# EFFORT=e run.sh [first] [last] [parallel]: every prompt of prompts.json (default all, 8 at a time); ask.sh skips those answered.
cd "$(dirname "$0")"
n=$(python3 -c "import json; print(len(json.load(open('prompts.json'))))")
seq ${1:-0} ${2:-$((n-1))} | xargs -P ${3:-8} -n 1 ./ask.sh
echo "done ${EFFORT:-medium}: $(ls replies-${EFFORT:-medium}/*.json | wc -l) of $n $(date)"
