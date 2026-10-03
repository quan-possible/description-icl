#!/bin/zsh
# ask.sh K: send prompt K of prompts.json to Codex (gpt-5.6-luna), save the reply.
# EFFORT (default low) sets the reasoning effort; replies go to replies/ (low) or replies-$EFFORT/.
set -u
cd "$(dirname "$0")"
k=$1; e=${EFFORT:-low}; d=replies; l=logs; [ "$e" = low ] || { d=replies-$e; l=logs-$e; }
mkdir -p $d $l
[ -s $d/$k.txt ] && exit 0
python3 -c "import json,sys; print(json.load(open('prompts.json'))[$k]['prompt'], end='')" \
 | codex exec -m gpt-5.6-luna -s read-only --ephemeral --skip-git-repo-check -C /tmp \
     -c model_reasoning_effort="\"$e\"" -c project_doc_max_bytes=0 -c 'notify=[]' \
     --json -o $d/$k.txt - > $l/$k.jsonl 2>&1
