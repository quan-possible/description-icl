#!/bin/zsh
# run_opus.sh: Claude Opus 4.6 in the matched stated-reliability design, through the Claude Code CLI on the account
# wrapper $CLAUDE_BIN (default claude-aclmr): at high effort with reasoning, the irrelevant arm (prompts.json) and what
# is missing of the relevant arm (../stated-fallback/prompts.json, 232 of 1,200 were done on 4 November); then the
# irrelevant arm without reasoning (MAX_THINKING_TOKENS=0, same effort). Repeats 15 minutes apart until complete.
cd "$(dirname "$0")"; F=$PWD; REL=$F/../stated-fallback
export CLAUDE_BIN=${CLAUDE_BIN:-claude-aclmr} MODEL=claude-opus-4-6
go() {  # go PROMPTS OUT PARALLEL
  n=$(python3 -c "import json; print(len(json.load(open('$1'))))")
  for pass in $(seq 60); do
    left=($(for k in $(seq 0 $((n-1))); do [ -s $2/$k.json ] || echo $k; done))
    echo "$2 pass $pass: ${#left} left $(date)"
    [ ${#left} = 0 ] && return
    [ $pass = 1 ] || sleep 900
    print -l $left | PROMPTS=$1 OUT=$2 EFFORT=high xargs -P $3 -n 1 ../haiku/ask.sh
  done
}
go $F/prompts.json $F/replies-opus46-high 4
go $REL/prompts.json $REL/replies-opus46-high 4
MAX_THINKING_TOKENS=0 go $F/prompts.json $F/replies-opus46-none 4
echo "done $(date)"
