#!/bin/zsh
# run.sh: the matched stated-reliability design on the subscriptions (no Colab). The irrelevant arm (prompts.json,
# make_prompts.py) for GPT-6 Sol at low effort and Claude Haiku 5.5 at medium effort, whose relevant arm is already in
# ../stated-fallback; then both arms for GPT-6 Sol without reasoning. A call that hits a rate limit leaves no reply, so
# each run repeats, 15 minutes apart, until every prompt has one. Haiku runs on the account wrapper $CLAUDE_BIN.
cd "$(dirname "$0")"; F=$PWD; REL=$F/../stated-fallback
go() {  # go ASK PROMPTS OUT EFFORT EXT PARALLEL
  n=$(python3 -c "import json; print(len(json.load(open('$2'))))")
  for pass in $(seq 60); do
    left=($(for k in $(seq 0 $((n-1))); do [ -s $3/$k.$5 ] || echo $k; done))
    echo "$3 pass $pass: ${#left} left $(date)"
    [ ${#left} = 0 ] && return
    [ $pass = 1 ] || sleep 900
    print -l $left | PROMPTS=$2 OUT=$3 EFFORT=$4 xargs -P $6 -n 1 $1
  done
}
export CLAUDE_BIN=${CLAUDE_BIN:-claude-calgary}
go ../haiku/ask.sh $F/prompts.json $F/replies-haiku-medium medium json 6 &
go ../sol-check/ask.sh $F/prompts.json $F/replies-sol-low low txt 12
go ../sol-check/ask.sh $F/prompts.json $F/replies-sol-none none txt 12
go ../sol-check/ask.sh $REL/prompts.json $REL/replies-sol-none-bare none txt 12
wait
echo "done $(date)"
