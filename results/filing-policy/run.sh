#!/bin/zsh
# run.sh: prompts.json (200 tasks) on GPT-6 Sol without reasoning and at low effort (Codex CLI) and Claude Haiku 5.5 at
# medium effort (Claude Code CLI), through the ../llm-test harnesses (bare calls, no tools), which skip answered prompts. A failed call leaves no
# reply, so each run repeats until every prompt has one, up to twelve passes, fifteen minutes apart. Codex answered 429
# (rate limit) at 20 parallel calls on the earlier account; on the Pro account (from 19:50, 4 November 2026) Sol runs
# 20 at a time.
# The property check (props.json, first 96 messages) ran with the pilot; see archive/pilot/run.sh.
cd "$(dirname "$0")"; F=$PWD; S=../llm-test/sol-check/ask.sh; H=../llm-test/haiku/ask.sh
n=$(python3 -c "import json; print(len(json.load(open('prompts.json'))))")
todo() { for k in $(seq 0 $((n-1))); do [ -s $1/$k.$2 ] || echo $k; done; }
go() {  # go DIR EXT PARALLEL HARNESS EFFORT
  for pass in $(seq 12); do
    left=$(todo $1 $2 | wc -l | tr -d ' '); echo "$1 pass $pass: $left left $(date)"
    [ $left = 0 ] && return
    [ $pass = 1 ] || sleep 900
    todo $1 $2 | PROMPTS=$F/prompts.json OUT=$F/$1 EFFORT=$5 xargs -P $3 -n 1 $4
  done
  echo "$1: $(todo $1 $2 | wc -l | tr -d ' ') still missing after twelve passes"
}
go replies-sol-none txt 20 $S none &
go replies-sol-low txt 20 $S low &
go replies-haiku-medium json 8 $H medium &
wait
echo "done $(date)"
