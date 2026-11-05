#!/bin/zsh
# run.sh: prompts.json (make_prompts.py) on GPT-6 Sol at low effort (Codex CLI) and Claude Haiku 5.5 at medium effort
# (Claude Code CLI), both on the subscription; the ask.sh harnesses skip answered prompts.
cd "$(dirname "$0")"; F=$PWD
seq 0 1199 | PROMPTS=$F/prompts.json OUT=$F/replies-sol-low EFFORT=low xargs -P 20 -n 1 ../sol-check/ask.sh &
seq 0 1199 | PROMPTS=$F/prompts.json OUT=$F/replies-haiku-medium EFFORT=medium xargs -P 8 -n 1 ../haiku/ask.sh &
wait
echo "done $(date)"
