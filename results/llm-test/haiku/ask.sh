#!/bin/zsh
# ask.sh K: send prompt K (CLAUDE_BIN: the CLI or account wrapper, default claude; MAX_THINKING_TOKENS=0 in the environment turns thinking off) of $PROMPTS (default prompts.json) to $MODEL (default Claude Haiku 5.5) through the Claude Code CLI (subscription, one call per
# prompt, no tools, no project or user instructions, a one-line system prompt; EFFORT, default medium, is the CLI's
# reasoning effort), and save the CLI's JSON reply (result text and token usage) as $OUT/K.json (default replies-EFFORT). Skips
# prompts already answered.
set -u
cd "$(dirname "$0")"
k=$1; e=${EFFORT:-medium}; m=${MODEL:-claude-haiku-5-5}; d=${OUT:-replies-$e}; p=${PROMPTS:-prompts.json}
mkdir -p $d
[ -s $d/$k.json ] && exit 0
python3 -c "import json; print(json.load(open('$p'))[$k]['prompt'], end='')" \
 | CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 ${CLAUDE_BIN:-claude} -p --model $m --effort $e \
     --system-prompt 'You are a helpful assistant.' --tools '' --setting-sources '' --strict-mcp-config \
     --disable-slash-commands --no-session-persistence --output-format json > $d/$k.tmp 2> $d/$k.err \
 && python3 -c "import json,sys; d=json.load(open('$d/$k.tmp')); sys.exit(d.get('is_error', False))" \
 && mv $d/$k.tmp $d/$k.json && rm -f $d/$k.err
