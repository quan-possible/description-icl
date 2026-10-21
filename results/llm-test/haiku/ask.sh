#!/bin/zsh
# ask.sh K: send prompt K of prompts.json to Claude Haiku 5.5 through the Claude Code CLI (subscription, one call per
# prompt, no tools, no project or user instructions, a one-line system prompt; EFFORT, default medium, is the CLI's
# reasoning effort), and save the CLI's JSON reply (result text and token usage) as replies-EFFORT/K.json. Skips
# prompts already answered.
set -u
cd "$(dirname "$0")"
k=$1; e=${EFFORT:-medium}; d=replies-$e
mkdir -p $d
[ -s $d/$k.json ] && exit 0
python3 -c "import json; print(json.load(open('prompts.json'))[$k]['prompt'], end='')" \
 | CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 claude -p --model claude-haiku-5-5 --effort $e \
     --system-prompt 'You are a helpful assistant.' --tools '' --setting-sources '' --strict-mcp-config \
     --disable-slash-commands --no-session-persistence --output-format json > $d/$k.tmp 2> $d/$k.err \
 && python3 -c "import json,sys; d=json.load(open('$d/$k.tmp')); sys.exit(d.get('is_error', False))" \
 && mv $d/$k.tmp $d/$k.json && rm -f $d/$k.err
