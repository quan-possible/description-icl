"""tool_use.py: how often GPT-6 Sol ran a shell command (Python, awk) in each run made through sol-check/ask.sh before
4 November 2026, when the harness still gave it a shell; per run, and per condition of the main runs.

    python3 tool_use.py
"""
import collections
import json
import pathlib

import llm_test as T

H = pathlib.Path(__file__).parent
RUNS = ["sol-check/replies-gpt-6-sol-none", "sol-check/replies-gpt-6-sol-low", "sol-check/replies-gpt-6-sol-low-paraphrase",
        "stated-fallback/replies-sol-low", "agent-check/replies-none", "../filing-policy/archive/pilot/replies-sol-none",
        "../filing-policy/archive/pilot/replies-sol-low", "../filing-policy/archive/pilot/props-sol-none"]
used = lambda f: '"command_execution"' in f.read_text(errors="ignore")
prompts = json.load(open(H / "haiku/prompts.json"))
for d in RUNS:
    logs = sorted((H / d).glob("*.jsonl"))
    print(f"{d}: {sum(map(used, logs))} of {len(logs)} replies ran a command")
    if d.startswith("sol-check/replies-gpt-6-sol-") and "paraphrase" not in d:
        c, tot = collections.Counter(), collections.Counter()
        for f in logs:
            cond, n = T.REASON_CELLS[prompts[int(f.stem)]["cell"]]
            key = ("no description" if cond is T.REASON_CONDS[0] else f"sd {cond[1]}" + (f", {round(100 * (1 - cond[3]))}% wrong" if cond[3] else ""), n)
            tot[key] += 1
            c[key] += used(f)
        print("   " + "; ".join(f"{k[0]}, n = {k[1]}: {c[k]}/{tot[k]}" for k in sorted(tot, key=str) if c[k] >= 5))
