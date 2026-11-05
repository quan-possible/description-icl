"""report.py DIR ...: the filing-policy runs scored against the exact oracle.

For each run and the oracle: the mean Brier loss per cell, the ESS of each note against the same learner's no-note curve
(Definition 1, and the monotone reading as a check), each with a 95% bootstrap interval over tasks; for the unreliable
note, the share of queries closer to the updating oracle than to the fixed-chance rule; missing replies; and the share of
replies that reasoned. The oracle's ESS is read on the models' grid, and on every n from 0 to 32 ("exact")."""
import json
import pathlib
import sys

import numpy as np

import filing as F

H = pathlib.Path(__file__).parent
B = 1000
tasks = F.make_tasks()
rows = json.load(open(H / "prompts.json"))
NOTES = [c for c in F.CELLS if c[0] != "none"]


TOOLS = ('"command_execution"', '"mcp_tool_call"', '"collab_tool_call"', '"web_search"', '"file_change"')


def reply(d, k):
    """The reply text of prompt k in folder d and its reasoning tokens (Codex .txt/.jsonl, or Claude Code .json). A Codex
    reply that called a tool counts as missing: the model may have computed its answer in code."""
    f = H / d / str(k)
    if f.with_suffix(".txt").exists() and f.with_suffix(".txt").stat().st_size:
        log = open(f.with_suffix(".jsonl"), errors="ignore").read() if f.with_suffix(".jsonl").exists() else ""
        if any(t in log for t in TOOLS):
            return None, np.nan
        tok = next((json.loads(l)["usage"]["reasoning_output_tokens"] for l in log.splitlines() if '"turn.completed"' in l), np.nan)
        return open(f.with_suffix(".txt")).read(), tok
    if f.with_suffix(".json").exists():
        j = json.load(open(f.with_suffix(".json")))
        return j["result"], j["usage"].get("output_tokens_details", {}).get("thinking_tokens", np.nan)
    return None, np.nan


def fixed(task, n):
    """The fixed-chance rule: the stated 50% kept as the weight on the note's policies, whatever the demonstrations show."""
    xs, ys = task["x"][:n], [F.label(task["policy"], x) for x in task["x"][:n]]
    fit = lambda pols: [p for p in pols if all(F.label(p, x) == y for x, y in zip(xs, ys))]
    share = lambda pols: np.array([np.mean([F.label(p, q) for p in pols]) for q in task["x"][F.NMAX:]])
    fd, fa = fit(F.described(task, "unreliable")), fit(list(F.POLICIES))
    return F.P_STATED * share(fd or fa) + (1 - F.P_STATED) * share(fa)


def ess_both(loss, grid=F.GRID):
    """{note cell: (Definition 1, monotone reading)} from {cell: mean loss}."""
    curve = [loss[("none", n)] for n in grid]
    mono = F.monotone(curve)
    return {c: (F.ess(grid, curve, loss[c], c[1]), F.ess(grid, mono, loss[c], c[1])) for c in NOTES}


def boot(per_task, stat):
    """95% percentile interval of stat({cell: mean over resampled tasks})."""
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(B):
        i = rng.integers(0, F.TASKS, F.TASKS)
        draws.append(stat({c: np.nanmean(v[i]) for c, v in per_task.items()}))
    return draws


def fmt(e):
    return f"{e:5.1f}" if np.isfinite(e) else (" >max" if e > 0 else " <min")


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{fmt(lo).strip()}, {fmt(hi).strip()}]"


def show(name, per_task, missing=None, think=None):
    loss = {c: np.nanmean(v) for c, v in per_task.items()}
    lb = boot(per_task, lambda m: m)
    eb = boot(per_task, ess_both)
    e = ess_both(loss)
    print(f"\n{name}" + (f"  ({missing} replies with a missing probability)" if missing is not None else ""))
    for c in F.CELLS:
        lo, hi = np.percentile([d[c] for d in lb], [2.5, 97.5])
        s = f"  {c[0]:10s} n={c[1]:2d}  loss {loss[c]:.3f} [{lo:.3f}, {hi:.3f}]"
        if c in e:
            s += f"   ESS {fmt(e[c][0])} {ci([d[c][0] for d in eb])}   monotone {fmt(e[c][1])} {ci([d[c][1] for d in eb])}"
        print(s)
    if think is not None and np.isfinite(think).any():
        t = think[np.isfinite(think)]
        print(f"  replies that reasoned: {np.mean(t > 0):.0%} (median {np.median(t):.0f} tokens)")
    return loss


def per_task_oracle():
    out = {c: np.array([np.mean((F.oracle(t, *c) - F.truth(t)) ** 2) for t in tasks]) for c in F.CELLS}
    fine = {n: np.mean([np.mean((F.oracle(t, "none", n) - F.truth(t)) ** 2) for t in tasks]) for n in range(F.NMAX + 1)}
    return out, fine


def run(d):
    per_task = {c: np.full(F.TASKS, np.nan) for c in F.CELLS}
    think, missing, rule = [], 0, {}
    for k, r in enumerate(rows):
        txt, tok = reply(d, k)
        task, c = tasks[r["task"]], (r["kind"], r["n"])
        p = F.read(txt) if txt else np.full(F.QUERIES, np.nan)
        missing += int(np.isnan(p).any())
        think.append(tok)
        if not np.isnan(p).all():
            per_task[c][r["task"]] = np.nanmean((p - F.truth(task)) ** 2)
        if r["kind"] == "unreliable" and r["n"]:
            o, fx = F.oracle(task, "unreliable", r["n"]), fixed(task, r["n"])
            for q in range(F.QUERIES):
                if abs(o[q] - fx[q]) >= 0.05 and not np.isnan(p[q]):
                    rule.setdefault(r["n"], []).append(abs(p[q] - o[q]) < abs(p[q] - fx[q]))
    return per_task, missing, np.array(think, dtype=float), rule


def props():
    ps = json.load(open(H / "props.json"))
    right, n = np.zeros(6), 0
    for k, r in enumerate(ps):
        txt, _ = reply("props-sol-none", k)
        digs = [int(c) for c in txt.split() if c in ("0", "1")][:6] if txt else []
        if len(digs) == 6:
            right += np.array(digs) == np.array(r["x"])
            n += 1
    print(f"Property check (GPT-6 Sol, no reasoning, {n} messages): share read correctly per property",
          " ".join(f"{a:.2f}" for a in right / max(n, 1)))


if __name__ == "__main__":
    props()
    po, fine = per_task_oracle()
    oloss = show("Oracle (exact Bayes), ESS on the models' grid", po)
    ex = ess_both(oloss | {("none", n): v for n, v in fine.items()}, grid=range(F.NMAX + 1))
    print("  ESS read on every n from 0 to 32: " + "  ".join(f"{k[:4]} {n}: {ex[(k, n)][0]:.1f}" for k, n in NOTES))
    for d in sys.argv[1:]:
        per_task, missing, think, rule = run(d)
        show(d, per_task, missing, think)
        for n, v in sorted(rule.items()):
            print(f"  unreliable n={n}: closer to the updating oracle than to the fixed-chance rule in {sum(v)} of {len(v)} queries ({np.mean(v):.0%})")
