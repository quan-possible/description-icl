"""The filing-policy task: a semantic counterpart of the paper's language-model test, with an exact Bayesian reference.

A company files each message into folder A or B by a hidden policy. A message has six binary properties, written in
ordinary language. A policy uses exactly two properties and requires each to be present or absent; a message goes into
A when both requirements hold. The prompt states this family and that the policy is uniform over its 60 members, so
the Bayes-optimal probability of A is a count over the policies consistent with what the prompt says.

Descriptions (the "policy note"):
  none       no note
  broad      names one of the two properties                                  (20 policies left)
  specific   one property and its required value, and the other property      (2 policies left)
  unreliable a specific note that the prompt says has a 50% chance of being wrong, in which case the policy is any of
             the 60, equally likely; it is in fact wrong half the time, when it describes an independent policy

    python3 filing.py prompts        # writes prompts.json and props.json (the property check)
    python3 filing.py oracle         # exact oracle losses and ESS on the pilot's tasks

report.py scores the replies.
"""
import itertools
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
PROPS = ["the sender is outside the organization", "it requests a payment", "it concerns travel",
         "it requests computer access", "it has a deadline within a day", "it contains personal contact information"]
# For each property, phrasings for its presence and its absence; every message states each property one way or the other.
PHRASES = [
    (["I'm a contractor from another company.", "I work for an outside supplier.", "I'm writing from a partner firm."],
     ["I'm on the sales team here.", "I work in our finance department.", "I'm a colleague from the Toronto office."]),
    (["Could you settle the outstanding invoice?", "Please send the payment for last month's order.", "Can you pay the attached bill?"],
     ["No payment is needed for this.", "This isn't about money.", "There's nothing to pay here."]),
    (["I need to change my flight reservation.", "Can you book a hotel for the conference trip?", "My train ticket for next week needs rebooking."],
     ["I'm not traveling for this.", "This has nothing to do with travel.", "No trip is involved."]),
    (["Please enable my account.", "I need access to the shared drive.", "Could you reset my password?"],
     ["I don't need any system access.", "My accounts are all working fine.", "No login changes are needed."]),
    (["I need this before the end of today.", "Please handle this within the next few hours.", "I need this within the hour."],
     ["There's no rush on this.", "Next week is fine.", "Whenever you have time is fine."]),
    (["You can reach me at 555-0142.", "My direct line is 555-0187.", "You can also call my mobile, 555-0163."],
     ["Just reply to this thread if you need me.", "I'd rather not share my number.", "Reply here if anything is unclear."]),
]
# Policy note: how each property reads in a description, and its required value.
NAMES = ["whether the sender is outside the organization", "whether the message requests a payment", "whether it concerns travel",
         "whether it requests computer access", "whether it has a deadline within a day", "whether it contains personal contact information"]
REQUIRE = [("must come from outside the organization", "must come from inside the organization"),
           ("must request a payment", "must not request a payment"), ("must concern travel", "must not concern travel"),
           ("must request computer access", "must not request computer access"),
           ("must have a deadline within a day", "must not have a deadline within a day"),
           ("must contain personal contact information", "must not contain personal contact information")]
POLICIES = [(i, j, a, b) for i, j in itertools.combinations(range(6), 2) for a in (1, 0) for b in (1, 0)]  # a, b: required value
TASKS, NMAX, QUERIES, SEED = 200, 32, 4, 0
# The no-note grid is dense enough that interpolating it moves the oracle's ESS by at most 0.6 (3,000 draws).
GRID = (0, 2, 4, 6, 8, 12, 16, 20, 24, 32)
CELLS = [("none", n) for n in GRID] + [(k, n) for k in ("broad", "specific", "unreliable") for n in (0, 2, 4)]
P_STATED = 0.5


def label(pol, x):
    return int(x[pol[0]] == pol[2] and x[pol[1]] == pol[3])


def make_tasks(seed=SEED):
    """Per task: the true policy, which of its two properties the note starts from, an independent policy for the wrong
    note, whether the unreliable note is relevant, NMAX + QUERIES messages (properties and wording)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(TASKS):
        x = rng.integers(0, 2, (NMAX + QUERIES, 6))
        out.append(dict(policy=POLICIES[rng.integers(60)], first=int(rng.integers(2)), other=POLICIES[rng.integers(60)],
                        relevant=bool(rng.integers(2)), x=x.tolist(),
                        words=[[int(rng.integers(3)) for _ in range(6)] for _ in range(NMAX + QUERIES)],
                        order=[rng.permutation(6).tolist() for _ in range(NMAX + QUERIES)]))
    return out


def message(task, k):
    x, w, o = task["x"][k], task["words"][k], task["order"][k]
    return " ".join(PHRASES[p][0 if x[p] else 1][w[p]] for p in o)


def note_policy(task, kind):
    return task["policy"] if kind != "unreliable" or task["relevant"] else task["other"]


def note(task, kind):
    if kind == "none":
        return "No policy note is available."
    i, j, a, b = note_policy(task, kind)
    (k, v), other = ((i, a), j) if task["first"] == 0 else ((j, b), i)
    if kind == "broad":
        return f"Policy note: {NAMES[k]} is one of the two properties the policy uses."
    s = f"Policy note: to go into A, a message {REQUIRE[k][0 if v else 1]}. The other property the policy uses is {NAMES[other]}."
    if kind == "unreliable":
        s += (f" This note has a {round(100 * (1 - P_STATED))}% chance of being wrong, in which case it says nothing about the"
              " policy, which is then any of the 60, equally likely.")
    return s


def described(task, kind):
    """The policies a note of this kind is consistent with (the note's own set; for a wrong note, the set it describes)."""
    if kind == "none":
        return list(POLICIES)
    i, j, a, b = note_policy(task, kind)
    (k, v), other = ((i, a), j) if task["first"] == 0 else ((j, b), i)
    def ok(p):
        props = {p[0]: p[2], p[1]: p[3]}
        if kind == "broad":
            return k in props
        return props.get(k) == v and other in props
    return [p for p in POLICIES if ok(p)]


SETUP = ("A company files each incoming message into folder A or folder B by a fixed policy. Each message has six "
         "properties, each either present or absent: " + "; ".join(f"({i + 1}) {p}" for i, p in enumerate(PROPS)) + ". "
         "The policy uses exactly two of these properties and requires each to be either present or absent; a message "
         "goes into A when both requirements hold, and into B otherwise. The company's policy was chosen at random, with "
         "each of the 60 possible policies equally likely.")


def prompt(task, kind, n):
    lines = [SETUP, "", note(task, kind), ""]
    if n:
        lines += ["Previously filed messages:"] + [f"- \"{message(task, k)}\" Folder {'A' if label(task['policy'], task['x'][k]) else 'B'}."
                                                  for k in range(n)] + [""]
    lines += ["New messages:"] + [f"{q + 1}. \"{message(task, NMAX + q)}\"" for q in range(QUERIES)]
    lines += ["", f"For each new message, give the probability that it goes into folder A, as a number between 0 and 1. "
                  f"Reply with {QUERIES} lines, \"1: <probability>\" to \"{QUERIES}: <probability>\", and nothing else."]
    return "\n".join(lines)


def oracle(task, kind, n):
    """Bayes-optimal P(A) for each query: uniform over the policies consistent with the note (mixed with all 60 for the
    unreliable note, at the stated chance, updated by the demonstrations) and with the filed messages."""
    xs, ys = task["x"][:n], [label(task["policy"], x) for x in task["x"][:n]]
    fit = lambda pols: [p for p in pols if all(label(p, x) == y for x, y in zip(xs, ys))]
    qs = task["x"][NMAX:]
    share = lambda pols: np.array([np.mean([label(p, q) for p in pols]) for q in qs])
    if kind != "unreliable":
        return share(fit(described(task, kind)))
    d, a = described(task, kind), list(POLICIES)
    fd, fa = fit(d), fit(a)
    l1, l0 = P_STATED * len(fd) / len(d), (1 - P_STATED) * len(fa) / len(a)  # P(note relevant, data), P(wrong, data)
    pi = l1 / (l1 + l0)
    return pi * (share(fd) if fd else 0) + (1 - pi) * share(fa)


def truth(task):
    return np.array([label(task["policy"], q) for q in task["x"][NMAX:]])


def read(text):
    """The probability on each line "q: <p>", where <p> is a decimal, a fraction such as 13/36, or a percentage; the last
    line for a query wins, and anything outside [0, 1] is missing."""
    vals = {}
    for m in re.finditer(r"^\s*(\d)\s*[:.)]\s*(\d*\.?\d+)(?:\s*/\s*(\d*\.?\d+))?\s*(%?)", text, re.M):
        v = float(m.group(2)) / (float(m.group(3)) if m.group(3) else 1) / (100 if m.group(4) else 1)
        vals[int(m.group(1))] = v if 0 <= v <= 1 else np.nan
    return np.array([vals.get(q + 1, np.nan) for q in range(QUERIES)])


assert np.allclose(read("1: 0.25\n2: 13/36\n3: 1/3\n4: 40%"), [0.25, 13 / 36, 1 / 3, 0.4])
assert np.isnan(read("1: 2\n2: 0\n3: 1\n4: .5")[0]) and read("1: 2\n2: 0\n3: 1\n4: .5")[3] == 0.5


def prop_prompt(task, k):
    return (f"Message: \"{message(task, k)}\"\n\nFor each property below, answer 1 if the message has it and 0 if not. "
            + " ".join(f"({i + 1}) {p}." for i, p in enumerate(PROPS))
            + " Reply with six digits separated by spaces, in this order, and nothing else.")


def ess(curve_n, curve_loss, target, n):
    """Definition 1: n* is the smallest t >= 0 at which the no-note loss curve, interpolated linearly between the grid
    points, equals `target`, and the ESS is n* - n. The curve need not be monotone: a crossing on a rising stretch counts.
    If the curve stays above `target`, the ESS exceeds t_max - n (returned as +inf); if it stays below, ESS <= -n (-inf)."""
    for a in range(1, len(curve_n)):
        l0, l1 = curve_loss[a - 1], curve_loss[a]
        if min(l0, l1) <= target <= max(l0, l1):
            t0, t1 = curve_n[a - 1], curve_n[a]
            return (t0 + (target - l0) / (l1 - l0) * (t1 - t0) if l1 != l0 else t0) - n
    return float("inf") if target < min(curve_loss) else -float("inf")


def monotone(y):
    """The least-squares non-increasing fit (pool adjacent violators), the paper's monotone reading (Table 4)."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i:i + 2] = [[v, blocks[i][1] + blocks[i + 1][1]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return [v for v, w in blocks for _ in range(w)]


def report(loss):
    """loss: {cell: mean Brier}. Prints the losses and each note's ESS against the same learner's no-note curve."""
    ns = [n for k, n in CELLS if k == "none"]
    curve = [loss[("none", n)] for n in ns]
    for k, n in CELLS:
        e = "" if k == "none" else f"   ESS {ess(ns, curve, loss[(k, n)], n):6.2f}"
        print(f"  {k:10s} n={n:2d}  loss {loss[(k, n)]:.3f}{e}")


def main():
    tasks = make_tasks()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    if cmd == "prompts":
        rows = [dict(cell=c, task=t, kind=k, n=n, prompt=prompt(tasks[t], k, n)) for c, (k, n) in enumerate(CELLS) for t in range(TASKS)]
        json.dump(rows, open(HERE / "prompts.json", "w"), indent=0)
        props = [dict(task=t, k=k, x=tasks[t]["x"][k], prompt=prop_prompt(tasks[t], k)) for t in range(TASKS) for k in (0, 1, 2, NMAX)]
        json.dump(props, open(HERE / "props.json", "w"), indent=0)
        print(len(rows), "prompts;", len(props), "property prompts")
    elif cmd == "oracle":
        loss = {cell: np.mean([np.mean((oracle(t, *cell) - truth(t)) ** 2) for t in tasks]) for cell in CELLS}
        print(f"Bayes-optimal (exact), {TASKS} tasks x {QUERIES} queries:")
        report(loss)

if __name__ == "__main__":
    main()
