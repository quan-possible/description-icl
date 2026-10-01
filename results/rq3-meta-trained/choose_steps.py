"""Pick the grid's step count from the pilot's snapshots: the first snapshot
whose mean gap to the exact learner (net minus Bayes regret, averaged over
n and over prompts with and without a description) is within 0.01 nats of
the best snapshot's. Prints the step count; writes pilot.csv next to this
script with the gap at every snapshot.

    python choose_steps.py pilot_step5000_regret.csv pilot_step10000_regret.csv ...
"""

import csv
import pathlib
import re
import sys

rows = []
for f in sys.argv[1:]:
    step = int(re.search(r"step(\d+)_regret", f).group(1))
    r = list(csv.DictReader(open(f)))
    gap = sum(float(x["net"]) - float(x["bayes"]) for x in r) / len(r)
    rows.append((step, gap))
rows.sort()
best = min(g for _, g in rows)
chosen = next(s for s, g in rows if g <= best + 0.01)
with (pathlib.Path(__file__).parent / "pilot.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "mean_gap_nats"]); w.writerows(rows)
print(chosen)
