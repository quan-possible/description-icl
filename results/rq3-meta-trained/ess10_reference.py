"""Bayes-optimal ESS of the r = 0.01, p = 0.9 description (d = 5, a0 = 10)
with n = 0, 5, 10 demonstrations in hand, from expected regrets (mx.regret)
on 8 batches of 50,000 prompts with a demonstrations-only curve to n = 30.

The paper quotes n = 5: 8.58 (batch se 0.03), against 8.69 read off the
network's 20,000 evaluation prompts in ess_n.csv. At n = 10 that reading
(9.04) is about one demonstration low (10.05 here), because the
demonstrations-only curve is nearly flat near n = 20; the paper therefore
reports n = 5. n = 0 gives 6.90.

    uv run python results/rq3-meta-trained/ess10_reference.py   # ~10 s
"""

import numpy as np

from description_icl import gaussian as g, mixture as mx

D, A0, K = 5, 10.0, 31


def curves(seed, S=50000):
    P = mx.simulate(D, A0, 0.01 * A0, 0.9, K, S, seed)
    desc = mx.regret(P, 0.9, 1)
    P0 = mx.simulate(D, A0, 0.01 * A0, 0.0, K, S, seed + 1000)  # tasks from the base prior
    plain = mx.regret(P0, 0.0, 1)
    return plain, desc


res = [curves(s) for s in range(8)]
plain = np.mean([r[0] for r in res], 0)
desc = np.mean([r[1] for r in res], 0)
for n in (0, 5, 10):
    per = [g.first_crossing(r[0], r[1][n])[1] - n for r in res]
    se = np.std(per, ddof=1) / np.sqrt(len(per))
    print(f"n={n}: ESS {g.first_crossing(plain, desc[n])[1] - n:.2f} (batch se {se:.2f})")
