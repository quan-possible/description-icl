# Misspecified reliability

**Status:** derived and validated. Not in the current paper. Weakens the
proposal's explanation for why base models under-weight instructions.

**Result.** A predictor that assumes reliability $q$ when the truth is $p$
pays, over a long horizon, $\mathrm{KL}(p\,\|\,q)$ nats more than the
calibrated predictor, independently of $d$: the identification term tends to
the cross-entropy $H(p, q)$. This is the redundancy of coding under a wrong
prior (Cover & Thomas). Check: $p = 0.7$, $q = 0.999$, $d = 16$: excess 1.46
nats, as predicted.

**One-step numbers** ($d = 16$, $a_0 = 10$, $r = 0.001$, $p = 0.7$; from
`results/rq2-reliability/trust.csv`):

| $q$ | 0 | 0.1 | 0.5 | 0.7 | 0.9 | 0.99 | 0.999 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regret | 2.51 | 1.95 | 1.33 | 1.27 | 1.39 | 1.90 | 2.44 | 40.5 |

**Reading.** Trust is cheap to mis-set unless it is complete. The large cost
at $q = 1$ is a property of log loss: the likelihood under the wrong
component is unbounded. The campaign record's earlier "40 nats" headline
came from $q = 1$ only.

**Relevance.** The network readout "effective reliability" (the $q$ at which
the Bayes predictor is closest to the network) uses this; code in
`results/rq3-meta-trained/evaluate.py`.
