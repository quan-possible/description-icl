"""Compare a meta-trained model with the exact Bayes-optimal learner.

    uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/<name>.pt

Writes, next to this script and named after the checkpoint:

  <name>_regret.csv   single-query regret after n examples, network and
                      Bayes, with and without a description
  <name>_ess.csv      ESS of the description for the network and for Bayes
  <name>_trust.csv    the trust q at which the Bayes learner's prediction
                      is closest (in KL) to the network's

The evaluation data use the checkpoint's training reliability unless
--p-test is given. Bayes here always means the learner with trust q equal
to the training reliability, which is optimal for the training distribution.
"""

import argparse
import csv
import pathlib

import numpy as np
import torch

from descriptor_icl import gaussian as g
from descriptor_icl import meta
from descriptor_icl import mixture as mx

QS = np.round(np.concatenate([[0.001, 0.01], np.arange(0.05, 1.0, 0.05), [0.99, 0.999]]), 3)
TRUST_STEPS = (0, 2, 8)

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--p-test", type=float)
ap.add_argument("--S", type=int, default=20_000)
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
cfg = ck["args"]
d, a0, K, p_train = cfg["d"], cfg["a_base"], cfg["K"], cfg["p"]
RS = (cfg["r"],)  # each model is trained at one precision
p_test = p_train if args.p_test is None else args.p_test
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = (meta.Transformer if cfg["arch"] == "transformer" else meta.LSTM)(d)
model.load_state_dict(ck["model"])
model.to(dev).eval()
name = pathlib.Path(args.ckpt).stem + ("" if args.p_test is None else f"_ptest{p_test}")
here = pathlib.Path(__file__).parent


def predict(batch):
    out = []
    with torch.no_grad():
        for i in range(0, args.S, 2000):
            part = {k: v[i : i + 2000].to(dev) for k, v in batch.items()}
            out.append([t.cpu() for t in model(meta.tokens(part))])
    return [torch.cat(t) for t in zip(*out)]


def draw(r, p_desc, seed):
    gen = torch.Generator().manual_seed(seed)
    return meta.sample_batch(args.S, K, d, a0, torch.tensor([r]), p_test, p_desc, "cpu", gen)


def oracle_logp(batch):
    err = batch["Y"] - torch.einsum("bkd,bd->bk", batch["X"], batch["w"])
    return (-0.5 * err**2 - 0.5 * np.log(2 * np.pi)).numpy()


def bayes_paths(batch, r):
    n = lambda k: batch[k].double().numpy()
    return mx.paths_from_data(n("X"), n("Y"), n("m"), batch["correct"].numpy(), a0, r * a0)


def write(suffix, rows):
    with (here / f"{name}_{suffix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_se(x):
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x))


def kl_bayes_to_net(paths, pred, k, q):
    """Mean KL(Bayes_q || network) for the prediction of y_{k+1}, by quadrature."""
    l1 = np.log(q) + paths.L1[:, k]
    l0 = np.log1p(-q) + paths.L0[:, k]
    w1 = np.exp(l1 - np.logaddexp(l1, l0))
    mu = np.stack([paths.mu1[:, k], paths.mu0[:, k]], 1)
    sd = np.sqrt(np.exp(np.stack([paths.inc1[:, k], paths.inc0[:, k]], 1)))
    lo, hi = (mu - 7 * sd).min(1), (mu + 7 * sd).max(1)
    grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, 801)[None, :]
    dens = lambda j: np.exp(-0.5 * ((grid - mu[:, j, None]) / sd[:, j, None]) ** 2) / (
        sd[:, j, None] * np.sqrt(2 * np.pi)
    )
    pb = w1[:, None] * dens(0) + (1 - w1)[:, None] * dens(1)
    step = [t[:, k : k + 1] for t in pred]
    ln = meta.log_density(step, torch.from_numpy(grid).float()[:, None, :])[:, 0].numpy()
    integrand = pb * (np.log(pb + 1e-300) - ln)
    return (integrand.sum(1) * (hi - lo) / 800).mean()


regret_rows, ess_rows, trust_rows = [], [], []

# prompts without a description: the network's own examples-only curve
plain = draw(RS[0], 0.0, seed=1)
pred = predict(plain)
net_lp = meta.log_density(pred, plain["Y"]).numpy()
paths = bayes_paths(plain, RS[0])
bayes_lp = np.diff(paths.L0, axis=1)
orc = oracle_logp(plain)
net_plain, net_plain_se = mean_se(orc - net_lp)
bayes_plain, _ = mean_se(orc - bayes_lp)
for n in range(K):
    regret_rows.append(
        dict(desc=0, r="", n=n, net=round(net_plain[n], 4), net_se=round(net_plain_se[n], 4),
             bayes=round(bayes_plain[n], 4))
    )

for i, r in enumerate(RS):
    batch = draw(r, 1.0, seed=10 + i)
    pred = predict(batch)
    net_lp = meta.log_density(pred, batch["Y"]).numpy()
    paths = bayes_paths(batch, r)
    l1 = np.log(p_train) + paths.L1 if p_train > 0 else np.full_like(paths.L1, -np.inf)
    l0 = np.log1p(-p_train) + paths.L0 if p_train < 1 else np.full_like(paths.L0, -np.inf)
    bayes_lp = np.diff(np.logaddexp(l1, l0), axis=1)
    orc = oracle_logp(batch)
    net, net_se = mean_se(orc - net_lp)
    bayes, bayes_se = mean_se(orc - bayes_lp)
    for n in range(K):
        regret_rows.append(
            dict(desc=1, r=r, n=n, net=round(net[n], 4), net_se=round(net_se[n], 4),
                 bayes=round(bayes[n], 4))
        )
    for N in (1, 8):
        # description alone over horizon N, against examples-only curves
        window = lambda c: np.convolve(c, np.ones(N), "valid")
        ess_rows.append(
            dict(r=r, N=N,
                 net_ess=round(g.first_crossing(window(net_plain), net[:N].sum())[1], 3),
                 bayes_ess=round(g.first_crossing(window(bayes_plain), bayes[:N].sum())[1], 3))
        )
    for k in TRUST_STEPS:
        kls = [kl_bayes_to_net(paths, pred, k, q) for q in QS]
        trust_rows.append(
            dict(r=r, n=k, implied_q=QS[int(np.argmin(kls))], kl_at_implied=round(min(kls), 4),
                 kl_at_train_p=round(kl_bayes_to_net(paths, pred, k, min(max(p_train, 1e-3), 1 - 1e-3)), 4))
        )

write("regret", regret_rows)
write("ess", ess_rows)
write("trust", trust_rows)
for rows in (ess_rows, trust_rows):
    print(*rows, sep="\n")
