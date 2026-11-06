"""Compare a meta-trained model with the exact Bayes-optimal learner.

    uv run python results/rq3-meta-trained/evaluate.py tmp/rq3/<name>.pt

Regret is the squared error of a point prediction against the noiseless
label w^T x, averaged over prompts (Garg et al.); the network's point
prediction is the mean of its predictive distribution.
Writes, next to this script and named after the checkpoint:

  <name>_regret.csv   regret after n examples, network and Bayes, with and
                      without a description
  <name>_ess.csv      ESS of the description for the network and for Bayes
  <name>_trust.csv    the trust q at which the Bayes learner's prediction
                      is closest to the network's

The evaluation data use the checkpoint's training reliability unless
--p-test is given. Bayes here always means the learner with trust q equal
to the training reliability, which is optimal for the training distribution.
"""

import argparse
import csv
import pathlib

import numpy as np
import torch

from description_icl import gaussian as g
from description_icl import meta
from description_icl import mixture as mx

QS = np.round(np.concatenate([[0.001, 0.01], np.arange(0.05, 1.0, 0.05), [0.99, 0.999]]), 3)
TRUST_STEPS = (0, 2, 8)


def draw(S, K, d, a0, r, p, p_desc, seed, k=None):
    gen = torch.Generator().manual_seed(seed)
    return meta.sample_batch(S, K, d, a0, torch.tensor([r]), p, p_desc, "cpu", gen, k=k)


def bayes_paths(batch, a0, r, k=None):
    """A partial description (k of d directions) has prior variance r a0 in the described directions, a0 in the rest."""
    n = lambda key: batch[key].double().numpy()
    d = batch["X"].shape[2]
    a_desc = r * a0 if k is None else np.where(np.arange(d) < k, r * a0, a0)
    return mx.paths_from_data(n("X"), n("Y"), n("m"), batch["correct"].numpy(), a0, a_desc)


def regret(mean, batch):
    """Squared error of the predictions against the noiseless label w^T x, in
    the exact learner's units: (mean, standard error) over prompts, for each
    number of examples."""
    oracle = torch.einsum("bkd,bd->bk", batch["X"], batch["w"]).double().numpy()
    x = (mean - oracle) ** 2
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x))


def bayes_regret(paths, q, batch):
    """Squared error of the exact learner with trust q, (mean, standard error) per n."""
    return regret(mx.mean_prediction(paths, q), batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--p-test", type=float)
    ap.add_argument("--S", type=int, default=20_000)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["args"]
    d, a0, K, p_train, r, kd = cfg["d"], cfg["a_base"], cfg["K"], cfg["p"], cfg["r"], cfg.get("k")
    p_test = p_train if args.p_test is None else args.p_test
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = meta.Transformer(d)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    name = pathlib.Path(args.ckpt).stem + ("" if args.p_test is None else f"_ptest{p_test}")
    here = pathlib.Path(__file__).parent

    def predict(batch):
        """The network's predictive for y_{n+1} after n examples: mean and log
        variance, each (S, K), converted to the exact learner's units."""
        means, lvs = [], []
        with torch.no_grad():
            for i in range(0, args.S, 2000):
                part = {k: v[i : i + 2000].to(dev) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
                per_n = []
                for n in range(K):
                    count = torch.full((len(part["X"]),), n, device=dev)
                    per_n.append(torch.stack(model(*meta.tokens(part, count), count), 1))
                out = torch.stack(per_n, 1).cpu()  # (B, K, 2)
                means.append(out[..., 0]); lvs.append(out[..., 1])
        mean = torch.cat(means).double().numpy() * a0**0.5
        log_var = torch.cat(lvs).double().numpy() + np.log(a0)
        return mean, log_var

    def write(suffix, rows):
        with (here / f"{name}_{suffix}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # prompts without a description: the examples-only curves
    plain = draw(args.S, K, d, a0, r, p_test, 0.0, seed=1, k=kd)
    net_plain, net_plain_se = regret(predict(plain)[0], plain)
    plain["correct"] = torch.zeros_like(plain["correct"])  # w came from the base prior
    bayes_plain, _ = bayes_regret(bayes_paths(plain, a0, r, kd), 0.0, plain)

    batch = draw(args.S, K, d, a0, r, p_test, 1.0, seed=10, k=kd)
    pred, paths = predict(batch)[0], bayes_paths(batch, a0, r, kd)
    net, net_se = regret(pred, batch)
    bayes, _ = bayes_regret(paths, p_train, batch)

    rows = [dict(desc=0, n=n, net=round(net_plain[n], 4), net_se=round(net_plain_se[n], 4),
                 bayes=round(bayes_plain[n], 4)) for n in range(K)]
    rows += [dict(desc=1, n=n, net=round(net[n], 4), net_se=round(net_se[n], 4),
                  bayes=round(bayes[n], 4)) for n in range(K)]
    write("regret", rows)

    # the description alone against the examples-only curves
    ess = [dict(r=r, p_train=p_train, p_test=p_test,
                net_ess=round(g.first_crossing(net_plain, net[0])[1], 3),
                bayes_ess=round(g.first_crossing(bayes_plain, bayes[0])[1], 3))]
    write("ess", ess)

    trust = []
    for k in TRUST_STEPS:
        gap = [((pred[:, k] - mx.mean_prediction(paths, q)[:, k]) ** 2).mean() for q in QS]
        trust.append(dict(n=k, implied_q=QS[int(np.argmin(gap))], gap_at_implied=round(min(gap), 4)))
    write("trust", trust)
    print(*ess, *trust, sep="\n")
