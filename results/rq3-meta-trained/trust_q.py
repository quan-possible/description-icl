"""trust_q.csv: each network's implied trust with no demonstrations, in closed form.

With no demonstrations the Bayes-optimal prediction under trust q is q m^T x (the relevant reading predicts m^T x,
the irrelevant one 0), so the q whose predictions are closest to the network's in squared error is the regression
coefficient of the network's prediction on m^T x, with no grid. Its standard error is over prompts.

    uv run python results/rq3-meta-trained/trust_q.py CKPT [CKPT ...]
"""

import csv
import pathlib
import sys

import numpy as np
import torch

from description_icl import meta

here = pathlib.Path(__file__).parent
S = 20_000

rows = []
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu")
    cfg = ck["args"]
    model = meta.Transformer(cfg["d"])
    model.load_state_dict(ck["model"])
    model.eval()
    gen = torch.Generator().manual_seed(10)  # evaluate.py's prompts with a description
    batch = meta.sample_batch(S, cfg["K"], cfg["d"], cfg["a_base"], torch.tensor([cfg["r"]]), cfg["p"], 1.0, "cpu", gen)
    count = torch.zeros(S, dtype=torch.long)
    with torch.no_grad():
        yhat = model(*meta.tokens(batch, count), count)[0].double().numpy() * cfg["a_base"] ** 0.5  # in the units of m and x, as in evaluate.py
    z = torch.einsum("bd,bd->b", batch["m"], batch["X"][:, 0]).double().numpy()  # m^T x for the query (n = 0)
    q = (yhat * z).sum() / (z * z).sum()
    se = np.sqrt(((yhat - q * z) ** 2).sum() / (S - 1) / (z * z).sum())
    rows.append(dict(model=pathlib.Path(path).stem, q=round(float(q), 4), se=round(float(se), 4)))
    print(rows[-1], flush=True)

out = here / "trust_q.csv"
old = {r["model"]: r for r in csv.DictReader(out.open())} if out.exists() else {}
old.update({r["model"]: r for r in rows})
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "q", "se"])
    w.writeheader()
    w.writerows(sorted(old.values(), key=lambda r: r["model"]))
