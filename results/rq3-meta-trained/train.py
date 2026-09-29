"""Meta-train one sequence model on prompts with and without descriptions.

    uv run python results/rq3-meta-trained/train.py --p 0.7 --r 0.05

Checkpoints go to tmp/rq3/ (ignored by Git); the training log is printed.
"""

import argparse
import pathlib
import time

import torch

from descriptor_icl import meta


ap = argparse.ArgumentParser()
ap.add_argument("--p", type=float, required=True)
ap.add_argument("--r", type=float, required=True)
ap.add_argument("--arch", default="transformer")
ap.add_argument("--d", type=int, default=8)
ap.add_argument("--a-base", type=float, default=10.0)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--steps", type=int, default=100_000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = (meta.Transformer if args.arch == "transformer" else meta.LSTM)(args.d).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
rs = torch.tensor([args.r], device=dev)
out = pathlib.Path("tmp/rq3")
out.mkdir(parents=True, exist_ok=True)
name = f"{args.arch}_p{args.p}_r{args.r}_d{args.d}_s{args.seed}"

t0, run = time.time(), 0.0
for step in range(1, args.steps + 1):
    batch = meta.sample_batch(args.batch, args.K, args.d, args.a_base, rs, args.p, 0.5, dev)
    loss = -meta.log_density(model(meta.tokens(batch)), batch["Y"]).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    run = loss.item() if step == 1 else 0.99 * run + 0.01 * loss.item()
    if step % 1000 == 0:
        print(f"{name} step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    if step % 10_000 == 0 or step == args.steps:
        torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / f"{name}.pt")
