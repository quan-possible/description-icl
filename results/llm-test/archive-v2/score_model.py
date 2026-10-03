"""Score a language model on the tasks of llm_test.py (run on a GPU).

For every condition and task, one forward pass over the prompt gives the model's
predictive distribution over the next reading at each of the K positions: the
next-token probabilities over the 900 three-digit number tokens, renormalized.
From each distribution we keep the regret (expected log loss under the true
reading distribution, minus the oracle's), the log probability of the observed
reading, its mean and standard deviation, the probability the model put on the
number tokens before renormalizing, and the probability it put on the number
stated in the line (to see how far it copies that number). Saved as arrays
(condition, task, K).

    python3 score_model.py REPO OUT.npz [--tasks 2000] [--batch 8] [--dtype float16] [--conditions a,b]
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_test as T

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("--tasks", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--conditions", default=",".join(T.CONDITIONS))
ap.add_argument("--device", default="cuda")
ap.add_argument("--random-tiny", action="store_true", help="a tiny random GPT-2 with the repo's tokenizer, to test the pipeline")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.repo)
if args.random_tiny:
    from transformers import GPT2Config, GPT2LMHeadModel
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_layer=1, n_head=2, n_embd=32, n_positions=1100)).to(args.device).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=getattr(torch, args.dtype), device_map=args.device).eval()
first = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

number_ids = [tok.encode(str(v), add_special_tokens=False) for v in T.GRID]
assert all(len(i) == 1 for i in number_ids), "every three-digit integer must be one token"
number_ids = torch.tensor([i[0] for i in number_ids], device=args.device)
grid = torch.tensor(T.GRID, device=args.device, dtype=torch.float64)

tasks = T.make_tasks(args.tasks, args.seed)
w = torch.tensor(tasks["w"], device=args.device, dtype=torch.float64)
y = torch.tensor(tasks["y"], device=args.device)
edges = torch.cat([grid - 0.5, grid[-1:] + 0.5])
p_true = torch.special.ndtr((edges[None, :] - w[:, None]) / T.SIG).diff(dim=-1)
p_true = p_true / p_true.sum(-1, keepdim=True)  # the reading's true distribution, (tasks, 900)
log_p_true = torch.log(p_true.clamp_min(1e-300))

conditions = args.conditions.split(",")
per_line = len(tok.encode("547\n", add_special_tokens=False))
out = {k: np.zeros((len(conditions), args.tasks, T.K), np.float32) for k in ("regret", "logp", "mean", "sd", "mass", "pstated")}
start = time.time()
with torch.inference_mode():
    for ci, cond in enumerate(conditions):
        for a in range(0, args.tasks, args.batch):
            idx = range(a, min(a + args.batch, args.tasks))
            ids = tok([T.prompt(tasks, cond, i) for i in idx], add_special_tokens=False)["input_ids"]
            assert len({len(i) for i in ids}) == 1, "prompts of one condition must tokenize to one length"
            ids = torch.tensor([[first] + i for i in ids], device=args.device)
            pos = ids.shape[1] - per_line * T.K + per_line * torch.arange(T.K, device=args.device) + per_line - 2  # the number tokens
            assert (ids[:, pos] == number_ids[y[a : a + len(idx)] - T.LO]).all(), "number tokens not where expected"
            logp_vocab = torch.log_softmax(model(ids).logits[:, pos - 1].double(), -1)  # position pos - 1 predicts the number at pos
            lq = logp_vocab[..., number_ids]
            mass = lq.exp().sum(-1)
            lq = lq - mass.log()[..., None]
            q = lq.exp()
            b = slice(a, a + len(idx))
            mean = (q * grid).sum(-1)
            out["regret"][ci, b] = (p_true[b, None, :] * (log_p_true[b, None, :] - lq)).sum(-1).cpu()
            out["logp"][ci, b] = lq.gather(-1, (y[b] - T.LO)[..., None])[..., 0].cpu()
            out["mean"][ci, b] = mean.cpu()
            out["sd"][ci, b] = ((q * grid**2).sum(-1) - mean**2).clamp_min(0).sqrt().cpu()
            out["mass"][ci, b] = mass.cpu()
            if T.CONDITIONS[cond]:  # the probability on exactly the stated value
                stated = torch.tensor(tasks["m"][T.CONDITIONS[cond][0]][a : a + len(idx)], device=args.device)
                out["pstated"][ci, b] = q.gather(-1, (stated - T.LO)[:, None, None].expand(-1, T.K, 1))[..., 0].cpu()
        print(f"{cond}: regret at n = 0, 5, 50, 199: {out['regret'][ci].mean(0)[[0, 5, 50, 199]].round(3)}; "
              f"mass on number tokens {out['mass'][ci].mean():.4f}; {time.time() - start:.0f} s", flush=True)
        np.savez_compressed(args.out, conditions=np.array(conditions), repo=args.repo, tasks=args.tasks, seed=args.seed, dtype=args.dtype, **out)
print("done", flush=True)
