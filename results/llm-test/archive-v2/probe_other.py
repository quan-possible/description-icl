"""Where the probability goes when a model does not expect a number next: for the first numbers of a
prompt, the most probable tokens that are not three-digit numbers, averaged over tasks.

    python3 probe_other.py REPO OUT.json [--tasks 64]
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_test as T

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("out")
ap.add_argument("--tasks", type=int, default=64)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.repo)
model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=torch.float16, device_map="cuda").eval()
first = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
numbers = torch.tensor([tok.encode(str(v), add_special_tokens=False)[0] for v in T.GRID], device="cuda")
tasks = T.make_tasks(2000, 0)  # the tasks of the main run; the first --tasks of them are used
out = {}
with torch.inference_mode():
    for cond in ("none", "valid10", "invalid10", "valid10_sentence", "valid10_interval"):
        ids = tok([T.prompt(tasks, cond, i) for i in range(args.tasks)], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([[first] + i for i in ids], device="cuda")
        pos = ids.shape[1] - 2 * T.K + 2 * torch.arange(13, device="cuda")  # the first 13 numbers
        p = torch.softmax(model(ids[:, : int(pos[-1]) + 1]).logits[:, pos - 1].double(), -1).mean(0)  # (13, vocab)
        p[:, numbers] = 0
        top = p.topk(6, -1)
        out[cond] = [{"n": n, "off_numbers": round(float(p[n].sum()), 3), "top": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values[n], top.indices[n])]} for n in range(13)]
        print(cond, [(o["n"], o["off_numbers"], o["top"][:3]) for o in out[cond] if o["n"] in (0, 1, 5, 10)], flush=True)
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
