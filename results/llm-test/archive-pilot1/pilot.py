"""Pilot: does an instruction make a language model use the description?

The setting of llm_test.py (the first S tasks of the main run, N numbers each) under
several prompt heads, from the bare line to an explicit instruction. Each head comes
with and without the description; the description is valid or invalid at three
tolerances. With --chat the head is the user turn of the model's chat template and
the numbers are the assistant's turn. The "stated" head (chat only) instead puts the
numbers so far in the user's turn and asks for the next one, at the n in USED.

Every three-digit number must be one token for the model, so that one pass gives its
distribution before every number.

A model uses the description when, before any number, its prediction follows the
stated value (slope near 1) with a spread that narrows with the tolerance, and when
the implied strength over n = 5..10 rises with the stated precision (ideal 1, 9, 100).

    python3 pilot.py                              # print one prompt per head
    python3 pilot.py score REPO OUT.npz [--chat] [--tasks S] [--heads a,b]  # on a GPU
    python3 pilot.py read OUT.npz [...]           # locally
"""

import sys

import numpy as np

import llm_test as T

S, N = 500, 20
NORMAL = "Samples from a normal distribution with standard deviation 30"
# name: (head without a description, head with one). {m}: stated value, {tol}: tolerance, {c}: (30 / tol)^2,
# {lo} and {hi}: two tolerances either side of the stated value
HEADS = {
    "bare": ("", "Mean: {m} ± {tol}\n"),
    "below": ("", "The numbers below have a mean of {m} ± {tol}.\n"),
    "noisy": ("Noisy measurements of one quantity:\n", "Noisy measurements of one quantity, which is {m} ± {tol}:\n"),
    "normal": (NORMAL + ":\n", "Samples from a normal distribution with mean {m} ± {tol} and standard deviation 30:\n"),
    "stderr": (NORMAL + ":\n", NORMAL + ". Its mean was estimated beforehand as {m}, with a standard error of {tol}:\n"),
    "told": (NORMAL + ". Predict each next sample:\n", NORMAL + ". Its mean is {m} ± {tol}. Use this together with the samples so far to predict each next sample:\n"),
    "range": ("", "The mean is between {lo} and {hi}.\n"),
    "range_normal": (NORMAL + ":\n", NORMAL + ". Its mean is between {lo} and {hi}:\n"),
    "earlier": (NORMAL + ":\n", NORMAL + ". {c} earlier sample{s} had a mean of {m}:\n"),  # the description as data
}
USED = (0, 5, 6, 7, 8, 9, 10)  # numbers in hand at which a question head asks for the next one
# heads that put the numbers so far in the user's turn and end with a question ({next}: "next", or "first" before any number)
QUESTIONS = {"stated": "Predict the {next} number."}
DRAWN = "The numbers below are drawn from a normal distribution with standard deviation 30.\n"
HEADS["stated"] = (DRAWN + "Its mean is unknown.\n", DRAWN + "Its mean was drawn from a normal distribution with mean {m} and standard deviation {tol}.\n")
LINES = ("none", "valid30", "valid10", "valid3", "invalid10")
NAMES = [f"{h}/{line}" for h in HEADS for line in LINES]


def head(tasks, name, i):
    h, line = name.split("/")
    if line == "none":
        return HEADS[h][0]
    tol = int(line.lstrip("invalid"))
    c = round(T.SIG**2 / tol**2)
    m = tasks["m"][line][i]
    return HEADS[h][1].format(m=m, tol=tol, c=c, s="" if c == 1 else "s", lo=m - 2 * tol, hi=m + 2 * tol)


def numbers(tasks, i, n=N):
    return "".join(f"{v}\n" for v in tasks["y"][i, :n])


def ask(tasks, name, i, n):
    """The user's turn of a question head with n numbers in hand."""
    return head(tasks, name, i) + "\n" + numbers(tasks, i, n) + "\n" + QUESTIONS[name.split("/")[0]].format(next="next" if n else "first") + " Reply with the number only."


def score(repo, out, chat, S, heads):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="cuda").eval()
    first = next(i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) if i is not None)
    number_ids = [tok.encode(str(v), add_special_tokens=False) for v in T.GRID]
    assert all(len(i) == 1 for i in number_ids) and len(tok.encode("547\n", add_special_tokens=False)) == 2, "every three-digit integer must be one token"
    number_ids = torch.tensor([i[0] for i in number_ids], device="cuda")
    grid = torch.tensor(T.GRID, device="cuda", dtype=torch.float64)
    tasks = T.make_tasks(2000, 0)
    y = torch.tensor(tasks["y"][:S, :N], device="cuda")
    edges = torch.cat([grid - 0.5, grid[-1:] + 0.5])
    p_true = torch.special.ndtr((edges[None, :] - torch.tensor(tasks["w"][:S], device="cuda")[:, None]) / T.SIG).diff(dim=-1)
    p_true = p_true / p_true.sum(-1, keepdim=True)
    log_p_true = torch.log(p_true.clamp_min(1e-300))

    def turn(content):  # the template's text carries the model's first token; no thinking before the answer
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text + "<|channel|>final<|message|>" if text.endswith("<|start|>assistant") else text  # gpt-oss: answer in the final channel

    def encode(name, i):
        if chat:
            return tok.encode(turn(head(tasks, name, i).rstrip("\n")) + numbers(tasks, i), add_special_tokens=False)
        return [first] + tok.encode(head(tasks, name, i) + numbers(tasks, i), add_special_tokens=False)

    def logits(ids):
        """The model's logits on a batch of prompts padded on the left, and the padded length."""
        L = max(map(len, ids))
        x = torch.tensor([[first] * (L - len(i)) + i for i in ids], device="cuda")
        mask = torch.tensor([[0] * (L - len(i)) + [1] * len(i) for i in ids], device="cuda")
        return x, model(x, attention_mask=mask).logits

    names = [n for n in NAMES if (not heads or n.split("/")[0] in heads) and (chat or n.split("/")[0] not in QUESTIONS)]
    res = {k: np.full((len(names), S, N), np.nan, np.float32) for k in ("regret", "mean", "sd", "mass")}
    with torch.inference_mode():
        for ci, name in enumerate(names):
            for a in range(0, S, 50):
                b = slice(a, min(a + 50, S))
                if name.split("/")[0] in QUESTIONS:  # one prompt per number of numbers in hand; the reply's first token is the prediction
                    ns = list(USED)
                    lq = torch.stack([logits([tok.encode(turn(ask(tasks, name, i, n)), add_special_tokens=False) for i in range(b.start, b.stop)])[1][:, -1] for n in ns], 1)
                else:  # one prompt; the prediction before each number
                    ns = list(range(N))
                    x, lq = logits([encode(name, i) for i in range(b.start, b.stop)])
                    pos = x.shape[1] - 2 * N + 2 * torch.arange(N, device="cuda")  # the number tokens
                    assert (x[:, pos] == number_ids[y[b] - T.LO]).all(), "number tokens not where expected"
                    lq = lq[:, pos - 1]
                lq = torch.log_softmax(lq.double(), -1)[..., number_ids]
                mass = lq.exp().sum(-1)
                lq = lq - mass.log()[..., None]
                q = lq.exp()
                mean = (q * grid).sum(-1)
                res["regret"][ci][b][:, ns] = (p_true[b, None, :] * (log_p_true[b, None, :] - lq)).sum(-1).cpu()
                res["mean"][ci][b][:, ns] = mean.cpu()
                res["sd"][ci][b][:, ns] = ((q * grid**2).sum(-1) - mean**2).clamp_min(0).sqrt().cpu()
                res["mass"][ci][b][:, ns] = mass.cpu()
            print(f"{name}: mass on numbers at n = 0, 5: {res['mass'][ci].mean(0)[[0, 5]].round(2)}", flush=True)
    np.savez_compressed(out, names=np.array(names), repo=repo, chat=chat, **res)
    print("done", flush=True)


def read(paths):
    tasks = T.make_tasks(2000, 0)
    print("ideal: slope 1; spread 42, 32, 30; strength 1, 9, 100; strength of an invalid line below 0.1")
    for path in paths:
        d = np.load(path)
        names = [str(n) for n in d["names"]]
        S, n = d["mean"].shape[1:]
        y = tasks["y"][:S, :n]
        print(f"\n{d['repo']}{' (chat template)' if d['chat'] else ''}")
        print(f"{'head':12s} | before any number: slope  spread ±30 ±10 ±3  P(number) | strength ±30   ±10    ±3 | invalid ±10 | regret n=5..10: no line, ±10 | P(number) n=5 | uses it")
        for h in HEADS:
            if f"{h}/none" not in names:
                continue
            ix = lambda line: names.index(f"{h}/{line}")
            m = {line: tasks["m"][line][:S].astype(float) for line in LINES[1:]}
            slope = np.polyfit(m["valid10"], d["mean"][ix("valid10"), :, 0], 1)[0]
            spread = [d["sd"][ix(f"valid{t}"), :, 0].mean() for t in T.TOLS]
            c = [T.implied_c(d["mean"][ix(f"valid{t}")], m[f"valid{t}"], y) for t in T.TOLS]
            bad = T.implied_c(d["mean"][ix("invalid10")], m["invalid10"], y)
            regret = [d["regret"][ix(line)][:, 5:11].mean() for line in ("none", "valid10")]
            uses = slope >= 0.8 and spread[0] > spread[1] > spread[2] and c[1] >= 3 * c[0] and c[2] > c[1]
            print(f"{h:12s} | {slope:24.2f}  {spread[0]:6.0f} {spread[1]:3.0f} {spread[2]:3.0f}  {d['mass'][ix('valid10'), :, 0].mean():8.2f}  | {c[0]:12.1f} {c[1]:5.1f} {c[2]:5.1f} | {bad:11.1f} | {regret[0]:18.2f} {regret[1]:8.2f} | {d['mass'][ix('valid10'), :, 5].mean():13.2f} | {'YES' if uses else 'no'}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        opt = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
        score(sys.argv[2], sys.argv[3], "--chat" in sys.argv, int(opt("--tasks", S)), opt("--heads", "").split(",") if "--heads" in sys.argv else [])
    elif sys.argv[1:2] == ["read"]:
        read(sys.argv[2:])
    else:
        tasks = T.make_tasks(2000, 0)
        for h in HEADS:
            print(f"--- {h}\n{head(tasks, f'{h}/valid10', 0)}" + "".join(f"{v}\n" for v in tasks["y"][0, :3]) + "...")
        assert head(tasks, "earlier/valid30", 0).count("1 earlier sample had") == 1 and head(tasks, "bare/none", 0) == ""
