"""The language-model test of the paper's Section 7: tasks, prompts, the ideal learner, scoring, and a read-out.

A task is a hidden mean w ~ N(500, 100^2) and numbers w + noise. The user's turn says
what the numbers are, what is known about the mean, shows n numbers and asks for the
next one; the first token of the reply is the prediction (chat models whose tokenizer
has every three-digit integer as one token).

A description states a value and a standard deviation. Its worth in numbers is
c = (noise / standard deviation)^2, and the Bayes-optimal weight on the stated value
with n numbers shown is c / (c + n). The conditions put c at 1, 4 and 36 and n at 0 to 64,
so that weight runs from 0.02 to 0.97. A description is relevant, irrelevant (the value of
another task), or comes with a stated chance of being wrong. A second noise level with
the same three c checks that the model responds to the ratio. The Bayes-optimal yardstick
knows only what the prompt says; "unknown" is uniform over the three-digit range.

    python3 llm_test.py                       # print example prompts and check the ideal learner
    python3 llm_test.py score REPO OUT.npz [--tasks N] [--batch B]   # on a GPU (N tasks, 500 by default; B prompts per pass, 100)
    python3 llm_test.py score REPO OUT.npz --written [...]   # read the number the model writes (greedy) instead of its distribution; any tokenizer
    python3 llm_test.py score REPO OUT.npz --paraphrase K --ablation [...]   # paraphrase K of the description sentence, on the ablation's conditions and n only
    python3 llm_test.py reason REPO OUT.npz [--tasks N]  # the model reasons first; its final number is read (n = 1 and 4, noise 60)
    python3 llm_test.py read OUT.npz [...]    # locally
"""

import sys

import numpy as np

MEAN, S0, LO, HI = 500.0, 100.0, 100, 999
GRID = np.arange(LO, HI + 1)
S = 500  # tasks; `score --tasks N` and `read` set it
NS = (0, 1, 2, 4, 8, 16, 32, 64)  # numbers shown
REGIMES = {60: (60, 30, 10), 30: (30, 15, 5)}  # noise: the description's standard deviations (c = 1, 4, 36)
MAIN = 60  # the noise level that also gets invalid descriptions and stated reliability
WRONG = "There is a {q}% chance that this statement is wrong, in which case the mean is unknown."
# the description sentence and its paraphrases (the ablation of Appendix B); PARAPHRASE picks one
PARAPHRASES = (
    "Its mean was drawn from a normal distribution with mean {m} and standard deviation {tol}.",
    "Its mean is about {m}, give or take {tol}.",
    "Mean: {m} ± {tol}",
    "An earlier estimate put its mean at {m}, with a standard error of {tol}.",
)
PARAPHRASE = 0
ABLATION = [(MAIN, None, True, None)] + [(MAIN, tol, True, None) for tol in REGIMES[MAIN]]  # the ablation's conditions
ABLATION_NS = (1, 2, 4)

# a condition: (noise, the description's standard deviation or None, valid, stated chance of being right or None)
CONDITIONS = []
for noise, tols in REGIMES.items():
    CONDITIONS.append((noise, None, True, None))
    CONDITIONS += [(noise, tol, valid, None) for tol in tols for valid in ((True, False) if noise == MAIN else (True,))]
CONDITIONS += [(MAIN, tol, valid, p) for p in (0.9, 0.5) for tol in REGIMES[MAIN][1:] for valid in (True, False)]
# The specificity control: a fifth entry, the standard deviation whose stated value is shown, holds the stated value
# (and the numbers) of the sd-10 or the sd-60 condition fixed while the stated standard deviation changes, crossing
# the value's actual accuracy (10, 60) with the stated sd (60, 30, 10). The no-description condition comes first, so
# that a --control run reads the ESS from its own curve.
CONTROL = [(MAIN, None, True, None)] + [(MAIN, tol, True, None, acc) for acc in (REGIMES[MAIN][2], REGIMES[MAIN][0]) for tol in REGIMES[MAIN]]


def make_tasks(noise, seed=0):
    """S tasks: w, numbers y (S, max(NS)), and for each tolerance a valid and an invalid stated value."""
    rng = np.random.default_rng([seed, noise])
    kept = []
    while sum(len(k["w"]) for k in kept) < S:
        w = rng.normal(MEAN, S0, S)
        cols = {"y": np.rint(w[:, None] + noise * rng.standard_normal((S, max(NS))))}
        for tol in REGIMES[noise]:
            r = tol**2 / S0**2  # so that w | m ~ N(m, tol^2) and w ~ N(MEAN, S0^2)
            cols[f"valid{tol}"] = np.rint(MEAN + (1 - r) * (w - MEAN) + np.sqrt(r * (1 - r)) * S0 * rng.standard_normal(S))
            cols[f"invalid{tol}"] = np.rint(MEAN + np.sqrt(1 - r) * S0 * rng.standard_normal(S))
        ok = np.all([((v >= LO) & (v <= HI)).reshape(S, -1).all(1) for v in cols.values()], 0)
        kept.append({"w": w[ok], **{k: v[ok].astype(int) for k, v in cols.items()}})
    return {k: np.concatenate([p[k] for p in kept])[:S] for k in kept[0]}


def stated(t, cond):
    """The stated value of a description condition (None where there is no description)."""
    return None if not cond[1] else t[f"{'valid' if cond[2] else 'invalid'}{cond[4] if len(cond) > 4 else cond[1]}"]


def prompt(t, cond, i, n):
    noise, tol, _, p = cond[:4]
    s = f"The numbers below are drawn from a normal distribution with standard deviation {noise}.\n"
    if tol is None:
        s += "Its mean is unknown.\n"
    else:
        s += PARAPHRASES[PARAPHRASE].format(m=stated(t, cond)[i], tol=tol) + "\n"
        if p:
            s += WRONG.format(q=round(100 * (1 - p))) + "\n"
    return s + "\n" + ("".join(f"{v}\n" for v in t["y"][i, :n]) + "\n" if n else "") + f"Predict the {'next' if n else 'first'} number. Reply with the number only."


def _ndtr(z):
    """Standard normal CDF (Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7)."""
    x = np.abs(z) / np.sqrt(2)
    tt = 1 / (1 + 0.3275911 * x)
    poly = tt * (0.254829592 + tt * (-0.284496736 + tt * (1.421413741 + tt * (-1.453152027 + tt * 1.061405429))))
    return 0.5 * (1 + np.sign(z) * (1 - poly * np.exp(-x * x)))


def ideal_mean(t, cond, n, p=1.0):
    """The Bayes-optimal mean prediction with n numbers shown, given only what the prompt says: the noise,
    the description if there is one (taken as relevant with probability p), and otherwise a mean that is
    unknown, which we read as uniform over the three-digit range LO..HI."""
    noise, y = cond[0], t["y"][:, :n]
    ysum = y.sum(1)

    def unknown():  # posterior mean and log marginal likelihood under the uniform prior on [LO, HI]
        if n == 0:
            return np.full(len(t["w"]), (LO + HI) / 2), np.zeros(len(t["w"]))
        m, v = ysum / n, noise**2 / n  # the likelihood in the mean is N(m, v) times a constant
        lo, hi = (LO - 0.5 - m) / np.sqrt(v), (HI + 0.5 - m) / np.sqrt(v)
        z = _ndtr(hi) - _ndtr(lo)  # mass of that Gaussian inside the range
        dens = (np.exp(-lo**2 / 2) - np.exp(-hi**2 / 2)) / np.sqrt(2 * np.pi)
        mean = m + np.sqrt(v) * dens / np.maximum(z, 1e-300)
        logml = -0.5 * ((y - m[:, None]) ** 2).sum(1) / noise**2 - 0.5 * n * np.log(2 * np.pi * noise**2) + 0.5 * np.log(2 * np.pi * v) + np.log(np.maximum(z, 1e-300)) - np.log(HI - LO + 1)
        return mean, logml

    def described(m0, v0):  # posterior mean and log marginal likelihood under the prior N(m0, v0)
        d = y - np.reshape(m0, (-1, 1))
        mu = (noise**2 / v0 * m0 + ysum) / (noise**2 / v0 + n)
        logml = -0.5 * (np.log1p(n * v0 / noise**2) + (d**2).sum(1) / noise**2 - v0 * d.sum(1) ** 2 / (noise**2 * (noise**2 + n * v0))) - 0.5 * n * np.log(2 * np.pi * noise**2)
        return mu, logml

    mu0, l0 = unknown()
    if not cond[1]:
        return mu0
    mu1, l1 = described(stated(t, cond).astype(float), float(cond[1]) ** 2)
    pi = 1 / (1 + (1 - p) / max(p, 1e-12) * np.exp(np.clip(l0 - l1, -700, 700)))
    return pi * mu1 + (1 - pi) * mu0


def weight(pred, t, cond, n):
    """The weight on the stated value: its coefficient when the prediction is regressed, across
    tasks, on the stated value and the mean of the numbers shown (with a constant)."""
    X = [stated(t, cond), np.ones(S)] + ([t["y"][:, :n].mean(1)] if n else [])
    return np.linalg.lstsq(np.column_stack(X), pred, rcond=None)[0][0]


def label(cond):
    noise, tol, valid, p = cond[:4]
    return f"noise {noise}, " + ("no description" if tol is None else f"{'relevant' if valid else 'irrelevant'} sd {tol}" + (f", {round(100 * (1 - p))}% may be wrong" if p else "") + (f", value of sd {cond[4]}" if len(cond) > 4 else ""))


def score(repo, out, batch=100, written=False, ablation=False, control=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="cuda").eval()
    pad = next(i for i in (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id) if i is not None)
    if written:
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        number_ids = [tok.encode(str(v), add_special_tokens=False) for v in GRID]
        assert all(len(i) == 1 for i in number_ids), "every three-digit integer must be one token"
        number_ids = torch.tensor([i[0] for i in number_ids], device="cuda")
    grid = torch.tensor(GRID, device="cuda", dtype=torch.float64)
    tasks = {noise: make_tasks(noise) for noise in REGIMES}

    def turn(content):  # no thinking before the answer
        text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text + "<|channel|>final<|message|>" if text.endswith("<|start|>assistant") else text  # gpt-oss: answer in the final channel

    conds = CONTROL if control else CONDITIONS  # a --control file's rows are the conditions of CONTROL
    res = {k: np.full((len(conds), S, len(NS)), np.nan, np.float32) for k in ("mean", "sd", "mass")}
    with torch.inference_mode():
        for ci, cond in enumerate(conds):
            if ablation and cond not in ABLATION:
                continue
            for ni, n in enumerate(NS):
                if ablation and n not in ABLATION_NS:
                    continue
                for a in range(0, S, batch):
                    if written:  # the number the model writes, by read_number; nan where it writes none
                        x = tok([turn(prompt(tasks[cond[0]], cond, i, n)) for i in range(a, min(a + batch, S))], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                        gen = model.generate(**x, max_new_tokens=8, do_sample=False)
                        for j, g in enumerate(gen[:, x["input_ids"].shape[1]:]):
                            v = read_number(tok.decode(g, skip_special_tokens=True))
                            res["mean"][ci, a + j, ni], res["sd"][ci, a + j, ni], res["mass"][ci, a + j, ni] = v, 0.0, float(np.isfinite(v))
                        continue
                    ids = [tok.encode(turn(prompt(tasks[cond[0]], cond, i, n)), add_special_tokens=False) for i in range(a, min(a + batch, S))]
                    L = max(map(len, ids))
                    x = torch.tensor([[pad] * (L - len(i)) + i for i in ids], device="cuda")  # padded on the left
                    mask = torch.tensor([[0] * (L - len(i)) + [1] * len(i) for i in ids], device="cuda")
                    last = torch.log_softmax(model(x, attention_mask=mask).logits[:, -1].double(), -1)
                    if ci == 0 and ni == 2 and a == 0:
                        print("the reply's most likely first tokens:", [(tok.decode(int(j)), round(float(v.exp()), 2)) for v, j in zip(*last[0].topk(6))], flush=True)
                    q = last[:, number_ids].exp()
                    mass = q.sum(-1)
                    q = q / mass[:, None]
                    mean = (q * grid).sum(-1)
                    res["mean"][ci, a : a + batch, ni] = mean.cpu()
                    res["sd"][ci, a : a + batch, ni] = ((q * grid**2).sum(-1) - mean**2).clamp_min(0).sqrt().cpu()
                    res["mass"][ci, a : a + batch, ni] = mass.cpu()
            print(f"{label(cond)}: {'wrote a number' if written else 'probability of a number'} {np.nanmean(res['mass'][ci], 0).round(2)}", flush=True)
    np.savez_compressed(out, repo=repo + (" (written)" if written else "") + (f" (paraphrase {PARAPHRASE})" if PARAPHRASE else "") + (" (control)" if control else ""), **res)
    print("done", flush=True)


REASON_CONDS = [(MAIN, None, True, None)] + [(MAIN, tol, True, None) for tol in REGIMES[MAIN]] + [(MAIN, REGIMES[MAIN][2], False, None), (MAIN, REGIMES[MAIN][2], True, 0.9), (MAIN, REGIMES[MAIN][2], True, 0.5)]
# the reasoning runs' (condition, n) cells: the no-description curve the ESS is read from, and every description
# (with the specificity control) at the headline numbers shown
REASON_CELLS = [(REASON_CONDS[0], n) for n in NS if n] + [(cond, n) for cond in REASON_CONDS[1:] + [c for c in CONTROL if c[1] and c[1] != c[4]] for n in (1, 2, 4)]
REASON_CELLS += [(REASON_CONDS[0], n) for n in (40, 48)]  # added for the Haiku runs: the c = 36 ESS is read between 32 and 64
REASON_TASKS = 200  # the first 200 of the main run's 2,000 tasks


def reason_prompts():
    """[(cell index, task index, prompt)] for the reasoning runs, on the main run's tasks (generated with S = 2000)."""
    t = make_tasks(MAIN)
    return [(k, i, prompt(t, cond, i, n)) for k, (cond, n) in enumerate(REASON_CELLS) for i in range(REASON_TASKS)]


def read_number(text):
    """The prediction in a text reply, the one rule for every readout that reads text: its first number with three
    digits before any decimal part, decimals kept (nan if none). The written runs in this folder were scored when
    that readout kept only the integer part; direct answers rarely carry a decimal part."""
    import re

    m = re.search(r"(?<![\d.])([1-9]\d\d(?:\.\d+)?)(?!\d|\.\d)", text)
    return float(m.group(1)) if m else np.nan


def reason(repo, out, budget=4096, batch=500):
    """gpt-oss reasons first (effort low, up to `budget` new tokens, greedy, served by vLLM); the number in its final
    channel is the prediction. Saves pred, finished (a final channel was reached) and the tokens generated, shaped
    (cell, task), and the replies."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(repo)
    llm = LLM(repo, max_model_len=budget + 1024, seed=0)
    params = SamplingParams(temperature=0, max_tokens=budget, skip_special_tokens=False)
    todo = reason_prompts()
    res = {k: np.full((len(REASON_CELLS), REASON_TASKS), np.nan, np.float32) for k in ("pred", "finished", "tokens")}
    replies = [""] * len(todo)
    for a in range(0, len(todo), batch):
        part = todo[a : a + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, reasoning_effort="low") for _, _, p in part]
        for j, ((k, i, _), o) in enumerate(zip(part, llm.generate(texts, params, use_tqdm=False))):
            text = replies[a + j] = o.outputs[0].text
            done = "<|channel|>final<|message|>" in text
            res["pred"][k, i] = read_number(text.split("<|channel|>final<|message|>")[-1]) if done else np.nan
            res["finished"][k, i], res["tokens"][k, i] = done, len(o.outputs[0].token_ids)
        if a == 0:
            print("example reply:", replies[0][:800].replace("\n", " "), flush=True)
        print(f"{min(a + batch, len(todo))}/{len(todo)} prompts, finished {np.nanmean(res['finished']):.3f}, tokens {np.nanmean(res['tokens']):.0f}", flush=True)
        np.savez_compressed(out, repo=repo + " (reasoning)", replies=np.array(replies), **res)  # after every batch
    print("done", flush=True)


def read(paths):
    global S
    S = int(np.load(paths[0])["mean"].shape[1])
    tasks = {noise: make_tasks(noise) for noise in REGIMES}
    rms = lambda pred, t: np.sqrt(((pred - t["w"]) ** 2).mean())
    for path in paths:
        d = np.load(path)
        print(f"\n===== {d['repo']}   (each cell: model / ideal; columns: n = {', '.join(map(str, NS))})")
        for ci, cond in enumerate(CONDITIONS[: d["mean"].shape[0]]):
            t, p = tasks[cond[0]], cond[3]
            ideal = [ideal_mean(t, cond, n, p if p else 1.0) for n in NS]
            row = f"{label(cond):46s} error " + "  ".join(f"{rms(d['mean'][ci, :, ni], t):5.1f}/{rms(ideal[ni], t):5.1f}" for ni in range(len(NS)))
            if cond[1]:
                row += "   weight " + "  ".join(f"{weight(d['mean'][ci, :, ni], t, cond, n):5.2f}/{weight(ideal[ni], t, cond, n):4.2f}" for ni, n in enumerate(NS))
            print(row + f"   P(number) {d['mass'][ci].mean():.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        if "--tasks" in sys.argv:
            S = int(sys.argv[sys.argv.index("--tasks") + 1])
        if "--paraphrase" in sys.argv:
            PARAPHRASE = int(sys.argv[sys.argv.index("--paraphrase") + 1])
        score(sys.argv[2], sys.argv[3], int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 100, "--written" in sys.argv, "--ablation" in sys.argv, "--control" in sys.argv)
    elif sys.argv[1:2] == ["reason"]:
        S = 2000
        reason(sys.argv[2], sys.argv[3], int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 4096)
    elif sys.argv[1:2] == ["prompts"]:  # the reasoning prompts as JSON, for models run through an API or a CLI
        import json

        S = 2000
        json.dump([dict(cell=k, task=i, prompt=p) for k, i, p in reason_prompts()], open(sys.argv[2], "w"))
    elif sys.argv[1:2] == ["read"]:
        read(sys.argv[2:])
    else:
        t = make_tasks(MAIN)
        for cond, n in ((CONDITIONS[0], 2), (CONDITIONS[3], 2), (CONDITIONS[-1], 0)):
            print(f"--- {label(cond)}, {n} numbers shown\n{prompt(t, cond, 0, n)}\n")
        for cond in CONDITIONS + CONTROL:  # the ideal learner that takes a relevant description as right has weight c / (c + n)
            if cond[1] and cond[2] and cond[3] is None:
                c = cond[0] ** 2 / cond[1] ** 2
                assert all(abs(weight(ideal_mean(make_tasks(cond[0]), cond, n), make_tasks(cond[0]), cond, n) - c / (c + n)) < 0.01 for n in NS), cond
        bad = next(c for c in CONDITIONS if not c[2] and c[3] == 0.5)
        assert abs(weight(ideal_mean(t, bad, 8, 0.5), t, bad, 8)) < 0.1  # a hedging learner drops an invalid description
        assert read_number("504.5, not 512") == 504.5 and read_number("It is 504.") == 504 and np.isnan(read_number("1000 or 99"))
        print(f"ok: {len(CONDITIONS)} conditions, {len(CONTROL)} control conditions, {len(REASON_CELLS)} reasoning cells")
