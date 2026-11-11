"""error_table.py: root-mean-square error with the description, beside the ESS of reason_table.tex.

    cd results/llm-test && python3 error_table.py     # after reasoning.py

Same runs and the same first 200 tasks as reasoning.py and figure_data.py: runs in reasoning.csv are read from it
(columns error1/2/4), and the open models without reasoning are scored from their 2,000-task arrays as in
figure_data.py. Writes error_table.tex."""
import csv
import pathlib

import numpy as np

import figure_data as F
import llm_test as T

here = pathlib.Path(__file__).parent
CONDS = {"relevant sd 30": (60, 30, True, None), "relevant sd 10": (60, 10, True, None)}  # c = 4, c = 36
ORDER = [("GPT-6 Sol, without reasoning", "GPT-6 Sol, none"), ("GPT-6 Sol, with reasoning (low)", "GPT-6 Sol, low"),
         ("Opus 4.6, without reasoning", "Opus 4.6, no thinking"), ("Opus 4.6, with reasoning (high)", "Opus 4.6, high"),
         ("Gemma 4 31B, without reasoning", "Gemma 4 31B"), ("Gemma 4 31B, with reasoning", "Gemma 4 31B, reasoning"),
         ("Qwen3.5 9B, without reasoning", "Qwen3.5 9B"), ("Qwen3.5 9B, with reasoning", "Qwen3.5 9B, reasoning"),
         ("gpt-oss-20b, without reasoning", "gpt-oss-20b"), ("gpt-oss-20b, with reasoning (low)", "gpt-oss-20b, low"),
         ("Haiku 5.5, with reasoning (medium)", "Haiku 5.5, medium"), ("Haiku 5.5, with reasoning (low)", "Haiku 5.5, low"),
         ("Olmo 3 7B Think, with reasoning", "Olmo 3 7B Think")]


def main():
    err = {}  # (run, condition) -> [rmse at n = 1, 2, 4]
    for r in csv.DictReader(open(here / "reasoning.csv")):
        if r["condition"] in CONDS:
            err[r["run"], r["condition"]] = [float(r[f"error{n}"]) for n in (1, 2, 4)]
            err["Reference", r["condition"]] = [float(r[f"ideal_error{n}"]) for n in (1, 2, 4)] if r["run"] == "Opus 4.6, high" else err.get(("Reference", r["condition"]))
    T.S = 2000
    t = {k: v[: F.K] for k, v in T.make_tasks(60).items()}
    for model, (main_file, _) in F.OPEN.items():
        d = np.load(here / main_file)
        for name, cond in CONDS.items():
            ci = T.CONDITIONS.index(cond)
            err[model, name] = [float(np.sqrt(np.nanmean((d["mean"][ci, : F.K, T.NS.index(n)].astype(float) - t["w"]) ** 2))) for n in (1, 2, 4)]
    line = lambda label, run: label + " & " + " & ".join(f"${v:.1f}$" for name in CONDS for v in err[run, name]) + " \\\\"
    out = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "& \\multicolumn{3}{c}{$c=4$} & \\multicolumn{3}{c}{$c=36$} \\\\",
           "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}", "Run & $n=1$ & $2$ & $4$ & $1$ & $2$ & $4$ \\\\", "\\midrule", line("Reference", "Reference")]
    out += [line(label, run) for label, run in ORDER] + ["\\bottomrule", "\\end{tabular}"]
    (here / "error_table.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
