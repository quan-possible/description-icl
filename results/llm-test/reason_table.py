"""Write reason_table.tex: gpt-oss-20b with low reasoning effort (100 tasks) against the Bayes-optimal predictor."""
import numpy as np, llm_test as L
L.S = 100
t = L.make_tasks(60)
pred = np.load("reason_gptoss20.npz")["mean"]
rows = []
for cond in L.REASON_CONDS:
    ci = L.CONDITIONS.index(cond)
    cells = []
    for n in L.REASON_NS:
        p = pred[ci, :, L.NS.index(n)]; ok = np.isfinite(p)
        ideal = L.ideal_mean(t, cond, n, p=cond[3] or 1.0)
        err = np.sqrt(np.mean((p[ok] - t["w"][ok]) ** 2)); ierr = np.sqrt(np.mean((ideal - t["w"]) ** 2))
        if cond[1]:
            X = np.c_[L.stated(t, cond), t["y"][:, :n].mean(1), np.ones(L.S)]
            a = np.linalg.lstsq(X[ok], p[ok], rcond=None)[0][0]
            ia = np.linalg.lstsq(X, ideal, rcond=None)[0][0]
            w = f"{a:.2f}/{ia:.2f}"
        else:
            w = "--"
        e = f"{err:.0f}/{ierr:.0f}"
        if ok.mean() < 0.75: w, e = f"\\textit{{{w}}}", f"\\textit{{{e}}}"
        cells += [f"{ok.mean():.2f}", w, e]
    name = L.label(cond).replace("noise 60, ", "").replace("relevant", "rel.").replace("irrelevant", "irrel.").replace("sd ", r"$\tau = ").replace(", 10% may be wrong", "$, 10\\% w.").replace(", 50% may be wrong", "$, 50\\% w.")
    if "$\\tau" in name and not name.endswith("w.") : name += "$"
    rows.append(" & ".join([name] + cells) + r" \\")
out = [r"\begin{tabular}{lcccccc}", r"\toprule",
       r"& \multicolumn{3}{c}{$n = 1$} & \multicolumn{3}{c}{$n = 4$} \\", r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
       r"Description & number & weight & error & number & weight & error \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
open("reason_table.tex", "w").write("\n".join(out) + "\n")
print("\n".join(rows))
