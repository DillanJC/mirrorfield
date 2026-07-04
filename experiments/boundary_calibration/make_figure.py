# -*- coding: utf-8 -*-
"""Reliability-by-boundary-distance figure for METHODS_NOTE.md.
Pure visualization of logged numbers (boundary_margin_results.json +
platt_baseline_results.json). No new analysis, no claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
margin = json.loads((HERE / "boundary_margin_results.json").read_text())
platt = json.loads((HERE / "platt_baseline_results.json").read_text())
PLATT_KEY = {"42": "fit1337_eval42", "1337": "fit42_eval1337"}  # evaluated-on-seed

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, seed in zip(axes, ("42", "1337")):
    qs = margin[seed]["quintiles"]
    x = [b["q"] for b in qs]
    acc = [b["accuracy"] for b in qs]
    lo = [b["accuracy"] - b["acc_ci"][0] for b in qs]
    hi = [b["acc_ci"][1] - b["accuracy"] for b in qs]
    pint = [b["mean_pint"] for b in qs]
    pl = [b["platt_mean_p"] for b in platt[PLATT_KEY[seed]]["quintiles"]]

    ax.errorbar(x, acc, yerr=[lo, hi], fmt="o-", capsize=4, lw=2, ms=6,
                color="#1a7a3a", label="actual accuracy (Wilson 95%)")
    ax.plot(x, pint, "s--", lw=2, ms=6, color="#c0392b",
            label="frozen calibrator $p_{int}$")
    ax.plot(x, pl, "^:", lw=2, ms=6, color="#5b6ee1",
            label="Platt on margin (cross-seed held-out)")
    g = qs[0]["gap_pint_minus_acc"]
    ax.annotate(f"torn region:\ngate says {pint[0]:.2f},\naccuracy {acc[0]:.2f}  (gap +{g:.2f})",
                xy=(1, (acc[0] + pint[0]) / 2), xytext=(1.35, 0.62), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_title(f"seed {seed}  (500 items, quintiles of raw margin)", fontsize=10)
    ax.set_xlabel("raw mean-margin quintile  (1 = most torn)")
    ax.set_xticks(x)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("probability / accuracy")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("Aggregate calibration hid a near-boundary failure — Qwen2.5-3B, RTE+QNLI "
             "(on this model; §4y, §4z)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "boundary_reliability.png"
fig.savefig(out, dpi=160)
print(f"saved -> {out}")
