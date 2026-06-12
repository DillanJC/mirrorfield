# -*- coding: utf-8 -*-
"""
Plan C Steps 3-4 — does the gate demonstrably help, on fresh items, live?

  --run      generation pass (GPU, ~60-120 min): 450 fresh RTE (train split) +
             500 fresh QNLI + 150 fresh SST-2 (both validation minus
             calibration indices, reproduced via the SHARED-RNG replay),
             answered by GatedAgent (InProcessGate), per-task context_id.
             Saves every row (features, prediction, correctness, indices).
  --analyze  CPU on saved rows: P1 (AUC + CI, pooled & per-task, stream
             orders 123/43/44), P2 (error-recall minus REALIZED abstention
             at frozen tau), risk-coverage curve, mixed-stream control,
             shuffled-label nulls, parse-sensitivity. Verdict against the
             pre-registered criteria in gate_thresholds.json (commit b1fdc08).

Labels = GLUE gold. The gate never defines its own target.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from calibrate_gate import TASKS, _first_of  # noqa  frozen prompts + parser
from gate_agent import GatedAgent, InProcessGate, LocalLM, THRESHOLDS  # noqa
from mirrorfield.mcp.uncertainty import RollingGate  # noqa

SEED_SELECT = 123
WARMUP = 20
N_BOOT = 2000
ROWS_NPZ = HERE / "eval_gate_value_rows.npz"
TEXTS_JSON = HERE / "eval_gate_value_texts.json"
RESULTS = HERE / "eval_gate_value_results.json"
PLOT = HERE / "risk_coverage.png"

FRESH = {"rte": ("train", 450), "qnli": ("validation", 500),
         "sst2": ("validation", 150)}
POS = {"rte": ("yes", "no"), "qnli": ("yes", "no"),
       "sst2": ("positive", "negative")}


def calibration_exclusion_indices():
    """Replay calibrate_gate.main's index selection with ONE shared
    RandomState(42) in the original task order (rte, qnli, sst2). The shared
    state is load-bearing: each shuffle advances it."""
    from datasets import load_dataset
    rng = np.random.RandomState(42)
    excl = {}
    for t in ["rte", "qnli", "sst2"]:
        ds = load_dataset("glue", t, split="validation")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        excl[t] = set(idx[:250])
        assert len(excl[t]) == 250
    return excl


def select_fresh():
    from datasets import load_dataset
    excl = calibration_exclusion_indices()
    rng = np.random.RandomState(SEED_SELECT)
    items = []
    for t, (split, n) in FRESH.items():
        ds = load_dataset("glue", t, split=split)
        cand = list(range(len(ds)))
        if split == "validation":
            cand = [i for i in cand if i not in excl[t]]
        rng.shuffle(cand)
        chosen = cand[:n]
        if split == "validation":
            assert not (set(chosen) & excl[t]), f"contamination in {t}"
        cfg = TASKS[t]
        f = cfg["fields"]
        for i in chosen:
            ex = ds[i]
            prompt = cfg["prompt"].format(*[ex[k] for k in f])
            items.append({"task": t, "ds_index": int(i), "split": split,
                          "prompt": prompt, "gold": cfg["gold"](ex["label"])})
    return items


def run_generation():
    items = select_fresh()
    counts = {}
    for it in items:
        counts[it["task"]] = counts.get(it["task"], 0) + 1
    print(f"fresh items: {counts} (total {len(items)})")

    lm = LocalLM()
    agent = GatedAgent(lm, InProcessGate())
    rows, texts = [], []
    pos_in_stream = {}
    for k, it in enumerate(items):           # stream = selection order (123)
        t = it["task"]
        pos_in_stream[t] = pos_in_stream.get(t, -1) + 1
        r = agent.answer(it["prompt"], context_id=t)
        pred = _first_of(r["answer_text"], *POS[t])
        correct = int(pred is not None and pred == it["gold"])
        rows.append({
            "task": t, "ds_index": it["ds_index"], "pos": pos_in_stream[t],
            "mm": r["features"][0], "me": r["features"][1],
            "br": r["features"][2],
            "p_rel_gen": (r["p_correct_relative"]
                          if r["p_correct_relative"] is not None else np.nan),
            "decision_gen": r["decision"], "parse_ok": int(pred is not None),
            "correct": correct,
        })
        texts.append({"task": t, "ds_index": it["ds_index"],
                      "answer": r["answer_text"], "gold": it["gold"]})
        if (k + 1) % 100 == 0:
            acc = np.mean([x["correct"] for x in rows])
            print(f"  {k+1}/{len(items)}  running acc={acc:.3f}")

    np.savez(ROWS_NPZ,
             task=np.array([r["task"] for r in rows]),
             ds_index=np.array([r["ds_index"] for r in rows]),
             pos=np.array([r["pos"] for r in rows]),
             X=np.array([[r["mm"], r["me"], r["br"]] for r in rows]),
             p_rel_gen=np.array([r["p_rel_gen"] for r in rows]),
             decision_gen=np.array([r["decision_gen"] for r in rows]),
             parse_ok=np.array([r["parse_ok"] for r in rows]),
             correct=np.array([r["correct"] for r in rows]))
    TEXTS_JSON.write_text(json.dumps(texts))
    print(f"saved {len(rows)} rows -> {ROWS_NPZ}")


# ---------------------------------------------------------------------------
def rescore(X, task, base_order_pos, order_seed=None):
    """Fresh RollingGate per task. order_seed None = generation order (pos)."""
    scores = np.full(len(task), np.nan)
    pos = np.full(len(task), -1)
    if order_seed is None:
        for t in np.unique(task):
            idx = np.where(task == t)[0]
            order = idx[np.argsort(base_order_pos[idx])]
            g = RollingGate(window=50, min_history=5)
            for j, i in enumerate(order):
                r = g.score(str(t), *X[i])
                pos[i] = j
                scores[i] = (r["p_correct_relative"]
                             if r["p_correct_relative"] is not None else np.nan)
    else:
        rng = np.random.RandomState(order_seed)
        for t in sorted(np.unique(task)):
            idx = np.where(task == t)[0]
            order = rng.permutation(idx)
            g = RollingGate(window=50, min_history=5)
            for j, i in enumerate(order):
                r = g.score(str(t), *X[i])
                pos[i] = j
                scores[i] = (r["p_correct_relative"]
                             if r["p_correct_relative"] is not None else np.nan)
    return scores, pos


def auc_ci(y_wrong, s, rng):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_wrong, -s)        # low score should mean wrong
    vals = []
    n = len(y_wrong)
    for _ in range(N_BOOT):
        bi = rng.randint(0, n, n)
        if len(np.unique(y_wrong[bi])) < 2:
            continue
        vals.append(roc_auc_score(y_wrong[bi], -s[bi]))
    return (round(float(auc), 4),
            [round(float(np.percentile(vals, 2.5)), 4),
             round(float(np.percentile(vals, 97.5)), 4)])


def analyze():
    d = np.load(ROWS_NPZ, allow_pickle=True)
    task, pos, X = d["task"], d["pos"], d["X"]
    correct, parse_ok = d["correct"], d["parse_ok"]
    wrong = 1 - correct
    rng = np.random.RandomState(42)
    res = {"n_total": int(len(task)),
           "acc_overall_by_task": {str(t): round(float(correct[task == t].mean()), 4)
                                   for t in np.unique(task)}}

    # ---- (a) P1: per-context scoring, generation order ----------------------
    s_base, pos_base = rescore(X, task, pos)
    gen_match = np.nanmax(np.abs(s_base - d["p_rel_gen"])) if np.any(
        ~np.isnan(d["p_rel_gen"])) else 0.0
    res["rescore_matches_generation"] = bool(gen_match < 1e-9)

    def scored_mask(scores, posv, tasks_in):
        return ((posv >= WARMUP) & np.isin(task, tasks_in)
                & ~np.isnan(scores))

    m = scored_mask(s_base, pos_base, ["rte", "qnli"])
    auc, ci = auc_ci(wrong[m], s_base[m], rng)
    res["P1_pooled"] = {"auc": auc, "ci95": ci, "n": int(m.sum()),
                        "n_wrong": int(wrong[m].sum())}
    for t in ["rte", "qnli"]:
        mt = scored_mask(s_base, pos_base, [t])
        a, c = auc_ci(wrong[mt], s_base[mt], rng)
        res[f"P1_{t}"] = {"auc": a, "ci95": c, "n": int(mt.sum()),
                          "n_wrong": int(wrong[mt].sum())}

    # ---- (e) order replication seeds 43/44 ---------------------------------
    res["P1_order_replication"] = {}
    for seed in (43, 44):
        s_o, p_o = rescore(X, task, pos, order_seed=seed)
        mo = scored_mask(s_o, p_o, ["rte", "qnli"])
        a, c = auc_ci(wrong[mo], s_o[mo], rng)
        res["P1_order_replication"][str(seed)] = {"auc": a, "ci95": c}

    # ---- (b) P2: selective prediction at frozen tau -------------------------
    tau = THRESHOLDS["tau_abstain"]
    sm, ym = s_base[m], wrong[m]
    abst = sm < tau
    realized = float(abst.mean())
    recall = float(ym[abst].sum() / ym.sum()) if ym.sum() else 0.0
    diffs, recalls, rates = [], [], []
    idx_all = np.arange(len(ym))
    for _ in range(N_BOOT):
        bi = rng.choice(idx_all, len(idx_all), replace=True)
        yb, sb = ym[bi], sm[bi]
        if yb.sum() == 0:
            continue
        ab = sb < tau
        r_ = yb[ab].sum() / yb.sum()
        rate_ = ab.mean()
        diffs.append(r_ - rate_)
        recalls.append(r_)
        rates.append(rate_)
    res["P2"] = {
        "tau_abstain": tau,
        "realized_abstention_rate": round(realized, 4),
        "error_recall": round(recall, 4),
        "recall_minus_rate": round(recall - realized, 4),
        "diff_ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                      round(float(np.percentile(diffs, 97.5)), 4)],
        "acc_presented": round(float(1 - ym[~abst].mean()), 4),
        "acc_overall": round(float(1 - ym.mean()), 4),
    }
    # rank-based descriptive (exact bottom-20%)
    k20 = int(0.2 * len(sm))
    bottom = np.argsort(sm)[:k20]
    res["P2"]["rank20_recall"] = round(float(ym[bottom].sum() / ym.sum()), 4)

    # risk-coverage curve
    qs = np.linspace(0.5, 1.0, 26)
    curve = []
    order = np.argsort(-sm)                  # keep most-confident first
    for cov in qs:
        kk = max(1, int(cov * len(sm)))
        kept = order[:kk]
        curve.append({"coverage": round(float(cov), 3),
                      "risk": round(float(ym[kept].mean()), 4)})
    res["risk_coverage"] = curve

    # ---- (c) mixed-stream negative control ----------------------------------
    rngm = np.random.RandomState(999)
    order_all = rngm.permutation(len(task))
    g = RollingGate(window=50, min_history=5)
    s_mix = np.full(len(task), np.nan)
    p_mix = np.full(len(task), -1)
    for j, i in enumerate(order_all):
        r = g.score("GLOBAL", *X[i])
        p_mix[i] = j
        s_mix[i] = (r["p_correct_relative"]
                    if r["p_correct_relative"] is not None else np.nan)
    mm_ = ((p_mix >= WARMUP) & np.isin(task, ["rte", "qnli"])
           & ~np.isnan(s_mix))
    a, c = auc_ci(wrong[mm_], s_mix[mm_], rng)
    res["mixed_stream_control"] = {"auc": a, "ci95": c,
                                   "degrades_vs_per_context":
                                   bool(res["P1_pooled"]["auc"] - a >= 0.05)}

    # ---- (d) shuffled-label nulls -------------------------------------------
    nulls = []
    for _ in range(10):
        ysh = ym.copy()
        rng.shuffle(ysh)
        from sklearn.metrics import roc_auc_score
        nulls.append(round(float(roc_auc_score(ysh, -sm)), 4))
    res["shuffled_nulls"] = {"values": nulls,
                             "mean": round(float(np.mean(nulls)), 4),
                             "max": round(float(np.max(nulls)), 4)}

    # ---- (g) parse-sensitivity ----------------------------------------------
    mp = m & (parse_ok == 1)
    if wrong[mp].sum() > 0:
        a, c = auc_ci(wrong[mp], s_base[mp], rng)
        res["sensitivity_parse_ok_only"] = {
            "auc": a, "ci95": c,
            "n_unparseable_in_scored": int((m & (parse_ok == 0)).sum())}

    # ---- verdict -------------------------------------------------------------
    p1 = (res["P1_pooled"]["auc"] >= 0.58
          and res["P1_pooled"]["ci95"][0] > 0.50)
    orders_ok = all(v["ci95"][0] > 0.50
                    for v in res["P1_order_replication"].values())
    spread = max(res["P1_pooled"]["auc"],
                 *[v["auc"] for v in res["P1_order_replication"].values()]) - \
        min(res["P1_pooled"]["auc"],
            *[v["auc"] for v in res["P1_order_replication"].values()])
    p2 = res["P2"]["diff_ci95"][0] > 0
    if p1 and p2:
        verdict = ("PASS: signal survives the live path AND the gate "
                   "demonstrably helps at the pre-registered operating point")
    elif p1:
        verdict = ("MODEST GATE: ranking signal real (P1 pass) but "
                   "operating-point value not demonstrated (P2 CI includes 0)")
    elif res["P1_qnli"]["ci95"][0] > 0.50:
        verdict = ("RTE-TRAIN-ARM SHIFT: pooled P1 failed but QNLI-alone "
                   "holds - foundation intact, transfer claim bounded")
    else:
        verdict = ("P1 FAILED incl. QNLI-alone: run the prescribed debugging "
                   "pass before any abandon talk")
    res["P1_pass"] = bool(p1)
    res["P1_order_spread"] = round(float(spread), 4)
    res["P1_orders_replicate"] = bool(orders_ok and spread < 0.06)
    res["P2_pass"] = bool(p2)
    res["verdict"] = verdict

    RESULTS.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["P1_pooled", "P1_rte", "P1_qnli", "P2",
                       "mixed_stream_control", "shuffled_nulls",
                       "P1_order_spread", "verdict"]}, indent=1))

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cov = [p["coverage"] for p in curve]
        risk = [p["risk"] for p in curve]
        plt.figure(figsize=(7, 5))
        plt.plot(cov, risk, "b-o", ms=3, label="gate (keep most confident)")
        plt.axhline(float(ym.mean()), color="gray", ls="--",
                    label="random abstention (overall error rate)")
        plt.axvline(1 - res["P2"]["realized_abstention_rate"], color="r",
                    ls=":", label=f"operating point (abstain "
                    f"{res['P2']['realized_abstention_rate']:.0%})")
        plt.xlabel("coverage (fraction of answers presented)")
        plt.ylabel("risk (error rate among presented)")
        plt.title("Risk-coverage, fresh RTE+QNLI, pre-registered threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT, dpi=130)
        print(f"plot -> {PLOT}")
    except Exception as e:  # noqa
        print(f"(plot skipped: {e})")
    print(f"\nVERDICT: {verdict}\nsaved -> {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_generation()
    if a.analyze:
        analyze()
    if not (a.run or a.analyze):
        print("dry: use --run (GPU generation) then --analyze (CPU)")
