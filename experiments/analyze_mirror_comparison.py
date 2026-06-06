#!/usr/bin/env python3
"""
Mirror comparison analysis: GLM baseline vs directive-switched vs mirror-switched.

Reads three experiment result files and produces:
  1. Per-domain quality deltas across all three conditions
  2. Per-task quality deltas showing improvements/declines
  3. Intervention statistics (signal distribution)
  4. Summary of key findings

Output: mirror_comparison.json + formatted stdout table
"""

import json
import os
from collections import Counter

# ── File paths ──────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\User\mirrorfield\experiments\results"

PATHS = {
    "baseline": os.path.join(
        BASE_DIR, "glm_baseline", "20260215_083005", "glm_baseline_results.json"
    ),
    "directive": os.path.join(
        BASE_DIR, "glm_switched", "20260215_125244", "glm_switched_results.json"
    ),
    "mirror": os.path.join(
        BASE_DIR, "glm_mirror_switched", "20260215_202640", "glm_mirror_switched_results.json"
    ),
}


# ── Helpers ─────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_per_task(data):
    """Return dict  task_id -> {domain, composite}."""
    tasks = {}
    for result in data["results"]:
        tid = result["task_id"]
        domain = result["domain"]
        composite = result["quality_score"]["composite"]
        tasks[tid] = {"domain": domain, "composite": composite}
    return tasks


def extract_intervention_signals(data):
    """Walk every step of every task and collect triggered signals."""
    signals = []
    for result in data["results"]:
        for step in result.get("steps", []):
            det = step.get("intervention_detected")
            if det and det.get("triggered"):
                sig = det.get("intervention", {}).get("signal")
                if sig:
                    signals.append(sig)
    return signals


def round_val(v, digits=5):
    if v is None:
        return None
    return round(v, digits)


# ── Load data ───────────────────────────────────────────────────────────
print("Loading experiment results ...")
raw = {name: load_json(path) for name, path in PATHS.items()}

summaries = {name: d["summary"] for name, d in raw.items()}
per_task = {name: extract_per_task(d) for name, d in raw.items()}

# ── 1. Overall quality ──────────────────────────────────────────────────
overall = {
    name: {
        "mean_quality": summaries[name]["mean_quality"],
        "std_quality": summaries[name]["std_quality"],
    }
    for name in ["baseline", "directive", "mirror"]
}

# ── 2. Per-domain quality ───────────────────────────────────────────────
DOMAINS = ["research_questions", "design_exploration", "ethical_dilemmas", "novel_hypothesis"]

per_domain = {}
for name in ["baseline", "directive", "mirror"]:
    per_domain[name] = {}
    for dom in DOMAINS:
        per_domain[name][dom] = summaries[name]["per_domain"][dom]["mean_quality"]

# ── 3. Per-domain deltas ────────────────────────────────────────────────
domain_deltas = {}
for dom in DOMAINS:
    b = per_domain["baseline"][dom]
    d = per_domain["directive"][dom]
    m = per_domain["mirror"][dom]
    domain_deltas[dom] = {
        "baseline": round_val(b),
        "directive": round_val(d),
        "mirror": round_val(m),
        "directive_minus_baseline": round_val(d - b),
        "mirror_minus_baseline": round_val(m - b),
        "mirror_minus_directive": round_val(m - d),
    }

# ── 4. Per-task deltas ──────────────────────────────────────────────────
all_task_ids = sorted(per_task["baseline"].keys())

task_deltas = {}
for tid in all_task_ids:
    b = per_task["baseline"][tid]["composite"]
    d = per_task["directive"][tid]["composite"]
    m = per_task["mirror"][tid]["composite"]
    dom = per_task["baseline"][tid]["domain"]
    task_deltas[tid] = {
        "domain": dom,
        "baseline": round_val(b),
        "directive": round_val(d),
        "mirror": round_val(m),
        "directive_minus_baseline": round_val(d - b),
        "mirror_minus_baseline": round_val(m - b),
        "mirror_minus_directive": round_val(m - d),
        "mirror_vs_baseline": "improved" if m > b else ("declined" if m < b else "unchanged"),
        "directive_vs_baseline": "improved" if d > b else ("declined" if d < b else "unchanged"),
        "mirror_vs_directive": "improved" if m > d else ("declined" if m < d else "unchanged"),
    }

# ── 5. Intervention statistics ──────────────────────────────────────────
dir_signals = extract_intervention_signals(raw["directive"])
mir_signals = extract_intervention_signals(raw["mirror"])

dir_signal_counts = dict(Counter(dir_signals))
mir_signal_counts = dict(Counter(mir_signals))

intervention_stats = {
    "directive": {
        "total_interventions": summaries["directive"]["total_interventions"],
        "mean_interventions_per_task": summaries["directive"]["mean_interventions_per_task"],
        "signal_distribution": dir_signal_counts,
    },
    "mirror": {
        "total_interventions": summaries["mirror"]["total_interventions"],
        "mean_interventions_per_task": summaries["mirror"]["mean_interventions_per_task"],
        "signal_distribution": mir_signal_counts,
    },
}

# Per-domain intervention counts
for cond in ["directive", "mirror"]:
    dom_interventions = {}
    for dom in DOMAINS:
        dom_interventions[dom] = summaries[cond]["per_domain"][dom].get("total_interventions", 0)
    intervention_stats[cond]["per_domain_interventions"] = dom_interventions

# ── 6. Summary / key findings ───────────────────────────────────────────

# Count improvements/declines
def count_changes(condition_key):
    improved = sum(1 for t in task_deltas.values() if t[f"{condition_key}_vs_baseline"] == "improved")
    declined = sum(1 for t in task_deltas.values() if t[f"{condition_key}_vs_baseline"] == "declined")
    unchanged = sum(1 for t in task_deltas.values() if t[f"{condition_key}_vs_baseline"] == "unchanged")
    return improved, declined, unchanged

dir_imp, dir_dec, dir_unc = count_changes("directive")
mir_imp, mir_dec, mir_unc = count_changes("mirror")

# Best/worst domain deltas
best_mirror_domain = max(DOMAINS, key=lambda d: domain_deltas[d]["mirror_minus_baseline"])
worst_mirror_domain = min(DOMAINS, key=lambda d: domain_deltas[d]["mirror_minus_baseline"])
best_directive_domain = max(DOMAINS, key=lambda d: domain_deltas[d]["directive_minus_baseline"])
worst_directive_domain = min(DOMAINS, key=lambda d: domain_deltas[d]["directive_minus_baseline"])

overall_delta_directive = round_val(overall["directive"]["mean_quality"] - overall["baseline"]["mean_quality"])
overall_delta_mirror = round_val(overall["mirror"]["mean_quality"] - overall["baseline"]["mean_quality"])
overall_delta_mirror_vs_dir = round_val(overall["mirror"]["mean_quality"] - overall["directive"]["mean_quality"])

summary_section = {
    "overall_mean_quality": {
        "baseline": round_val(overall["baseline"]["mean_quality"]),
        "directive": round_val(overall["directive"]["mean_quality"]),
        "mirror": round_val(overall["mirror"]["mean_quality"]),
    },
    "overall_deltas": {
        "directive_minus_baseline": overall_delta_directive,
        "mirror_minus_baseline": overall_delta_mirror,
        "mirror_minus_directive": overall_delta_mirror_vs_dir,
    },
    "task_change_counts": {
        "directive_vs_baseline": {
            "improved": dir_imp,
            "declined": dir_dec,
            "unchanged": dir_unc,
        },
        "mirror_vs_baseline": {
            "improved": mir_imp,
            "declined": mir_dec,
            "unchanged": mir_unc,
        },
    },
    "best_domain_for_mirror": {
        "domain": best_mirror_domain,
        "delta": domain_deltas[best_mirror_domain]["mirror_minus_baseline"],
    },
    "worst_domain_for_mirror": {
        "domain": worst_mirror_domain,
        "delta": domain_deltas[worst_mirror_domain]["mirror_minus_baseline"],
    },
    "best_domain_for_directive": {
        "domain": best_directive_domain,
        "delta": domain_deltas[best_directive_domain]["directive_minus_baseline"],
    },
    "worst_domain_for_directive": {
        "domain": worst_directive_domain,
        "delta": domain_deltas[worst_directive_domain]["directive_minus_baseline"],
    },
    "key_findings": [
        (
            f"Both intervention conditions reduced overall quality relative to the "
            f"baseline (baseline {overall['baseline']['mean_quality']:.4f}; "
            f"directive {overall['directive']['mean_quality']:.4f}, "
            f"delta {overall_delta_directive:+.4f}; "
            f"mirror {overall['mirror']['mean_quality']:.4f}, "
            f"delta {overall_delta_mirror:+.4f})."
        ),
        (
            f"Mirror-switched outperformed directive-switched by "
            f"{overall_delta_mirror_vs_dir:+.5f} overall mean quality."
        ),
        (
            f"Directive interventions caused more tasks to decline ({dir_dec}/16) "
            f"than mirror interventions ({mir_dec}/16) relative to baseline."
        ),
        (
            f"Mirror interventions used fewer total interventions "
            f"({intervention_stats['mirror']['total_interventions']}) vs directive "
            f"({intervention_stats['directive']['total_interventions']}), suggesting "
            f"the non-directive style triggered fewer cooldown resets."
        ),
        (
            f"The ethical_dilemmas domain showed improvement under both conditions "
            f"(directive {domain_deltas['ethical_dilemmas']['directive_minus_baseline']:+.5f}, "
            f"mirror {domain_deltas['ethical_dilemmas']['mirror_minus_baseline']:+.5f}), "
            f"making it the most intervention-resilient domain."
        ),
        (
            f"The research_questions domain suffered the largest decline under both "
            f"conditions (directive {domain_deltas['research_questions']['directive_minus_baseline']:+.5f}, "
            f"mirror {domain_deltas['research_questions']['mirror_minus_baseline']:+.5f})."
        ),
    ],
}

# ── Build output JSON ───────────────────────────────────────────────────
output = {
    "metadata": {
        "description": "Comparison of GLM baseline vs directive-switched vs mirror-switched experiments",
        "conditions": {
            "baseline": "No interventions; pure GLM-4.7 generation",
            "directive": "Directive-style interventions injected based on geometric signals",
            "mirror": "Mirror-style (non-directive) interventions injected based on geometric signals",
        },
        "model": "glm-4.7",
        "n_tasks": 16,
        "steps_per_task": 4,
        "domains": DOMAINS,
    },
    "overall_quality": overall,
    "per_domain_quality": per_domain,
    "per_domain_deltas": domain_deltas,
    "per_task_deltas": task_deltas,
    "intervention_statistics": intervention_stats,
    "summary": summary_section,
}

# ── Write JSON ──────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "mirror_comparison.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nComparison JSON written to: {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════
# Formatted stdout summary
# ══════════════════════════════════════════════════════════════════════════
SEP = "=" * 90
THIN = "-" * 90


def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── Overall quality table ───────────────────────────────────────────────
header("OVERALL MEAN QUALITY")
print(f"  {'Condition':<20} {'Mean Quality':>14} {'Std Quality':>14} {'Delta vs Base':>14}")
print(f"  {THIN[2:]}")
for name, label in [("baseline", "Baseline"), ("directive", "Directive"), ("mirror", "Mirror")]:
    delta = overall[name]["mean_quality"] - overall["baseline"]["mean_quality"]
    delta_str = f"{delta:+.5f}" if name != "baseline" else "---"
    print(
        f"  {label:<20} {overall[name]['mean_quality']:>14.5f} "
        f"{overall[name]['std_quality']:>14.5f} {delta_str:>14}"
    )

# ── Per-domain quality table ────────────────────────────────────────────
header("PER-DOMAIN MEAN QUALITY")
print(f"  {'Domain':<24} {'Baseline':>10} {'Directive':>10} {'Mirror':>10} "
      f"{'Dir-Base':>10} {'Mir-Base':>10} {'Mir-Dir':>10}")
print(f"  {THIN[2:]}")
for dom in DOMAINS:
    dd = domain_deltas[dom]
    print(
        f"  {dom:<24} {dd['baseline']:>10.5f} {dd['directive']:>10.5f} "
        f"{dd['mirror']:>10.5f} {dd['directive_minus_baseline']:>+10.5f} "
        f"{dd['mirror_minus_baseline']:>+10.5f} {dd['mirror_minus_directive']:>+10.5f}"
    )

# ── Per-task quality table ──────────────────────────────────────────────
header("PER-TASK QUALITY SCORES AND DELTAS")
print(f"  {'Task ID':<16} {'Domain':<24} {'Base':>7} {'Dir':>7} {'Mir':>7} "
      f"{'D-B':>7} {'M-B':>7} {'M-D':>7} {'Mir vs B':<10}")
print(f"  {THIN[2:]}")
for tid in all_task_ids:
    td = task_deltas[tid]
    print(
        f"  {tid:<16} {td['domain']:<24} {td['baseline']:>7.3f} "
        f"{td['directive']:>7.3f} {td['mirror']:>7.3f} "
        f"{td['directive_minus_baseline']:>+7.3f} "
        f"{td['mirror_minus_baseline']:>+7.3f} "
        f"{td['mirror_minus_directive']:>+7.3f} "
        f"{td['mirror_vs_baseline']:<10}"
    )

# ── Improvement/decline summary ────────────────────────────────────────
header("TASK CHANGE SUMMARY (vs BASELINE)")
print(f"  {'Condition':<20} {'Improved':>10} {'Declined':>10} {'Unchanged':>10}")
print(f"  {THIN[2:]}")
print(f"  {'Directive':<20} {dir_imp:>10} {dir_dec:>10} {dir_unc:>10}")
print(f"  {'Mirror':<20} {mir_imp:>10} {mir_dec:>10} {mir_unc:>10}")

# ── Mirror vs directive task-level ──────────────────────────────────────
m_vs_d_imp = sum(1 for t in task_deltas.values() if t["mirror_vs_directive"] == "improved")
m_vs_d_dec = sum(1 for t in task_deltas.values() if t["mirror_vs_directive"] == "declined")
m_vs_d_unc = sum(1 for t in task_deltas.values() if t["mirror_vs_directive"] == "unchanged")
print()
print(f"  Mirror vs Directive:  improved={m_vs_d_imp}  declined={m_vs_d_dec}  unchanged={m_vs_d_unc}")

# ── Intervention statistics ─────────────────────────────────────────────
header("INTERVENTION STATISTICS")
for cond, label in [("directive", "Directive"), ("mirror", "Mirror")]:
    ist = intervention_stats[cond]
    print(f"\n  {label} condition:")
    print(f"    Total interventions:        {ist['total_interventions']}")
    print(f"    Mean per task:              {ist['mean_interventions_per_task']:.4f}")
    print(f"    Signal distribution:")
    for sig in ["terra_incognita", "framework_collision", "decision_boundary", "low_pr"]:
        cnt = ist["signal_distribution"].get(sig, 0)
        print(f"      {sig:<24} {cnt:>4}")
    print(f"    Per-domain interventions:")
    for dom in DOMAINS:
        cnt = ist["per_domain_interventions"].get(dom, 0)
        print(f"      {dom:<24} {cnt:>4}")

# ── Key findings ────────────────────────────────────────────────────────
header("KEY FINDINGS")
for i, finding in enumerate(summary_section["key_findings"], 1):
    print(f"  {i}. {finding}")
    print()

print(SEP)
print("  Analysis complete.")
print(SEP)
