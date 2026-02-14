"""
Track 4 Evaluation — Compare switched vs baseline reasoning.

Analysis:
- Per-task quality delta (switched - baseline)
- Per-domain aggregation
- Intervention frequency and hit rate
- Key metric: Do interventions improve quality on tasks where they fire?
- Generates TRACK4_REPORT.md
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.track4_switches.run_baseline_reasoning import run_baseline, SEED
from experiments.track4_switches.run_switched_reasoning import run_switched

REPORT_PATH = PROJECT_ROOT / "experiments" / "track4_switches" / "TRACK4_REPORT.md"


def load_or_run(baseline_path: str = None, switched_path: str = None, seed: int = SEED):
    """Load existing results or run fresh experiments."""
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline from {baseline_path}")
    else:
        print("Running baseline...")
        baseline = run_baseline(seed=seed)

    if switched_path and Path(switched_path).exists():
        with open(switched_path) as f:
            switched = json.load(f)
        print(f"Loaded switched from {switched_path}")
    else:
        print("Running switched...")
        switched = run_switched(seed=seed)

    return baseline, switched


def compute_comparison(baseline: dict, switched: dict) -> dict:
    """Compute per-task and per-domain comparisons."""
    # Index baseline results by task_id
    baseline_by_task = {r['task_id']: r for r in baseline['results']}
    switched_by_task = {r['task_id']: r for r in switched['results']}

    task_comparisons = []
    for task_id in baseline_by_task:
        b = baseline_by_task[task_id]
        s = switched_by_task.get(task_id)
        if not s:
            continue

        b_quality = b['quality_score']['composite']
        s_quality = s['quality_score']['composite']
        delta = s_quality - b_quality
        n_interventions = s.get('n_interventions', 0)

        task_comparisons.append({
            'task_id': task_id,
            'domain': b['domain'],
            'baseline_quality': b_quality,
            'switched_quality': s_quality,
            'quality_delta': delta,
            'n_interventions': n_interventions,
            'intervention_fired': n_interventions > 0,
            'baseline_breadth': b['quality_score']['breadth'],
            'switched_breadth': s['quality_score']['breadth'],
            'baseline_depth': b['quality_score']['depth'],
            'switched_depth': s['quality_score']['depth'],
            'baseline_actionability': b['quality_score']['actionability'],
            'switched_actionability': s['quality_score']['actionability'],
            'baseline_uncertainty': b['quality_score']['uncertainty'],
            'switched_uncertainty': s['quality_score']['uncertainty'],
        })

    # Per-domain aggregation
    domains = set(tc['domain'] for tc in task_comparisons)
    domain_stats = {}
    for domain in sorted(domains):
        domain_tasks = [tc for tc in task_comparisons if tc['domain'] == domain]
        deltas = [tc['quality_delta'] for tc in domain_tasks]
        n_with_intervention = sum(1 for tc in domain_tasks if tc['intervention_fired'])
        domain_stats[domain] = {
            'n_tasks': len(domain_tasks),
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas)) if len(deltas) > 1 else 0.0,
            'n_with_interventions': n_with_intervention,
            'mean_baseline': float(np.mean([tc['baseline_quality'] for tc in domain_tasks])),
            'mean_switched': float(np.mean([tc['switched_quality'] for tc in domain_tasks])),
        }

    # Intervention effectiveness: quality delta on tasks where interventions fired
    tasks_with_interventions = [tc for tc in task_comparisons if tc['intervention_fired']]
    tasks_without_interventions = [tc for tc in task_comparisons if not tc['intervention_fired']]

    intervention_effectiveness = {
        'n_with_interventions': len(tasks_with_interventions),
        'n_without_interventions': len(tasks_without_interventions),
    }
    if tasks_with_interventions:
        intervention_effectiveness['mean_delta_with'] = float(
            np.mean([tc['quality_delta'] for tc in tasks_with_interventions])
        )
    if tasks_without_interventions:
        intervention_effectiveness['mean_delta_without'] = float(
            np.mean([tc['quality_delta'] for tc in tasks_without_interventions])
        )

    # Overall
    all_deltas = [tc['quality_delta'] for tc in task_comparisons]
    overall = {
        'n_tasks': len(task_comparisons),
        'mean_quality_delta': float(np.mean(all_deltas)),
        'std_quality_delta': float(np.std(all_deltas)) if len(all_deltas) > 1 else 0.0,
        'n_improved': sum(1 for d in all_deltas if d > 0),
        'n_degraded': sum(1 for d in all_deltas if d < 0),
        'n_unchanged': sum(1 for d in all_deltas if d == 0),
        'baseline_mean': baseline['summary']['mean_quality'],
        'switched_mean': switched['summary']['mean_quality'],
        'total_interventions': switched['summary']['total_interventions'],
    }

    return {
        'overall': overall,
        'domain_stats': domain_stats,
        'intervention_effectiveness': intervention_effectiveness,
        'task_comparisons': task_comparisons,
    }


def generate_report(comparison: dict, baseline: dict, switched: dict) -> str:
    """Generate TRACK4_REPORT.md content."""
    o = comparison['overall']
    d = comparison['domain_stats']
    ie = comparison['intervention_effectiveness']
    tc = comparison['task_comparisons']

    lines = [
        "# Track 4: Architectural Switches — Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seed:** {baseline['config']['seed']}",
        "",
        "## Summary",
        "",
        f"- **Tasks evaluated:** {o['n_tasks']}",
        f"- **Baseline mean quality:** {o['baseline_mean']:.4f}",
        f"- **Switched mean quality:** {o['switched_mean']:.4f}",
        f"- **Mean quality delta:** {o['mean_quality_delta']:+.4f}",
        f"- **Tasks improved:** {o['n_improved']} | "
        f"Degraded: {o['n_degraded']} | Unchanged: {o['n_unchanged']}",
        f"- **Total interventions fired:** {o['total_interventions']}",
        "",
        "## Per-Domain Results",
        "",
        "| Domain | N | Baseline | Switched | Delta | Interventions |",
        "|--------|---|----------|----------|-------|---------------|",
    ]

    for domain in sorted(d.keys()):
        ds = d[domain]
        lines.append(
            f"| {domain} | {ds['n_tasks']} | {ds['mean_baseline']:.3f} | "
            f"{ds['mean_switched']:.3f} | {ds['mean_delta']:+.3f} | "
            f"{ds['n_with_interventions']} |"
        )

    lines += [
        "",
        "## Intervention Effectiveness",
        "",
        f"- Tasks with interventions: {ie['n_with_interventions']}",
        f"- Tasks without interventions: {ie['n_without_interventions']}",
    ]
    if 'mean_delta_with' in ie:
        lines.append(f"- Mean delta (with interventions): {ie['mean_delta_with']:+.4f}")
    if 'mean_delta_without' in ie:
        lines.append(f"- Mean delta (without interventions): {ie['mean_delta_without']:+.4f}")

    lines += [
        "",
        "**Key question:** Do interventions improve quality on tasks where they fire?",
    ]
    if 'mean_delta_with' in ie:
        if ie['mean_delta_with'] > 0:
            lines.append(f"**YES** — interventions associated with +{ie['mean_delta_with']:.4f} quality gain.")
        elif ie['mean_delta_with'] < 0:
            lines.append(f"**NO** — interventions associated with {ie['mean_delta_with']:.4f} quality loss.")
        else:
            lines.append("**NEUTRAL** — no measurable effect.")

    lines += [
        "",
        "## Per-Task Breakdown",
        "",
        "| Task ID | Domain | Baseline | Switched | Delta | Interventions |",
        "|---------|--------|----------|----------|-------|---------------|",
    ]

    for t in tc:
        lines.append(
            f"| {t['task_id']} | {t['domain'][:12]} | {t['baseline_quality']:.3f} | "
            f"{t['switched_quality']:.3f} | {t['quality_delta']:+.3f} | "
            f"{t['n_interventions']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This report compares Task 4's hard-coded geometric interventions against "
        "an unmodified baseline. Quality is scored via heuristic rubric (breadth, "
        "depth, actionability, uncertainty acknowledgment).",
        "",
        "**Limitations:**",
        "- Simulated reasoning (not live LLM) — quality differences reflect "
        "intervention text injection, not genuine reasoning improvement",
        "- Quality scorer is heuristic-based — may not capture true response quality",
        "- Reference corpus is synthetic — real embeddings may produce different "
        "geometric signatures",
        "",
        "**Next step:** Track 5 (Recursive Self-Learning) uses these results "
        "as the starting policy and learns which interventions to keep, adjust, "
        "or discard.",
    ]

    return "\n".join(lines) + "\n"


def evaluate(
    baseline_path: str = None,
    switched_path: str = None,
    seed: int = SEED,
) -> dict:
    """Run full evaluation pipeline."""
    baseline, switched = load_or_run(baseline_path, switched_path, seed)

    print("\nComputing comparison...")
    comparison = compute_comparison(baseline, switched)

    print("\nGenerating report...")
    report = generate_report(comparison, baseline, switched)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    # Save comparison JSON
    comparison_path = REPORT_PATH.parent / "comparison_results.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Comparison data: {comparison_path}")
    print(f"\n=== Overall: {comparison['overall']['mean_quality_delta']:+.4f} quality delta ===")

    return comparison


if __name__ == "__main__":
    evaluate()
