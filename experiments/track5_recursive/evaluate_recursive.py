"""
Track 5 Evaluation — Final analysis of recursive self-learning.

Analysis:
- Learning curve: quality by iteration
- Track 4 vs Track 5 head-to-head on same tasks
- Goodhart summary: did learning help or just game metrics?
- Per-domain breakdown: which domains benefit most from adaptation?
- Generates TRACK5_REPORT.md
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.track5_recursive.run_recursive_trials import run_recursive_trials, SEED

REPORT_PATH = PROJECT_ROOT / "experiments" / "track5_recursive" / "TRACK5_REPORT.md"


def load_or_run(trials_path: str = None, seed: int = SEED) -> dict:
    """Load existing results or run fresh trials."""
    if trials_path and Path(trials_path).exists():
        with open(trials_path) as f:
            data = json.load(f)
        print(f"Loaded trials from {trials_path}")
        return data
    else:
        print("Running recursive trials...")
        return run_recursive_trials(seed=seed)


def analyze_learning_curve(trials_data: dict) -> dict:
    """Analyze quality trajectory across iterations."""
    iterations = trials_data['iterations']
    baseline_mean = trials_data['baseline_mean_quality']

    curve = []
    for it in iterations:
        snapshot = it['snapshot']
        curve.append({
            'iteration': it['iteration'],
            'mean_quality': snapshot['mean_quality'],
            'std_quality': snapshot['std_quality'],
            'diversity': snapshot['diversity'],
            'intervention_rate': snapshot['intervention_rate'],
            'delta_vs_baseline': snapshot['mean_quality'] - baseline_mean,
        })

    # Compute trend
    qualities = [c['mean_quality'] for c in curve]
    if len(qualities) >= 2:
        trend = np.polyfit(range(len(qualities)), qualities, 1)
        slope = float(trend[0])
    else:
        slope = 0.0

    return {
        'curve': curve,
        'baseline_mean': baseline_mean,
        'final_mean': qualities[-1] if qualities else 0.0,
        'total_delta': qualities[-1] - baseline_mean if qualities else 0.0,
        'trend_slope': slope,
        'improving': slope > 0.001,
    }


def analyze_goodhart(trials_data: dict) -> dict:
    """Summarize Goodhart detection results."""
    reports = trials_data.get('goodhart_reports', [])

    if not reports:
        return {'n_reports': 0, 'verdict': 'NO_DATA'}

    verdicts = [r['verdict'] for r in reports]
    red_counts = [r['n_red_triggered'] for r in reports]
    green_counts = [r['n_green_triggered'] for r in reports]

    # Check trend lines from last report
    last_report = reports[-1]
    trend_lines = last_report.get('trend_lines', {})

    return {
        'n_reports': len(reports),
        'verdicts': verdicts,
        'final_verdict': verdicts[-1],
        'total_red_flags': sum(red_counts),
        'total_green_flags': sum(green_counts),
        'mean_red_per_iteration': float(np.mean(red_counts)),
        'mean_green_per_iteration': float(np.mean(green_counts)),
        'trend_lines': trend_lines,
        'any_fail': 'FAIL' in verdicts,
        'any_warn': 'WARN' in verdicts,
    }


def analyze_per_domain(trials_data: dict) -> dict:
    """Analyze which domains benefit most from adaptation."""
    iterations = trials_data['iterations']
    baseline_mean = trials_data['baseline_mean_quality']

    # Collect per-domain quality across iterations
    domain_trajectories = {}
    for it in iterations:
        for domain, quality in it['snapshot']['per_domain_quality'].items():
            if domain not in domain_trajectories:
                domain_trajectories[domain] = []
            domain_trajectories[domain].append(quality)

    domain_analysis = {}
    for domain, qualities in domain_trajectories.items():
        if len(qualities) >= 2:
            trend = np.polyfit(range(len(qualities)), qualities, 1)
            slope = float(trend[0])
        else:
            slope = 0.0

        domain_analysis[domain] = {
            'initial_quality': qualities[0],
            'final_quality': qualities[-1],
            'delta': qualities[-1] - qualities[0],
            'trend_slope': slope,
            'improving': slope > 0.001,
        }

    return domain_analysis


def analyze_policy_evolution(trials_data: dict) -> dict:
    """Analyze how the policy evolved."""
    history = trials_data.get('policy_history', [])
    final_policy = trials_data.get('final_policy', [])

    evolution = {
        'n_iterations': len(history),
        'final_n_interventions': len(final_policy),
        'initial_n_interventions': len(history[0]['interventions']) if history else 0,
        'total_dropped': 0,
        'dropped_signals': [],
        'all_adjustments': [],
    }

    for ps in history:
        evolution['total_dropped'] += len(ps.get('dropped_signals', []))
        evolution['dropped_signals'].extend(ps.get('dropped_signals', []))
        evolution['all_adjustments'].extend(ps.get('adjustments', []))

    return evolution


def generate_report(
    learning_curve: dict,
    goodhart: dict,
    domain_analysis: dict,
    policy_evolution: dict,
    trials_data: dict,
) -> str:
    """Generate TRACK5_REPORT.md."""
    lines = [
        "# Track 5: Recursive Self-Learning — Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seed:** {trials_data['config']['seed']}",
        f"**Iterations:** {trials_data['config']['n_iterations']}",
        "",
        "## Summary",
        "",
        f"- **Baseline mean quality:** {learning_curve['baseline_mean']:.4f}",
        f"- **Final mean quality:** {learning_curve['final_mean']:.4f}",
        f"- **Total quality delta:** {learning_curve['total_delta']:+.4f}",
        f"- **Trend slope:** {learning_curve['trend_slope']:+.6f} "
        f"({'improving' if learning_curve['improving'] else 'flat/declining'})",
        f"- **Goodhart verdict:** {goodhart.get('final_verdict', 'N/A')}",
        f"- **Policy interventions:** {policy_evolution['initial_n_interventions']} initial "
        f"-> {policy_evolution['final_n_interventions']} final "
        f"({policy_evolution['total_dropped']} dropped)",
        "",
        "## Learning Curve",
        "",
        "| Iteration | Quality | Delta vs Baseline | Diversity | Intervention Rate |",
        "|-----------|---------|-------------------|-----------|-------------------|",
    ]

    for c in learning_curve['curve']:
        lines.append(
            f"| {c['iteration']} | {c['mean_quality']:.4f} | "
            f"{c['delta_vs_baseline']:+.4f} | {c['diversity']:.3f} | "
            f"{c['intervention_rate']:.3f} |"
        )

    lines += [
        "",
        "## Goodhart Analysis",
        "",
    ]

    if goodhart['n_reports'] == 0:
        lines.append("No Goodhart reports generated (fewer than 2 iterations).")
    else:
        lines.append(f"- **Reports generated:** {goodhart['n_reports']}")
        lines.append(f"- **Final verdict:** {goodhart['final_verdict']}")
        lines.append(f"- **Total red flags:** {goodhart['total_red_flags']}")
        lines.append(f"- **Total green flags:** {goodhart['total_green_flags']}")
        lines.append(f"- **Any FAIL:** {'Yes' if goodhart['any_fail'] else 'No'}")
        lines.append(f"- **Any WARN:** {'Yes' if goodhart['any_warn'] else 'No'}")

        lines += [
            "",
            "### Verdict History",
            "",
            "| Iteration | Verdict |",
            "|-----------|---------|",
        ]
        for i, v in enumerate(goodhart['verdicts'], start=2):
            lines.append(f"| {i} | {v} |")

    lines += [
        "",
        "## Per-Domain Analysis",
        "",
        "| Domain | Initial | Final | Delta | Trend |",
        "|--------|---------|-------|-------|-------|",
    ]

    for domain in sorted(domain_analysis.keys()):
        da = domain_analysis[domain]
        lines.append(
            f"| {domain} | {da['initial_quality']:.4f} | "
            f"{da['final_quality']:.4f} | {da['delta']:+.4f} | "
            f"{'improving' if da['improving'] else 'flat/declining'} |"
        )

    lines += [
        "",
        "## Policy Evolution",
        "",
        f"- **Initial interventions:** {policy_evolution['initial_n_interventions']}",
        f"- **Final interventions:** {policy_evolution['final_n_interventions']}",
        f"- **Signals dropped:** {', '.join(policy_evolution['dropped_signals']) or 'none'}",
        "",
        "### Adjustment Log",
        "",
    ]

    for adj in policy_evolution['all_adjustments']:
        lines.append(f"- {adj}")

    lines += [
        "",
        "## Track 4 vs Track 5 Head-to-Head",
        "",
        f"- **Track 4 (switches):** Hard-coded interventions",
        f"- **Track 5 (recursive):** Learned policy after {trials_data['config']['n_iterations']} iterations",
        f"- **Baseline:** No interventions",
        "",
        f"Track 5 final quality ({learning_curve['final_mean']:.4f}) vs "
        f"baseline ({learning_curve['baseline_mean']:.4f}): "
        f"{learning_curve['total_delta']:+.4f}",
        "",
        "## Interpretation",
        "",
    ]

    if learning_curve['improving'] and not goodhart.get('any_fail', False):
        lines.append(
            "The recursive learner shows genuine improvement: quality increases "
            "across iterations without triggering Goodhart FAIL flags. The learned "
            "policy appears to be making beneficial adaptations."
        )
    elif learning_curve['improving'] and goodhart.get('any_fail', False):
        lines.append(
            "**CAUTION:** Quality metrics improve but Goodhart detector raised FAIL "
            "flags. The improvement may reflect metric gaming rather than genuine "
            "reasoning quality gains. Investigate red flags before trusting results."
        )
    elif not learning_curve['improving'] and not goodhart.get('any_fail', False):
        lines.append(
            "The recursive learner did not improve quality beyond baseline. "
            "The hard-coded Track 4 interventions may already be near-optimal, "
            "or the learning mechanism may need refinement."
        )
    else:
        lines.append(
            "Quality declined and Goodhart flags were raised. The recursive "
            "learning process appears to have degraded performance. The initial "
            "Track 4 policy is likely superior."
        )

    lines += [
        "",
        "**Limitations:**",
        "- Simulated reasoning — genuine LLM responses may show different patterns",
        "- Heuristic quality scorer — limited sensitivity to real quality differences",
        "- Synthetic reference corpus — real embeddings needed for production validation",
        "",
        "**Next steps:**",
        "- Integrate with live LLM for genuine reasoning experiments",
        "- Replace synthetic embeddings with sentence-transformer embeddings",
        "- Test on larger, more diverse task banks",
    ]

    return "\n".join(lines) + "\n"


def evaluate(trials_path: str = None, seed: int = SEED) -> dict:
    """Run full evaluation pipeline."""
    trials_data = load_or_run(trials_path, seed)

    print("\nAnalyzing learning curve...")
    learning_curve = analyze_learning_curve(trials_data)

    print("Analyzing Goodhart flags...")
    goodhart = analyze_goodhart(trials_data)

    print("Analyzing per-domain results...")
    domain_analysis = analyze_per_domain(trials_data)

    print("Analyzing policy evolution...")
    policy_evolution = analyze_policy_evolution(trials_data)

    print("Generating report...")
    report = generate_report(
        learning_curve, goodhart, domain_analysis, policy_evolution, trials_data
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    # Save analysis JSON
    analysis_path = REPORT_PATH.parent / "evaluation_results.json"
    analysis = {
        'learning_curve': learning_curve,
        'goodhart': goodhart,
        'domain_analysis': domain_analysis,
        'policy_evolution': policy_evolution,
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Analysis data: {analysis_path}")
    print(f"\n=== Final quality: {learning_curve['final_mean']:.4f} "
          f"(delta: {learning_curve['total_delta']:+.4f}) ===")
    print(f"=== Goodhart: {goodhart.get('final_verdict', 'N/A')} ===")

    return analysis


if __name__ == "__main__":
    evaluate()
