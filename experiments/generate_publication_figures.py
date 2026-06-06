"""
Generate Publication Figures for Geometric Features Paper

Creates 4 publication-quality figures:
1. Variance Decomposition (between-seed vs within-seed)
2. Per-Seed Distributions (violin plots)
3. Overall ΔR² Distribution with 95% CI
4. All 50 Runs Timeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Load results
results_dir = Path("C:/Users/User/mirrorfield/runs/multirun_boundary_20260108_082252")
with open(results_dir / "summary.json", 'r') as f:
    data = json.load(f)

# Extract data
seeds = data['seeds']
all_results = data['all_results']

# Organize data by seed
seed_data = {}
for seed_result in all_results:
    seed = seed_result['seed']
    borderline_deltas = [r['borderline']['delta_r2'] for r in seed_result['runs']]
    seed_data[seed] = borderline_deltas

# Flatten all deltas
all_deltas = []
for deltas in seed_data.values():
    all_deltas.extend(deltas)
all_deltas = np.array(all_deltas)

# Statistics
overall_mean = data['borderline_statistics']['mean']
ci_low = data['borderline_statistics']['ci_95_low']
ci_high = data['borderline_statistics']['ci_95_high']
between_seed_std = data['variance_decomposition']['borderline']['between_seed_std']
within_seed_std = data['variance_decomposition']['borderline']['within_seed_std']

# Create output directory
output_dir = Path("C:/Users/User/mirrorfield/runs/multirun_boundary_20260108_082252/figures")
output_dir.mkdir(exist_ok=True)

print("Generating publication figures...")
print(f"Output directory: {output_dir}\n")

# ============================================================================
# Figure 1: Variance Decomposition
# ============================================================================

print("Figure 1: Variance decomposition...")

fig, ax = plt.subplots(figsize=(6, 4))

variance_sources = ['Between-Seed\n(Data Splits)', 'Within-Seed\n(Training\nRandomness)', 'Total']
variances = [between_seed_std**2, within_seed_std**2, np.var(all_deltas, ddof=1)]
std_devs = [between_seed_std, within_seed_std, np.std(all_deltas, ddof=1)]

colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(variance_sources, std_devs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, std_devs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{val:.3f}\n({val*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Standard Deviation (ΔR²)', fontsize=11, fontweight='bold')
ax.set_title('Variance Decomposition: Sources of Uncertainty', fontsize=12, fontweight='bold', pad=15)
ax.set_ylim(0, max(std_devs) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='5% Threshold')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'figure1_variance_decomposition.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure1_variance_decomposition.pdf', bbox_inches='tight')
print(f"  ✓ Saved: figure1_variance_decomposition.png/pdf")
plt.close()

# ============================================================================
# Figure 2: Per-Seed Distributions (Violin Plot)
# ============================================================================

print("Figure 2: Per-seed distributions...")

fig, ax = plt.subplots(figsize=(8, 5))

# Prepare data for violin plot
plot_data = []
plot_labels = []
for seed in seeds:
    plot_data.append(seed_data[seed])
    plot_labels.append(f'Seed {seed}')

# Create violin plot
parts = ax.violinplot(plot_data, positions=range(len(seeds)), showmeans=True, showextrema=True)

# Customize violin colors
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Add individual points (jittered)
for i, (seed, deltas) in enumerate(seed_data.items()):
    x = np.random.normal(i, 0.04, size=len(deltas))
    ax.scatter(x, deltas, alpha=0.4, s=30, color='black', zorder=3)

# Add mean markers
seed_means = [np.mean(deltas) for deltas in seed_data.values()]
ax.scatter(range(len(seeds)), seed_means, color='red', s=100, marker='D',
           zorder=4, label='Mean', edgecolors='black', linewidths=1.5)

# Reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='5% Threshold')
ax.axhline(y=overall_mean, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall Mean ({overall_mean:.3f})')

ax.set_xticks(range(len(seeds)))
ax.set_xticklabels(plot_labels)
ax.set_ylabel('ΔR² (Geometry - Baseline)', fontsize=11, fontweight='bold')
ax.set_title('Per-Seed Performance Distribution (10 runs each)', fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'figure2_seed_distributions.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure2_seed_distributions.pdf', bbox_inches='tight')
print(f"  ✓ Saved: figure2_seed_distributions.png/pdf")
plt.close()

# ============================================================================
# Figure 3: Overall Distribution with 95% CI
# ============================================================================

print("Figure 3: Overall distribution with confidence interval...")

fig, ax = plt.subplots(figsize=(7, 5))

# Histogram
n, bins, patches = ax.hist(all_deltas, bins=15, alpha=0.6, color='steelblue',
                            edgecolor='black', linewidth=1.2, density=True)

# Fit normal distribution
mu, sigma = overall_mean, np.std(all_deltas, ddof=1)
x = np.linspace(all_deltas.min(), all_deltas.max(), 100)
y = stats.norm.pdf(x, mu, sigma)
ax.plot(x, y, 'r-', linewidth=2.5, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')

# Add vertical lines
ax.axvline(x=overall_mean, color='red', linestyle='-', linewidth=2.5, label=f'Mean: {overall_mean:.3f}')
ax.axvline(x=ci_low, color='orange', linestyle='--', linewidth=2, label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
ax.axvline(x=ci_high, color='orange', linestyle='--', linewidth=2)
ax.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='5% Threshold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Shade CI region
ax.axvspan(ci_low, ci_high, alpha=0.2, color='yellow', label='95% CI Region')

ax.set_xlabel('ΔR² (Geometry - Baseline)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title(f'Overall Performance Distribution (N=50 runs)', fontsize=12, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'figure3_overall_distribution.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure3_overall_distribution.pdf', bbox_inches='tight')
print(f"  ✓ Saved: figure3_overall_distribution.png/pdf")
plt.close()

# ============================================================================
# Figure 4: Timeline of All 50 Runs
# ============================================================================

print("Figure 4: Timeline of all runs...")

fig, ax = plt.subplots(figsize=(10, 5))

# Organize data by run order
run_order = []
run_deltas = []
run_seeds = []
run_colors = []

color_map = {seed: colors[i % len(colors)] for i, seed in enumerate(seeds)}

run_idx = 0
for seed_result in all_results:
    seed = seed_result['seed']
    for run_result in seed_result['runs']:
        run_order.append(run_idx)
        run_deltas.append(run_result['borderline']['delta_r2'])
        run_seeds.append(seed)
        run_colors.append(color_map[seed])
        run_idx += 1

# Scatter plot
ax.scatter(run_order, run_deltas, c=run_colors, s=80, alpha=0.7,
           edgecolors='black', linewidths=1, zorder=3)

# Connect runs from same seed
for seed in seeds:
    seed_indices = [i for i, s in enumerate(run_seeds) if s == seed]
    seed_deltas_ordered = [run_deltas[i] for i in seed_indices]
    ax.plot(seed_indices, seed_deltas_ordered, color=color_map[seed],
            alpha=0.3, linewidth=1.5, zorder=1)

# Reference lines
ax.axhline(y=overall_mean, color='purple', linestyle='-', linewidth=2.5,
           alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
ax.axhline(y=ci_low, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=ci_high, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, alpha=0.6, label='5% Threshold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.4)

# Shade CI band
ax.fill_between(range(len(run_order)), ci_low, ci_high, alpha=0.15, color='yellow')

# Add seed labels
for i, seed in enumerate(seeds):
    seed_indices = [idx for idx, s in enumerate(run_seeds) if s == seed]
    mid_idx = seed_indices[len(seed_indices)//2]
    ax.text(mid_idx, -0.09, f'Seed {seed}', ha='center', va='top',
            fontsize=8, fontweight='bold', color=color_map[seed],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Run Number', fontsize=11, fontweight='bold')
ax.set_ylabel('ΔR² (Geometry - Baseline)', fontsize=11, fontweight='bold')
ax.set_title('Timeline: All 50 Training Runs (10 per seed)', fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(-1, len(run_order))
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'figure4_all_runs_timeline.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure4_all_runs_timeline.pdf', bbox_inches='tight')
print(f"  ✓ Saved: figure4_all_runs_timeline.png/pdf")
plt.close()

# ============================================================================
# Summary Statistics Table
# ============================================================================

print("\nGenerating summary statistics table...")

table_text = f"""
PUBLICATION SUMMARY STATISTICS
{"="*60}

Overall Performance (N=50 runs):
  Mean ΔR²:        {overall_mean:+.4f} ({overall_mean*100:+.2f}%)
  Standard Dev:    {np.std(all_deltas, ddof=1):.4f}
  Standard Error:  {data['borderline_statistics']['se']:.4f}
  95% CI:          [{ci_low:+.4f}, {ci_high:+.4f}]

Variance Decomposition:
  Between-Seed SD: {between_seed_std:.4f} ({between_seed_std**2/np.var(all_deltas, ddof=1)*100:.1f}% of variance)
  Within-Seed SD:  {within_seed_std:.4f} ({within_seed_std**2/np.var(all_deltas, ddof=1)*100:.1f}% of variance)

Per-Seed Results (Mean ± SD):
"""

for seed in seeds:
    deltas = seed_data[seed]
    mean = np.mean(deltas)
    std = np.std(deltas, ddof=1)
    table_text += f"  Seed {seed:3d}: {mean:+.4f} ± {std:.4f} (range: [{min(deltas):+.4f}, {max(deltas):+.4f}])\n"

table_text += f"""
Statistical Tests:
  Pass Threshold:  5% improvement
  CI Lower Bound:  {ci_low*100:.2f}% ({"PASS" if ci_low > 0.05 else "FAIL"})
  Mean > 0:        {"Yes" if overall_mean > 0 else "No"}
  Mean > 5%:       {"Yes" if overall_mean > 0.05 else "No"}

Verdict: {data['verdict']}
"""

table_path = output_dir / 'summary_statistics.txt'
with open(table_path, 'w', encoding='utf-8') as f:
    f.write(table_text)

print(f"  ✓ Saved: summary_statistics.txt")
print("\n" + "="*60)
print("All figures generated successfully!")
print(f"Location: {output_dir}")
print("="*60)
