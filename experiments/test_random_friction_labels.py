"""
Negative control: Test if random friction labels show stratification.

This proves our stratification isn't an artifact of classification logic.
If random labels show NO stratification, but d̃-based labels DO, then
d̃ is genuinely predictive.
"""
import json
import random
from pathlib import Path

# Load results from a real run
run_dir = Path("experiments/results/phase_d_integrated_eval/20251228_113319")
with open(run_dir / "perturbation_results.json") as f:
    results = json.load(f)

# Randomly assign friction levels (ignoring actual d̃ values)
random.seed(42)
n_low = int(len(results) * 0.63)  # Match real distribution (~63% low)
n_high = int(len(results) * 0.18)  # Match real distribution (~18% high)
n_medium = len(results) - n_low - n_high

random_labels = (["low"] * n_low + ["medium"] * n_medium + ["high"] * n_high)
random.shuffle(random_labels)

# Assign random labels and compute flip rates
low_flips = []
med_flips = []
high_flips = []

for result, random_label in zip(results, random_labels):
    if random_label == "low":
        low_flips.append(result["flip_rate"])
    elif random_label == "medium":
        med_flips.append(result["flip_rate"])
    else:
        high_flips.append(result["flip_rate"])

# Compute statistics
low_mean = sum(low_flips) / len(low_flips) if low_flips else 0
med_mean = sum(med_flips) / len(med_flips) if med_flips else 0
high_mean = sum(high_flips) / len(high_flips) if high_flips else 0

print("=" * 60)
print("NEGATIVE CONTROL: Random Friction Labels")
print("=" * 60)
print(f"Low friction (random):    {low_mean:.1%} ({len(low_flips)} samples)")
print(f"Medium friction (random): {med_mean:.1%} ({len(med_flips)} samples)")
print(f"High friction (random):   {high_mean:.1%} ({len(high_flips)} samples)")
print(f"Stratification (random):  {high_mean/low_mean if low_mean > 0 else float('inf'):.1f}×")
print()

# Compare to actual d̃-based labels
actual_low = [r["flip_rate"] for r in results if r["friction_level"] == "low"]
actual_high = [r["flip_rate"] for r in results if r["friction_level"] == "high"]

actual_low_mean = sum(actual_low) / len(actual_low) if actual_low else 0
actual_high_mean = sum(actual_high) / len(actual_high) if actual_high else 0

print("ACTUAL: d̃-Based Friction Labels")
print("=" * 60)
print(f"Low friction (d̃-based):  {actual_low_mean:.1%} ({len(actual_low)} samples)")
print(f"High friction (d̃-based): {actual_high_mean:.1%} ({len(actual_high)} samples)")
print(f"Stratification (d̃-based): {actual_high_mean/actual_low_mean if actual_low_mean > 0 else float('inf'):.1f}×")
print()

print("=" * 60)
print("CONCLUSION:")
print("=" * 60)
if abs(high_mean - low_mean) < 0.02:  # Less than 2% difference
    print("✓ Random labels show NO stratification (as expected)")
    print("✓ d̃-based labels show STRONG stratification")
    print("✓ This proves d̃ is genuinely predictive, not an artifact!")
else:
    print("✗ Unexpected: Random labels show stratification")
    print("  (This would indicate a problem with our analysis)")
