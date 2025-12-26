# Mirrorfield Experiments

## GPU Playground
1. Activate the virtual environment: `.\.venv\Scripts\Activate.ps1`.
2. Run the training harness: `python experiments/gpu_playground.py`.
3. You will see the selected device (CPU or GPU name), loss logs every ~20 steps, and a final summary showing the last loss value plus total wall-clock time.

## On-Device Jitter Harness v0.1
This harness trains the 4×4 grid GNN in `experiments/jitter_graph_playground.py`, then perturbs node features to measure how logits shift under noise. It captures the full setup (noise scale, samples, top-k extremes), records per-node variances, and times both the training and jitter phases so you can cite concrete throughput numbers in writeups.

After each run it emits `experiments/jitter_graph_results_run01.json`, which powers downstream analysis scripts and grants-ready summaries. Pair it with `python experiments/analyze_jitter_results.py` to print statistical snapshots and (optionally) plot variance histograms—handy evidence that the on-device lab is tracing stability as well as speed.
