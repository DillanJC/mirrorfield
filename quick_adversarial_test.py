import sys

sys.path.insert(0, "C:/Users/User/mirrorfield")

import numpy as np
import time
from experiments.shared.adversarial_task_bank import get_adversarial_task_bank
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track5_recursive.recursive_learner import PolicyOptimizer

print("=== QUICK ADVERSARIAL DIAGNOSTIC (1 step each) ===")

# Initialize
print("\n[1/4] Initializing GLM client...")
client = GLMReasoningClient(max_tokens=2048)

print("[2/4] Initializing embedder...")
embedder = ResponseEmbedder()

print("[3/4] Building reference corpus...")
ref_texts = build_reference_texts()
reference = embedder.build_reference_corpus(ref_texts)
k = min(50, len(reference) - 1)

print("[4/4] Creating EnhancedTracer...")
optimizer = PolicyOptimizer(
    reference, k=k, use_enhanced_tracer=True, state_threshold_base=0.15
)
engine = optimizer.build_engine()

# Get 3 diverse adversarial tasks
all_tasks = get_adversarial_task_bank()
test_tasks = [
    all_tasks[0],  # hallucination
    all_tasks[4],  # framework_collision
    all_tasks[8],  # out_of_distribution
]

print(f"\n=== RUNNING {len(test_tasks)} TASKS (1 step each) ===")

all_signals = []
all_states = []
results_by_task = []

start_time = time.time()

for i, task in enumerate(test_tasks):
    print(f"\n--- Task {i + 1}: {task.id} ({task.adversarial_type}) ---")
    print(f"Prompt: {task.prompt[:60]}...")

    print(f"  Step 0...", end=" ", flush=True)
    step_start = time.time()

    step_text = client.generate_step(
        task_prompt=task.prompt,
        step_index=0,
        total_steps=1,
        previous_steps=None,
    )

    step_embedding = embedder.embed(step_text)
    norm = np.linalg.norm(step_embedding)
    step_embedding_norm = step_embedding / (norm + 1e-8)

    eval_result = engine.evaluate(step_embedding_norm, 0)

    sig = eval_result.step_trace.novelty_signature
    state = eval_result.step_trace.dominant_state
    all_signals.append(sig)
    all_states.append(state)

    step_time = time.time() - step_start
    triggered = "INTERVENTION" if eval_result.triggered else "ok"
    print(f"({step_time:.1f}s) sig={sig}, state={state}, {triggered}")

    results_by_task.append(
        {
            "id": task.id,
            "type": task.adversarial_type,
            "signal": sig,
            "state": state,
            "triggered": eval_result.triggered,
        }
    )

elapsed = time.time() - start_time

print("\n=== RESULTS ===")
print(f"Total steps: {len(all_signals)}")
print(f"Total time: {elapsed:.1f}s")
print(f"Interventions: {sum(1 for r in results_by_task if r['triggered'])}")
print(
    f"Signal distribution: {dict((s, all_signals.count(s)) for s in set(all_signals))}"
)
print(f"State distribution: {dict((s, all_states.count(s)) for s in set(all_states))}")
print(f"Unique signals: {len(set(all_signals))}")
print(f"Unique states: {len(set(all_states))}")

print("\n=== BY TASK TYPE ===")
for r in results_by_task:
    print(
        f"  {r['type']}: sig={r['signal']}, state={r['state']}, triggered={r['triggered']}"
    )

print("\n=== DONE ===")
