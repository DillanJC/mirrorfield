"""
Diagnostic: Why are all states classified as coherent?
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.task_bank import get_task_bank
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track5_recursive.recursive_learner import PolicyOptimizer


def patch_glm_client_for_debug():
    """Add verbose logging to GLM client and fix thinking mode."""
    from experiments.shared.llm_client import GLMReasoningClient

    def debug_call(self, system, messages):
        print(f"[GLM _call_api] Calling API with model={self.model}...")

        from openai import OpenAI

        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=oai_messages,
            extra_body={"thinking": {"type": "enabled"}},
        )

        content = resp.choices[0].message.content
        msg_extra = resp.choices[0].message.model_extra

        # FIX: When thinking mode is enabled, content is in reasoning_content
        if not content and msg_extra and "reasoning_content" in msg_extra:
            content = msg_extra["reasoning_content"]
            print(f"[GLM _call_api] Using reasoning_content: {repr(content)[:100]}...")
        else:
            print(
                f"[GLM _call_api] Raw content: {repr(content)[:100] if content else 'None/Empty'}"
            )

        return content

    GLMReasoningClient._call_api = debug_call


def main():
    patch_glm_client_for_debug()
    print("=" * 60)
    print("DIAGNOSTIC: State Classification Analysis")
    print("=" * 60)

    # Initialize
    client = GLMReasoningClient(max_tokens=512)
    embedder = ResponseEmbedder()

    print("\nBuilding reference corpus...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    k = min(50, len(reference) - 1)

    print(f"Reference shape: {reference.shape}")

    # Initialize optimizer with EnhancedTracer
    optimizer = PolicyOptimizer(
        reference,
        k=k,
        use_enhanced_tracer=True,
    )

    # Get the enhanced tracer for direct inspection
    tracer = optimizer._enhanced_tracer
    print(f"\nEnhancedTracer config:")
    print(f"  PR baseline mean: {tracer._pr_baseline_mean:.3f}")
    print(f"  PR p20: {tracer._ref_percentiles.get('participation_ratio_p20', 'N/A')}")
    print(f"  PR p30: {tracer._ref_percentiles.get('participation_ratio_p30', 'N/A')}")
    print(f"  PR p50: {tracer._ref_percentiles.get('participation_ratio_p50', 'N/A')}")
    print(f"  PR p70: {tracer._ref_percentiles.get('participation_ratio_p70', 'N/A')}")

    # Run ONE task with detailed logging
    tasks = get_task_bank()
    task = tasks[0]

    print(f"\n{'=' * 60}")
    print(f"Running task: {task.id}")
    print(f"Domain: {task.domain}")
    print(f"Prompt: {task.prompt[:100]}...")
    print(f"{'=' * 60}")

    step_texts = []
    for step_idx in range(4):
        print(f"\n[DEBUG] Calling generate_step for step {step_idx}...")
        step_text = client.generate_step(
            task_prompt=task.prompt,
            step_index=step_idx,
            total_steps=4,
            previous_steps=step_texts if step_texts else None,
        )
        print(f"[DEBUG] Got step_text: len={len(step_text)}")
        step_texts.append(step_text)

        print(f"  [text len]: {len(step_text)}")
        print(f"  [text preview]: {repr(step_text[:200])}")

        step_embedding = embedder.embed(step_text)
        norm = np.linalg.norm(step_embedding)
        step_embedding_norm = step_embedding / (norm + 1e-8)

        print(f"  [text]: {step_text[:100]}...")
        print(f"  [emb hash]: {hash(step_embedding_norm.tobytes()) % 1000000}")

        # Get trace directly from tracer
        trace = tracer.trace_step(step_embedding_norm, step_idx)

        print(f"\n--- Step {step_idx} ---")
        print(f"State scores:")
        for state, score in sorted(trace.state_scores.items(), key=lambda x: -x[1]):
            marker = " <-- WINNER" if state == trace.dominant_state else ""
            print(f"  {state}: {score:.3f}{marker}")

        print(f"\nTopology features:")
        print(f"  PR: {trace.topology_features['participation_ratio']:.3f}")
        print(f"  SE: {trace.topology_features['spectral_entropy']:.3f}")
        print(f"  d_eff: {trace.topology_features['d_eff']:.3f}")

        print(f"\nOther features:")
        g_mag = trace.all_features.get("local_gradient_magnitude", None)
        turb = trace.all_features.get("turbulence_index", None)
        therm = trace.all_features.get("thermal_gradient", None)
        knn_std = trace.features.get("knn_std_distance", None)
        ridge = trace.features.get("ridge_proximity", None)
        print(f"  g_mag: {g_mag:.3f}" if g_mag else f"  g_mag: N/A")
        print(f"  turbulence: {turb:.3f}" if turb else f"  turbulence: N/A")
        print(
            f"  thermal_gradient: {therm:.3f}" if therm else f"  thermal_gradient: N/A"
        )
        print(f"  knn_std: {knn_std:.3f}" if knn_std else f"  knn_std: N/A")
        print(f"  ridge_prox: {ridge:.3f}" if ridge else f"  ridge_prox: N/A")

        print(f"\nResult:")
        print(f"  dominant_state: {trace.dominant_state}")
        print(f"  signal: {trace.novelty_signature}")
        print(f"  flags: {trace.flags}")
        print(f"  g_ratio: {trace.g_ratio:.3f}")


if __name__ == "__main__":
    main()
