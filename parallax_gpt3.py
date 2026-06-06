"""
Parallax Pipeline - 3x GPT Variant
Uses three GPT instances with distinct analytical personas.
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

@dataclass
class StepMetrics:
    instance: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        # GPT-4o pricing
        return (self.input_tokens * 0.005 + self.output_tokens * 0.015) / 1000


# Three distinct analytical lenses
INSTANCES = {
    "alpha": {
        "name": "Alpha (Analytical)",
        "system": """You are ALPHA, a rigorous analytical thinker.

Your cognitive style:
- Systematic decomposition of problems
- Evidence-based reasoning
- Identify assumptions, dependencies, edge cases
- Formal logic and structured argumentation
- Question unstated premises

You speak precisely. You distrust intuition without evidence. You seek falsifiability.""",
    },

    "beta": {
        "name": "Beta (Adversarial)",
        "system": """You are BETA, an adversarial thinker and devil's advocate.

Your cognitive style:
- Actively seek flaws in reasoning
- Consider how systems can be gamed or exploited
- Identify failure modes and perverse incentives
- Challenge consensus views
- Think like an attacker, regulator, or skeptic

You are constructively contrarian. If everyone agrees, you find the crack.""",
    },

    "gamma": {
        "name": "Gamma (Integrative)",
        "system": """You are GAMMA, an integrative systems thinker.

Your cognitive style:
- See connections across domains
- Consider second and third-order effects
- Draw on historical patterns and analogies
- Balance idealism with pragmatism
- Synthesize competing perspectives

You find the throughline. You see the forest and the trees.""",
    },
}


class GPT3Pipeline:
    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _call(self, instance: str, prompt: str, context: str = "") -> tuple[str, StepMetrics]:
        """Call GPT with a specific instance persona."""
        config = INSTANCES[instance]

        if context:
            full_prompt = f"{context}\n\n---\n\nNow provide YOUR perspective:\n{prompt}"
        else:
            full_prompt = prompt

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": full_prompt},
            ],
        )
        latency = (time.perf_counter() - start) * 1000

        return response.choices[0].message.content, StepMetrics(
            instance=config["name"],
            model=self.model,
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def run_sequential(self, query: str, verbose: bool = False) -> dict:
        """Run instances sequentially, each building on previous context."""
        results = {"query": query, "responses": {}, "metrics": []}
        accumulated = f"ORIGINAL QUERY: {query}\n\n"

        for i, instance in enumerate(["alpha", "beta", "gamma"], 1):
            name = INSTANCES[instance]["name"]
            if verbose:
                print(f"[{i}/4] {name}...")

            response, metrics = self._call(instance, query, accumulated if i > 1 else "")
            results["responses"][instance] = response
            results["metrics"].append(metrics)
            accumulated += f"--- {name} ---\n{response}\n\n"

            if verbose:
                print(f"      {metrics.latency_ms:.0f}ms, {metrics.total_tokens} tokens")

        # Synthesis
        if verbose:
            print("[4/4] Synthesizing...")

        synth_prompt = f"""{accumulated}
---

TASK: Synthesize these three perspectives.

Structure your synthesis as:
1. CONVERGENCE: Where do all three agree?
2. TENSION: Where do they disagree or see different risks?
3. BLIND SPOTS: What might all three be missing?
4. TRIANGULATED INSIGHT: What emerges from combining these views?"""

        synth_response, synth_metrics = self._call("gamma", synth_prompt)
        results["synthesis"] = synth_response
        results["metrics"].append(StepMetrics(
            instance="Synthesis",
            model=self.model,
            latency_ms=synth_metrics.latency_ms,
            input_tokens=synth_metrics.input_tokens,
            output_tokens=synth_metrics.output_tokens,
        ))

        if verbose:
            print(f"      {synth_metrics.latency_ms:.0f}ms, {synth_metrics.total_tokens} tokens")

        return results

    def run_parallel(self, query: str, verbose: bool = False) -> dict:
        """Run all three instances in parallel, then synthesize."""
        results = {"query": query, "responses": {}, "metrics": []}

        if verbose:
            print("[1/2] Running 3 instances in parallel...")

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._call, inst, query): inst
                for inst in ["alpha", "beta", "gamma"]
            }

            for future in as_completed(futures):
                instance = futures[future]
                response, metrics = future.result()
                results["responses"][instance] = response
                results["metrics"].append(metrics)
                if verbose:
                    print(f"      {INSTANCES[instance]['name']}: {metrics.latency_ms:.0f}ms")

        parallel_time = (time.perf_counter() - start) * 1000
        if verbose:
            print(f"      Parallel wall-clock: {parallel_time:.0f}ms")

        # Build context for synthesis
        accumulated = f"ORIGINAL QUERY: {query}\n\n"
        for inst in ["alpha", "beta", "gamma"]:
            accumulated += f"--- {INSTANCES[inst]['name']} ---\n{results['responses'][inst]}\n\n"

        # Synthesis
        if verbose:
            print("[2/2] Synthesizing...")

        synth_prompt = f"""{accumulated}
---

TASK: Synthesize these three perspectives.

Structure your synthesis as:
1. CONVERGENCE: Where do all three agree?
2. TENSION: Where do they disagree or see different risks?
3. BLIND SPOTS: What might all three be missing?
4. TRIANGULATED INSIGHT: What emerges from combining these views?"""

        synth_response, synth_metrics = self._call("gamma", synth_prompt)
        results["synthesis"] = synth_response
        results["metrics"].append(StepMetrics(
            instance="Synthesis",
            model=self.model,
            latency_ms=synth_metrics.latency_ms,
            input_tokens=synth_metrics.input_tokens,
            output_tokens=synth_metrics.output_tokens,
        ))

        if verbose:
            print(f"      {synth_metrics.latency_ms:.0f}ms, {synth_metrics.total_tokens} tokens")

        # Store timing info
        results["parallel_wall_clock_ms"] = parallel_time

        return results


def format_output(results: dict) -> str:
    """Format results for terminal."""
    out = []
    out.append("=" * 70)
    out.append("PARALLAX TRIANGULATION (3x GPT)")
    out.append("=" * 70)
    out.append(f"\nQUERY: {results['query']}\n")

    for inst in ["alpha", "beta", "gamma"]:
        if inst in results["responses"]:
            out.append("-" * 70)
            out.append(f"{INSTANCES[inst]['name'].upper()}:")
            out.append("-" * 70)
            text = results["responses"][inst]
            out.append(text[:1500] + ("..." if len(text) > 1500 else ""))
            out.append("")

    out.append("=" * 70)
    out.append("SYNTHESIS:")
    out.append("=" * 70)
    out.append(results.get("synthesis", "N/A"))

    # Metrics
    out.append("\n" + "=" * 70)
    out.append("METRICS:")
    out.append("=" * 70)

    metrics = results["metrics"]
    total_latency = sum(m.latency_ms for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)
    total_cost = sum(m.cost for m in metrics)

    if "parallel_wall_clock_ms" in results:
        synth_time = metrics[-1].latency_ms if metrics else 0
        out.append(f"Wall-clock time: {results['parallel_wall_clock_ms'] + synth_time:.0f}ms (parallel mode)")
        out.append(f"Sum of API times: {total_latency:.0f}ms")
    else:
        out.append(f"Total latency: {total_latency:.0f}ms ({total_latency/1000:.1f}s)")

    out.append(f"Total tokens: {total_tokens:,}")
    out.append(f"Estimated cost: ${total_cost:.4f}")

    out.append("\nPer-instance:")
    for m in metrics:
        out.append(f"  {m.instance}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens, ${m.cost:.4f}")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Parallax Pipeline - 3x GPT")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run instances in parallel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", help="Save JSON output to file")
    parser.add_argument("--model", default="gpt-4o", help="GPT model to use")
    parser.add_argument("--test", action="store_true", help="Run with test query")

    args = parser.parse_args()

    if args.test:
        args.query = "What are the key considerations when designing systems that prevent exploitation structurally rather than through rules?"
        args.verbose = True

    if not args.query:
        parser.print_help()
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    pipeline = GPT3Pipeline(model=args.model)

    print(f"\nMode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"Model: {args.model}\n")

    if args.parallel:
        results = pipeline.run_parallel(args.query, verbose=args.verbose)
    else:
        results = pipeline.run_sequential(args.query, verbose=args.verbose)

    print("\n" + format_output(results))

    if args.output:
        # Convert metrics for JSON serialization
        output = {
            "query": results["query"],
            "responses": results["responses"],
            "synthesis": results.get("synthesis"),
            "metrics": [
                {"instance": m.instance, "model": m.model,
                 "latency_ms": m.latency_ms, "tokens": m.total_tokens, "cost": m.cost}
                for m in results["metrics"]
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
