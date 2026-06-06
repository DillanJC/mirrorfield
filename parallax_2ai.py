"""
Parallax Pipeline - Claude + OpenAI (2-AI Variant)
Runs when Gemini is unavailable or rate-limited.
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StepMetrics:
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        if "claude" in self.model.lower():
            return (self.input_tokens * 0.003 + self.output_tokens * 0.015) / 1000
        elif "gpt" in self.model.lower():
            return (self.input_tokens * 0.005 + self.output_tokens * 0.015) / 1000
        return 0


class Parallax2AI:
    """Claude + OpenAI pipeline."""

    def __init__(self):
        import anthropic
        import openai

        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.claude_model = "claude-sonnet-4-20250514"
        self.openai_model = "gpt-4o"

    def _call_claude(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        response = self.claude.messages.create(
            model=self.claude_model,
            max_tokens=2000,
            system=system or "You provide thoughtful, analytical perspectives.",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.content[0].text
        return text, StepMetrics(
            model=self.claude_model,
            latency_ms=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _call_openai(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        response = self.openai.chat.completions.create(
            model=self.openai_model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system or "You provide thoughtful, analytical perspectives."},
                {"role": "user", "content": prompt},
            ],
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content
        return text, StepMetrics(
            model=self.openai_model,
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def run_sequential(self, query: str, verbose: bool = False) -> dict:
        """Claude first, then OpenAI with context, then Claude synthesizes."""
        results = {"query": query, "responses": {}, "metrics": []}

        # Step 1: Claude initial
        if verbose:
            print("[1/3] Claude (initial perspective)...")
        claude_resp, claude_m = self._call_claude(query)
        results["responses"]["claude"] = claude_resp
        results["metrics"].append(claude_m)
        if verbose:
            print(f"      {claude_m.latency_ms:.0f}ms, {claude_m.total_tokens} tokens")

        # Step 2: OpenAI with Claude's context
        if verbose:
            print("[2/3] OpenAI (responding to Claude)...")
        openai_prompt = f"""Original query: {query}

Claude's perspective:
{claude_resp}

---

Provide YOUR perspective. Where do you agree? Disagree? What would you add or challenge?"""

        openai_resp, openai_m = self._call_openai(openai_prompt)
        results["responses"]["openai"] = openai_resp
        results["metrics"].append(openai_m)
        if verbose:
            print(f"      {openai_m.latency_ms:.0f}ms, {openai_m.total_tokens} tokens")

        # Step 3: Claude synthesizes
        if verbose:
            print("[3/3] Claude (synthesis)...")
        synth_prompt = f"""Original query: {query}

CLAUDE'S INITIAL PERSPECTIVE:
{claude_resp}

OPENAI'S RESPONSE:
{openai_resp}

---

Synthesize these two perspectives:
1. CONVERGENCE: Where do both agree?
2. TENSION: Where do they differ?
3. SYNTHESIS: What integrated insight emerges?"""

        synth_resp, synth_m = self._call_claude(synth_prompt,
            system="You synthesize multiple AI perspectives while preserving distinct viewpoints.")
        results["synthesis"] = synth_resp
        results["metrics"].append(synth_m)
        if verbose:
            print(f"      {synth_m.latency_ms:.0f}ms, {synth_m.total_tokens} tokens")

        return results

    def run_parallel(self, query: str, verbose: bool = False) -> dict:
        """Both respond independently, then synthesize."""
        results = {"query": query, "responses": {}, "metrics": []}

        if verbose:
            print("[1/2] Claude + OpenAI in parallel...")

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            claude_future = executor.submit(self._call_claude, query)
            openai_future = executor.submit(self._call_openai, query)

            claude_resp, claude_m = claude_future.result()
            openai_resp, openai_m = openai_future.result()

        wall_clock = (time.perf_counter() - start) * 1000

        results["responses"]["claude"] = claude_resp
        results["responses"]["openai"] = openai_resp
        results["metrics"].extend([claude_m, openai_m])
        results["parallel_wall_clock_ms"] = wall_clock

        if verbose:
            print(f"      Claude: {claude_m.latency_ms:.0f}ms, OpenAI: {openai_m.latency_ms:.0f}ms")
            print(f"      Wall-clock: {wall_clock:.0f}ms")

        # Synthesis
        if verbose:
            print("[2/2] Claude (synthesis)...")

        synth_prompt = f"""Original query: {query}

CLAUDE'S PERSPECTIVE:
{claude_resp}

OPENAI'S PERSPECTIVE:
{openai_resp}

---

Synthesize these two perspectives:
1. CONVERGENCE: Where do both agree?
2. TENSION: Where do they differ?
3. SYNTHESIS: What integrated insight emerges?"""

        synth_resp, synth_m = self._call_claude(synth_prompt,
            system="You synthesize multiple AI perspectives while preserving distinct viewpoints.")
        results["synthesis"] = synth_resp
        results["metrics"].append(synth_m)
        if verbose:
            print(f"      {synth_m.latency_ms:.0f}ms, {synth_m.total_tokens} tokens")

        return results


def format_output(results: dict) -> str:
    out = []
    out.append("=" * 70)
    out.append("PARALLAX TRIANGULATION (Claude + OpenAI)")
    out.append("=" * 70)
    out.append(f"\nQUERY: {results['query']}\n")

    for name in ["claude", "openai"]:
        if name in results["responses"]:
            out.append("-" * 70)
            out.append(f"{name.upper()}:")
            out.append("-" * 70)
            text = results["responses"][name]
            out.append(text[:1500] + ("..." if len(text) > 1500 else ""))
            out.append("")

    out.append("=" * 70)
    out.append("SYNTHESIS:")
    out.append("=" * 70)
    out.append(results.get("synthesis", "N/A"))

    out.append("\n" + "=" * 70)
    out.append("METRICS:")
    out.append("=" * 70)

    metrics = results["metrics"]
    total_latency = sum(m.latency_ms for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)
    total_cost = sum(m.cost for m in metrics)

    if "parallel_wall_clock_ms" in results:
        synth_time = metrics[-1].latency_ms
        out.append(f"Wall-clock: {results['parallel_wall_clock_ms'] + synth_time:.0f}ms (parallel)")
    else:
        out.append(f"Total latency: {total_latency:.0f}ms ({total_latency/1000:.1f}s)")

    out.append(f"Total tokens: {total_tokens:,}")
    out.append(f"Estimated cost: ${total_cost:.4f}")

    out.append("\nPer-step:")
    labels = ["Claude", "OpenAI", "Synthesis"]
    for i, m in enumerate(metrics):
        out.append(f"  {labels[i]}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens, ${m.cost:.4f}")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Parallax Pipeline - Claude + OpenAI")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", help="Save JSON output")
    parser.add_argument("--test", action="store_true", help="Run test query")

    args = parser.parse_args()

    if args.test:
        args.query = "What are the key considerations when designing systems that prevent exploitation structurally rather than through rules?"
        args.verbose = True

    if not args.query:
        parser.print_help()
        sys.exit(1)

    pipeline = Parallax2AI()

    print(f"\nMode: {'Parallel' if args.parallel else 'Sequential'}\n")

    if args.parallel:
        results = pipeline.run_parallel(args.query, verbose=args.verbose)
    else:
        results = pipeline.run_sequential(args.query, verbose=args.verbose)

    print("\n" + format_output(results))

    if args.output:
        output = {
            "query": results["query"],
            "responses": results["responses"],
            "synthesis": results.get("synthesis"),
            "metrics": [
                {"model": m.model, "latency_ms": m.latency_ms,
                 "tokens": m.total_tokens, "cost": m.cost}
                for m in results["metrics"]
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
