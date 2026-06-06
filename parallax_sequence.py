"""
Parallax Sequence Experimenter
Test different orderings of AI models to find optimal sequence.
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StepMetrics:
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class SequenceExperimenter:
    """Test different AI orderings."""

    # Model "power" rankings (subjective, adjustable)
    MODELS = {
        "claude": {
            "name": "Claude",
            "model": "claude-sonnet-4-20250514",
            "tier": 2,  # 1=highest, 3=lowest
        },
        "gpt": {
            "name": "GPT-4o",
            "model": "gpt-4o",
            "tier": 1,
        },
        "gemini": {
            "name": "Gemini",
            "model": "gemini-2.0-flash",
            "tier": 3,
        },
        "grok": {
            "name": "Grok",
            "model": "grok-2",
            "tier": 1,
        },
    }

    # Pre-defined sequences to test
    SEQUENCES = {
        "default": [
            "gemini",
            "claude",
            "gpt",
            "claude",
        ],  # Optimal: weak start, Claude synth
        "optimal": ["gemini", "claude", "gpt", "claude"],  # Same as default (alias)
        "weak_to_strong": [
            "gemini",
            "claude",
            "gpt",
            "gpt",
        ],  # Weakest first, strongest synthesizes
        "strong_to_weak": ["gpt", "claude", "gemini", "gpt"],  # Strongest sets frame
        "strong_middle": ["gemini", "gpt", "claude", "claude"],  # Strongest in middle
        "bookend": ["gpt", "gemini", "claude", "gpt"],  # Strong-weak-mid-strong
        "gemini_synth": [
            "claude",
            "gpt",
            "gemini",
            "gemini",
        ],  # Gemini synthesizes (cheap)
        "all_claude": [
            "claude",
            "claude",
            "claude",
            "claude",
        ],  # Baseline: single model
        "all_gpt": ["gpt", "gpt", "gpt", "gpt"],  # Baseline: single model
    }

    def __init__(self):
        import anthropic
        import openai
        import google.generativeai as genai

        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.grok_client = openai.OpenAI(
            base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY")
        )

        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.gemini_client = genai.GenerativeModel("gemini-2.0-flash")

    def _call(
        self, model_key: str, prompt: str, system: str = ""
    ) -> tuple[str, StepMetrics]:
        """Call any model by key."""
        config = self.MODELS[model_key]

        if model_key == "claude":
            return self._call_claude(prompt, system)
        elif model_key == "gpt":
            return self._call_gpt(prompt, system)
        elif model_key == "gemini":
            return self._call_gemini(prompt, system)
        elif model_key == "grok":
            return self._call_grok(prompt, system)

    def _call_claude(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        response = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system or "Provide your unique analytical perspective.",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000
        return response.content[0].text, StepMetrics(
            model="Claude",
            latency_ms=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _call_gpt(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": system or "Provide your unique analytical perspective.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        latency = (time.perf_counter() - start) * 1000
        return response.choices[0].message.content, StepMetrics(
            model="GPT-4o",
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def _call_gemini(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.gemini_client.generate_content(full_prompt)
        latency = (time.perf_counter() - start) * 1000

        try:
            in_tokens = self.gemini_client.count_tokens(full_prompt).total_tokens
            out_tokens = self.gemini_client.count_tokens(response.text).total_tokens
        except:
            in_tokens = len(full_prompt) // 4
            out_tokens = len(response.text) // 4

        return response.text, StepMetrics(
            model="Gemini",
            latency_ms=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    def _call_grok(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        start = time.perf_counter()
        response = self.grok_client.chat.completions.create(
            model="grok-4",
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": system or "Provide your unique analytical perspective.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        latency = (time.perf_counter() - start) * 1000
        return response.choices[0].message.content, StepMetrics(
            model="Grok",
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def run_sequence(
        self, query: str, sequence: list[str], verbose: bool = False
    ) -> dict:
        """
        Run a specific sequence.
        sequence = [responder1, responder2, responder3, synthesizer]
        """
        results = {
            "query": query,
            "sequence": sequence,
            "sequence_display": " ->".join(self.MODELS[s]["name"] for s in sequence),
            "responses": [],
            "metrics": [],
        }

        accumulated = f"QUERY: {query}\n\n"

        # First 3 are responders
        for i, model_key in enumerate(sequence[:3], 1):
            name = self.MODELS[model_key]["name"]
            if verbose:
                print(f"[{i}/4] {name}...")

            if i == 1:
                prompt = query
            else:
                prompt = f"{accumulated}---\n\nProvide YOUR distinct perspective. Build on, challenge, or extend what came before."

            response, metrics = self._call(model_key, prompt)
            results["responses"].append({"model": name, "response": response})
            results["metrics"].append(metrics)
            accumulated += f"--- {name}'s Perspective ---\n{response}\n\n"

            if verbose:
                print(
                    f"      {metrics.latency_ms:.0f}ms, {metrics.total_tokens} tokens"
                )

        # 4th is synthesizer
        synth_key = sequence[3]
        synth_name = self.MODELS[synth_key]["name"]
        if verbose:
            print(f"[4/4] {synth_name} (synthesis)...")

        synth_prompt = f"""{accumulated}---

TASK: Synthesize these three perspectives.
1. CONVERGENCE: Where do they agree?
2. TENSION: Where do they disagree?
3. UNIQUE INSIGHTS: What did each contribute that others missed?
4. TRIANGULATED CONCLUSION: What emerges from combining these views?"""

        response, metrics = self._call(
            synth_key,
            synth_prompt,
            "You synthesize multiple AI perspectives while preserving distinct viewpoints.",
        )
        results["synthesis"] = response
        results["metrics"].append(metrics)

        if verbose:
            print(f"      {metrics.latency_ms:.0f}ms, {metrics.total_tokens} tokens")

        # Summary stats
        results["total_latency_ms"] = sum(m.latency_ms for m in results["metrics"])
        results["total_tokens"] = sum(m.total_tokens for m in results["metrics"])

        return results

    def compare_sequences(
        self, query: str, sequence_names: list[str] = None, verbose: bool = False
    ) -> dict:
        """Run multiple sequences and compare."""
        if sequence_names is None:
            sequence_names = list(self.SEQUENCES.keys())

        comparison = {
            "query": query,
            "results": {},
        }

        for name in sequence_names:
            if name not in self.SEQUENCES:
                print(f"Unknown sequence: {name}")
                continue

            print(f"\n{'=' * 60}")
            print(f"SEQUENCE: {name}")
            print(f"{'=' * 60}")

            sequence = self.SEQUENCES[name]
            result = self.run_sequence(query, sequence, verbose=verbose)
            comparison["results"][name] = result

            print(
                f"\nLatency: {result['total_latency_ms']:.0f}ms | Tokens: {result['total_tokens']}"
            )

        return comparison


def format_result(result: dict) -> str:
    """Format single sequence result."""
    out = []
    out.append("=" * 70)
    out.append(f"SEQUENCE: {result['sequence_display']}")
    out.append("=" * 70)
    out.append(f"\nQUERY: {result['query']}\n")

    for resp in result["responses"]:
        out.append("-" * 70)
        out.append(f"{resp['model']}:")
        out.append("-" * 70)
        text = resp["response"]
        out.append(text[:1200] + ("..." if len(text) > 1200 else ""))
        out.append("")

    out.append("=" * 70)
    out.append("SYNTHESIS:")
    out.append("=" * 70)
    out.append(result.get("synthesis", "N/A"))

    out.append(
        f"\n\nTotal: {result['total_latency_ms']:.0f}ms, {result['total_tokens']} tokens"
    )

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Parallax Sequence Experimenter")
    parser.add_argument("query", nargs="?", help="Query to test")
    parser.add_argument(
        "-s",
        "--sequence",
        help="Sequence name or custom (e.g., 'gpt,claude,gemini,claude')",
    )
    parser.add_argument("-c", "--compare", nargs="*", help="Compare multiple sequences")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", help="Save results to JSON")
    parser.add_argument("--list", action="store_true", help="List available sequences")
    parser.add_argument("--test", action="store_true", help="Run test query")

    args = parser.parse_args()

    exp = SequenceExperimenter()

    if args.list:
        print("Available sequences:")
        for name, seq in exp.SEQUENCES.items():
            display = " -> ".join(exp.MODELS[s]["name"] for s in seq)
            print(f"  {name:20} {display}")
        return

    if args.test:
        args.query = "What matters more: being right or being kind?"
        args.verbose = True

    if not args.query:
        parser.print_help()
        return

    if args.compare is not None:
        # Compare mode
        seqs = args.compare if args.compare else None
        results = exp.compare_sequences(args.query, seqs, verbose=args.verbose)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nSaved to: {args.output}")
    else:
        # Single sequence mode
        if args.sequence:
            if args.sequence in exp.SEQUENCES:
                sequence = exp.SEQUENCES[args.sequence]
            else:
                # Custom sequence
                sequence = args.sequence.split(",")
        else:
            sequence = exp.SEQUENCES["default"]

        result = exp.run_sequence(args.query, sequence, verbose=args.verbose)
        print("\n" + format_result(result))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
