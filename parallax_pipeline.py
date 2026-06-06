"""
Parallax Triangulation Pipeline
Automated multi-AI sequential chain for synthesizing perspectives.

Chains: Claude -> OpenAI -> Gemini -> Synthesis
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class StepMetrics:
    """Metrics for a single API call step."""
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_length: int

    @property
    def estimated_cost(self) -> float:
        """Rough cost estimation in USD."""
        # Approximate pricing (varies by model)
        costs = {
            "claude": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "gpt": {"input": 0.005, "output": 0.015},
            "gemini": {"input": 0.00035, "output": 0.00105},
        }
        model_type = self.model.split("-")[0].lower()
        if "claude" in self.model.lower():
            model_type = "claude"
        elif "gpt" in self.model.lower():
            model_type = "gpt"
        elif "gemini" in self.model.lower():
            model_type = "gemini"

        rates = costs.get(model_type, {"input": 0.01, "output": 0.03})
        return (self.input_tokens * rates["input"] +
                self.output_tokens * rates["output"]) / 1000


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    query: str
    claude_response: str
    openai_response: str
    gemini_response: str
    synthesis: str
    synthesis_method: str
    metrics: list[StepMetrics] = field(default_factory=list)
    total_latency_ms: float = 0
    total_tokens: int = 0
    context_pressure: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "responses": {
                "claude": self.claude_response,
                "openai": self.openai_response,
                "gemini": self.gemini_response,
            },
            "synthesis": self.synthesis,
            "synthesis_method": self.synthesis_method,
            "metrics": {
                "total_latency_ms": self.total_latency_ms,
                "total_tokens": self.total_tokens,
                "per_step": [
                    {
                        "model": m.model,
                        "latency_ms": m.latency_ms,
                        "tokens": {"in": m.input_tokens, "out": m.output_tokens},
                        "cost_usd": round(m.estimated_cost, 4),
                    }
                    for m in self.metrics
                ],
                "context_pressure": self.context_pressure,
            },
        }


class ParallaxPipeline:
    """Multi-AI sequential chain pipeline."""

    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        claude_model: str = "claude-sonnet-4-20250514",
        openai_model: str = "gpt-4o",
        gemini_model: str = "gemini-2.0-flash",
    ):
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self.claude_model = claude_model
        self.openai_model = openai_model
        self.gemini_model = gemini_model

        # Context window limits (approximate)
        self.context_limits = {
            "claude": 200000,
            "openai": 128000,
            "gemini": 1000000,
        }

        self._init_clients()

    def _init_clients(self):
        """Initialize API clients."""
        self.clients = {}

        if self.claude_api_key:
            import anthropic
            self.clients["claude"] = anthropic.Anthropic(api_key=self.claude_api_key)

        if self.openai_api_key:
            import openai
            self.clients["openai"] = openai.OpenAI(api_key=self.openai_api_key)

        if self.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self.clients["gemini"] = genai.GenerativeModel(self.gemini_model)

    def _call_claude(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        """Call Claude API and return response with metrics."""
        if "claude" not in self.clients:
            raise ValueError("Claude API key not configured")

        start = time.perf_counter()

        response = self.clients["claude"].messages.create(
            model=self.claude_model,
            max_tokens=4096,
            system=system or "You are a thoughtful analyst providing your unique perspective.",
            messages=[{"role": "user", "content": prompt}],
        )

        latency = (time.perf_counter() - start) * 1000

        text = response.content[0].text
        metrics = StepMetrics(
            model=self.claude_model,
            latency_ms=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            response_length=len(text),
        )

        return text, metrics

    def _call_openai(self, prompt: str, system: str = "") -> tuple[str, StepMetrics]:
        """Call OpenAI API and return response with metrics."""
        if "openai" not in self.clients:
            raise ValueError("OpenAI API key not configured")

        start = time.perf_counter()

        response = self.clients["openai"].chat.completions.create(
            model=self.openai_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system or "You are a thoughtful analyst providing your unique perspective."},
                {"role": "user", "content": prompt},
            ],
        )

        latency = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content
        metrics = StepMetrics(
            model=self.openai_model,
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            response_length=len(text),
        )

        return text, metrics

    def _call_gemini(self, prompt: str) -> tuple[str, StepMetrics]:
        """Call Gemini API and return response with metrics."""
        if "gemini" not in self.clients:
            raise ValueError("Gemini API key not configured")

        start = time.perf_counter()

        response = self.clients["gemini"].generate_content(prompt)

        latency = (time.perf_counter() - start) * 1000

        text = response.text

        # Gemini token counting
        try:
            input_tokens = self.clients["gemini"].count_tokens(prompt).total_tokens
            output_tokens = self.clients["gemini"].count_tokens(text).total_tokens
        except Exception:
            # Fallback estimation
            input_tokens = len(prompt) // 4
            output_tokens = len(text) // 4

        metrics = StepMetrics(
            model=self.gemini_model,
            latency_ms=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            response_length=len(text),
        )

        return text, metrics

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def run_sequential(
        self,
        query: str,
        synthesis_method: str = "A",
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Run the sequential chain: Claude -> OpenAI -> Gemini -> Synthesis

        synthesis_method:
            A: Final Claude call synthesizes all
            B: Structured extraction (programmatic)
            C: Parallel first, then synthesis
        """
        result = PipelineResult(
            query=query,
            claude_response="",
            openai_response="",
            gemini_response="",
            synthesis="",
            synthesis_method=synthesis_method,
        )

        accumulated_context = f"Original Query: {query}\n\n"

        # Step 1: Claude
        if verbose:
            print("[1/4] Calling Claude...")

        claude_prompt = query
        claude_response, claude_metrics = self._call_claude(claude_prompt)
        result.claude_response = claude_response
        result.metrics.append(claude_metrics)

        if verbose:
            print(f"      Claude: {claude_metrics.latency_ms:.0f}ms, {claude_metrics.total_tokens} tokens")

        accumulated_context += f"--- Claude's Perspective ---\n{claude_response}\n\n"

        # Step 2: OpenAI (with Claude's context)
        if verbose:
            print("[2/4] Calling OpenAI...")

        openai_prompt = f"""{accumulated_context}
---

Given the original query and Claude's perspective above, provide YOUR distinct perspective.
Build on, challenge, or extend the analysis. Identify what was missed or could be seen differently.

Your response:"""

        openai_response, openai_metrics = self._call_openai(openai_prompt)
        result.openai_response = openai_response
        result.metrics.append(openai_metrics)

        if verbose:
            print(f"      OpenAI: {openai_metrics.latency_ms:.0f}ms, {openai_metrics.total_tokens} tokens")

        accumulated_context += f"--- OpenAI's Perspective ---\n{openai_response}\n\n"

        # Step 3: Gemini (with full context)
        if verbose:
            print("[3/4] Calling Gemini...")

        gemini_prompt = f"""{accumulated_context}
---

Given the original query and the two perspectives above (Claude and OpenAI), provide YOUR distinct perspective.
Identify tensions, overlooked angles, or synthesis opportunities. What would a third viewpoint add?

Your response:"""

        gemini_response, gemini_metrics = self._call_gemini(gemini_prompt)
        result.gemini_response = gemini_response
        result.metrics.append(gemini_metrics)

        if verbose:
            print(f"      Gemini: {gemini_metrics.latency_ms:.0f}ms, {gemini_metrics.total_tokens} tokens")

        accumulated_context += f"--- Gemini's Perspective ---\n{gemini_response}\n\n"

        # Step 4: Synthesis
        if verbose:
            print("[4/4] Synthesizing...")

        synthesis, synth_metrics = self._synthesize(
            query=query,
            claude_response=claude_response,
            openai_response=openai_response,
            gemini_response=gemini_response,
            method=synthesis_method,
        )
        result.synthesis = synthesis
        if synth_metrics:
            result.metrics.append(synth_metrics)

        # Calculate totals
        result.total_latency_ms = sum(m.latency_ms for m in result.metrics)
        result.total_tokens = sum(m.total_tokens for m in result.metrics)

        # Context pressure analysis
        final_context_tokens = self._estimate_tokens(accumulated_context)
        result.context_pressure = {
            "accumulated_tokens": final_context_tokens,
            "claude_usage_pct": round(final_context_tokens / self.context_limits["claude"] * 100, 1),
            "openai_usage_pct": round(final_context_tokens / self.context_limits["openai"] * 100, 1),
            "gemini_usage_pct": round(final_context_tokens / self.context_limits["gemini"] * 100, 1),
        }

        return result

    def run_parallel_then_synthesize(
        self,
        query: str,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Option C: Run all three in parallel, then synthesize.
        Note: Python's threading isn't true parallelism but can overlap I/O.
        """
        import concurrent.futures

        result = PipelineResult(
            query=query,
            claude_response="",
            openai_response="",
            gemini_response="",
            synthesis="",
            synthesis_method="C",
        )

        if verbose:
            print("[1/2] Calling all three APIs in parallel...")

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            claude_future = executor.submit(self._call_claude, query)
            openai_future = executor.submit(self._call_openai, query)
            gemini_future = executor.submit(self._call_gemini, query)

            claude_response, claude_metrics = claude_future.result()
            openai_response, openai_metrics = openai_future.result()
            gemini_response, gemini_metrics = gemini_future.result()

        parallel_latency = (time.perf_counter() - start) * 1000

        result.claude_response = claude_response
        result.openai_response = openai_response
        result.gemini_response = gemini_response
        result.metrics.extend([claude_metrics, openai_metrics, gemini_metrics])

        if verbose:
            print(f"      Parallel phase: {parallel_latency:.0f}ms (wall clock)")
            print(f"      Claude: {claude_metrics.latency_ms:.0f}ms, OpenAI: {openai_metrics.latency_ms:.0f}ms, Gemini: {gemini_metrics.latency_ms:.0f}ms")

        # Synthesis
        if verbose:
            print("[2/2] Synthesizing...")

        synthesis, synth_metrics = self._synthesize(
            query=query,
            claude_response=claude_response,
            openai_response=openai_response,
            gemini_response=gemini_response,
            method="A",  # Use Claude for synthesis
        )
        result.synthesis = synthesis
        if synth_metrics:
            result.metrics.append(synth_metrics)

        result.total_latency_ms = parallel_latency + (synth_metrics.latency_ms if synth_metrics else 0)
        result.total_tokens = sum(m.total_tokens for m in result.metrics)

        return result

    def _synthesize(
        self,
        query: str,
        claude_response: str,
        openai_response: str,
        gemini_response: str,
        method: str = "A",
    ) -> tuple[str, Optional[StepMetrics]]:
        """
        Synthesize the three responses.

        Method A: Claude synthesizes
        Method B: Structured extraction (programmatic)
        """
        if method == "A":
            # Claude synthesis
            synthesis_prompt = f"""You are synthesizing three AI perspectives on a question.

ORIGINAL QUERY: {query}

CLAUDE'S PERSPECTIVE:
{claude_response}

OPENAI'S PERSPECTIVE:
{openai_response}

GEMINI'S PERSPECTIVE:
{gemini_response}

---

Create a synthesis that:
1. Identifies points of AGREEMENT across all three
2. Identifies points of TENSION or DISAGREEMENT
3. Notes UNIQUE insights from each that others missed
4. Provides a TRIANGULATED conclusion that preserves distinct perspectives rather than flattening them

Format your synthesis clearly with headers for each section."""

            return self._call_claude(
                synthesis_prompt,
                system="You are an expert synthesizer of multiple AI perspectives. Your goal is to preserve distinct viewpoints while identifying patterns.",
            )

        elif method == "B":
            # Programmatic structured extraction
            synthesis = self._structured_extraction(
                query, claude_response, openai_response, gemini_response
            )
            return synthesis, None

        else:
            raise ValueError(f"Unknown synthesis method: {method}")

    def _structured_extraction(
        self,
        query: str,
        claude_response: str,
        openai_response: str,
        gemini_response: str,
    ) -> str:
        """Programmatic extraction of key points."""
        # Simple extraction - split into sentences and identify key themes
        def extract_key_sentences(text: str, n: int = 5) -> list[str]:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
            return sentences[:n]

        claude_points = extract_key_sentences(claude_response)
        openai_points = extract_key_sentences(openai_response)
        gemini_points = extract_key_sentences(gemini_response)

        synthesis = f"""## Structured Synthesis: {query[:50]}...

### Claude's Key Points:
{chr(10).join(f"- {p}." for p in claude_points)}

### OpenAI's Key Points:
{chr(10).join(f"- {p}." for p in openai_points)}

### Gemini's Key Points:
{chr(10).join(f"- {p}." for p in gemini_points)}

### Summary
Three distinct perspectives captured. Method B provides raw extraction without AI-mediated synthesis.
Total unique points extracted: {len(claude_points) + len(openai_points) + len(gemini_points)}
"""
        return synthesis


def format_result(result: PipelineResult) -> str:
    """Format result for terminal output."""
    output = []
    output.append("=" * 70)
    output.append("PARALLAX TRIANGULATION RESULT")
    output.append("=" * 70)
    output.append(f"\nQUERY: {result.query}\n")

    output.append("-" * 70)
    output.append("CLAUDE'S PERSPECTIVE:")
    output.append("-" * 70)
    output.append(result.claude_response[:1500] + ("..." if len(result.claude_response) > 1500 else ""))

    output.append("\n" + "-" * 70)
    output.append("OPENAI'S PERSPECTIVE:")
    output.append("-" * 70)
    output.append(result.openai_response[:1500] + ("..." if len(result.openai_response) > 1500 else ""))

    output.append("\n" + "-" * 70)
    output.append("GEMINI'S PERSPECTIVE:")
    output.append("-" * 70)
    output.append(result.gemini_response[:1500] + ("..." if len(result.gemini_response) > 1500 else ""))

    output.append("\n" + "=" * 70)
    output.append(f"SYNTHESIS (Method {result.synthesis_method}):")
    output.append("=" * 70)
    output.append(result.synthesis)

    output.append("\n" + "=" * 70)
    output.append("METRICS:")
    output.append("=" * 70)
    output.append(f"Total Latency: {result.total_latency_ms:.0f}ms ({result.total_latency_ms/1000:.1f}s)")
    output.append(f"Total Tokens: {result.total_tokens:,}")

    total_cost = sum(m.estimated_cost for m in result.metrics)
    output.append(f"Estimated Cost: ${total_cost:.4f}")

    output.append("\nPer-Step Breakdown:")
    for m in result.metrics:
        output.append(f"  {m.model}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens, ${m.estimated_cost:.4f}")

    if result.context_pressure:
        output.append(f"\nContext Window Pressure:")
        output.append(f"  Accumulated: ~{result.context_pressure.get('accumulated_tokens', 0):,} tokens")
        output.append(f"  Claude: {result.context_pressure.get('claude_usage_pct', 0)}% of limit")
        output.append(f"  OpenAI: {result.context_pressure.get('openai_usage_pct', 0)}% of limit")
        output.append(f"  Gemini: {result.context_pressure.get('gemini_usage_pct', 0)}% of limit")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Parallax Triangulation Pipeline - Multi-AI synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("-m", "--method", choices=["A", "B", "C"], default="A",
                       help="Synthesis method: A=Claude, B=Structured, C=Parallel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--test", action="store_true", help="Run test query")

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.query = "What are the key considerations when designing systems that prevent exploitation structurally rather than through rules?"
        args.verbose = True

    if not args.query:
        parser.print_help()
        sys.exit(1)

    # Check API keys
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY")

    if missing:
        print("ERROR: Missing API keys in environment:")
        for key in missing:
            print(f"  - {key}")
        print("\nSet them in .env file or environment variables.")
        sys.exit(1)

    # Run pipeline
    pipeline = ParallaxPipeline()

    try:
        if args.method == "C":
            result = pipeline.run_parallel_then_synthesize(args.query, verbose=args.verbose)
        else:
            result = pipeline.run_sequential(args.query, synthesis_method=args.method, verbose=args.verbose)

        # Output
        print(format_result(result))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\nFull results saved to: {args.output}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
