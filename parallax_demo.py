"""
Parallax Pipeline Demo - Single API Multi-Perspective Simulation

Demonstrates the pipeline concept using OpenAI with different persona prompts.
Use this to validate the concept while waiting for other API keys.
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class StepMetrics:
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

    @property
    def estimated_cost(self) -> float:
        return (self.input_tokens * 0.005 + self.output_tokens * 0.015) / 1000


def call_openai_with_persona(client, persona: str, query: str, context: str = "") -> tuple[str, StepMetrics]:
    """Call OpenAI with a specific persona prompt."""

    personas = {
        "analytical": """You are a rigorous analytical thinker. Focus on:
- Logical structure and systematic reasoning
- Evidence and verifiable claims
- Identifying assumptions and dependencies
- Breaking complex ideas into components
Provide your ANALYTICAL perspective.""",

        "creative": """You are a creative divergent thinker. Focus on:
- Unexpected connections and metaphors
- Alternative framings of the problem
- What others might overlook
- Emergent possibilities
Provide your CREATIVE perspective.""",

        "practical": """You are a pragmatic systems thinker. Focus on:
- Real-world implementation challenges
- Second-order effects and unintended consequences
- Historical precedents and patterns
- What actually works vs. what sounds good
Provide your PRACTICAL perspective.""",

        "synthesis": """You are synthesizing three distinct perspectives on a question.
Create a synthesis that:
1. Identifies points of AGREEMENT across all three
2. Identifies points of TENSION or DISAGREEMENT
3. Notes UNIQUE insights from each perspective
4. Provides a TRIANGULATED conclusion that preserves distinct viewpoints"""
    }

    system = personas.get(persona, personas["analytical"])

    if context:
        prompt = f"{context}\n\n---\n\nNow provide YOUR {persona.upper()} perspective on: {query}"
    else:
        prompt = query

    start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    latency = (time.perf_counter() - start) * 1000

    return response.choices[0].message.content, StepMetrics(
        model="gpt-4o",
        latency_ms=latency,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )


def run_demo(query: str, verbose: bool = True):
    """Run the multi-perspective demo."""
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = {}
    metrics = []
    accumulated = f"ORIGINAL QUERY: {query}\n\n"

    # Step 1: Analytical perspective
    if verbose:
        print("[1/4] Getting ANALYTICAL perspective...")
    response, m = call_openai_with_persona(client, "analytical", query)
    results["analytical"] = response
    metrics.append(m)
    accumulated += f"--- ANALYTICAL PERSPECTIVE ---\n{response}\n\n"
    if verbose:
        print(f"      Done: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    # Step 2: Creative perspective (with context)
    if verbose:
        print("[2/4] Getting CREATIVE perspective...")
    response, m = call_openai_with_persona(client, "creative", query, accumulated)
    results["creative"] = response
    metrics.append(m)
    accumulated += f"--- CREATIVE PERSPECTIVE ---\n{response}\n\n"
    if verbose:
        print(f"      Done: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    # Step 3: Practical perspective (with full context)
    if verbose:
        print("[3/4] Getting PRACTICAL perspective...")
    response, m = call_openai_with_persona(client, "practical", query, accumulated)
    results["practical"] = response
    metrics.append(m)
    accumulated += f"--- PRACTICAL PERSPECTIVE ---\n{response}\n\n"
    if verbose:
        print(f"      Done: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    # Step 4: Synthesis
    if verbose:
        print("[4/4] Synthesizing perspectives...")

    synth_prompt = f"""{accumulated}

---

Synthesize these three perspectives. Preserve what's unique in each while identifying patterns."""

    response, m = call_openai_with_persona(client, "synthesis", synth_prompt)
    results["synthesis"] = response
    metrics.append(m)
    if verbose:
        print(f"      Done: {m.latency_ms:.0f}ms, {m.total_tokens} tokens")

    return results, metrics


def format_output(query: str, results: dict, metrics: list) -> str:
    """Format results for display."""
    out = []
    out.append("=" * 70)
    out.append("PARALLAX TRIANGULATION DEMO (Single-API Multi-Perspective)")
    out.append("=" * 70)
    out.append(f"\nQUERY: {query}\n")

    for name in ["analytical", "creative", "practical"]:
        out.append("-" * 70)
        out.append(f"{name.upper()} PERSPECTIVE:")
        out.append("-" * 70)
        text = results[name]
        out.append(text[:1200] + ("..." if len(text) > 1200 else ""))
        out.append("")

    out.append("=" * 70)
    out.append("SYNTHESIS:")
    out.append("=" * 70)
    out.append(results["synthesis"])

    out.append("\n" + "=" * 70)
    out.append("METRICS:")
    out.append("=" * 70)

    total_latency = sum(m.latency_ms for m in metrics)
    total_tokens = sum(m.total_tokens for m in metrics)
    total_cost = sum(m.estimated_cost for m in metrics)

    out.append(f"Total Latency: {total_latency:.0f}ms ({total_latency/1000:.1f}s)")
    out.append(f"Total Tokens: {total_tokens:,}")
    out.append(f"Estimated Cost: ${total_cost:.4f}")

    perspectives = ["analytical", "creative", "practical", "synthesis"]
    out.append("\nPer-Step:")
    for i, m in enumerate(metrics):
        out.append(f"  {perspectives[i]}: {m.latency_ms:.0f}ms, {m.total_tokens} tokens, ${m.estimated_cost:.4f}")

    return "\n".join(out)


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "What are the key considerations when designing systems that "
        "prevent exploitation structurally rather than through rules?"
    )

    print(f"\nRunning Parallax Demo with query:\n\"{query}\"\n")

    results, metrics = run_demo(query)
    print("\n" + format_output(query, results, metrics))

    # Save full results
    output = {
        "query": query,
        "perspectives": results,
        "metrics": {
            "total_latency_ms": sum(m.latency_ms for m in metrics),
            "total_tokens": sum(m.total_tokens for m in metrics),
            "total_cost_usd": sum(m.estimated_cost for m in metrics),
        }
    }

    with open("parallax_demo_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n\nFull output saved to: parallax_demo_output.json")


if __name__ == "__main__":
    main()
