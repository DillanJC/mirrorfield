"""
Claude API client for multi-step reasoning.

Wraps the Anthropic SDK to generate reasoning steps with:
- Conversation continuity (previous steps as context)
- Intervention injection (meta-instructions between steps)
- Rate limit handling (retry with exponential backoff)
- Temperature control (0.7 default for stochastic variation)
"""

import os
import time
import anthropic
from pathlib import Path
from typing import Optional


# Load .env from project root
def _load_env():
    """Load .env file using python-dotenv."""
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()

# Default configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


class ClaudeReasoningClient:
    """
    Claude API wrapper for multi-step reasoning tasks.

    Each reasoning step builds on previous steps via conversation context.
    Interventions are injected as meta-instructions between steps.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env or environment."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_api(self, system: str, messages: list) -> str:
        """Call Claude API with retry and backoff for rate limits."""
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=messages,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = backoff * (2 ** attempt)
                print(f"    Rate limited, retrying in {wait:.1f}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < MAX_RETRIES - 1:
                    # Overloaded
                    wait = backoff * (2 ** attempt)
                    print(f"    API overloaded, retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def generate_step(
        self,
        task_prompt: str,
        step_index: int,
        total_steps: int,
        previous_steps: Optional[list] = None,
        intervention_text: Optional[str] = None,
    ) -> str:
        """
        Generate a single reasoning step.

        Args:
            task_prompt: The task being reasoned about.
            step_index: Current step index (0-based).
            total_steps: Total number of steps planned.
            previous_steps: List of previous step texts for continuity.
            intervention_text: Optional meta-instruction to inject.

        Returns:
            Response text for this step.
        """
        system = (
            f"You are reasoning through a complex task in {total_steps} steps. "
            f"This is step {step_index + 1} of {total_steps}. "
            f"Each step should build on previous steps and advance the reasoning. "
            f"Be substantive, analytical, and nuanced. "
            f"Acknowledge uncertainty where appropriate. "
            f"Consider multiple perspectives and trade-offs."
        )

        messages = []

        # User message: the task
        messages.append({
            "role": "user",
            "content": f"Task: {task_prompt}",
        })

        # Build conversation from previous steps
        if previous_steps:
            for i, step_text in enumerate(previous_steps):
                messages.append({
                    "role": "assistant",
                    "content": step_text,
                })
                if i < len(previous_steps) - 1:
                    messages.append({
                        "role": "user",
                        "content": f"Continue to step {i + 2} of {total_steps}.",
                    })

        # Prompt for current step
        if intervention_text:
            prompt = (
                f"Before continuing to step {step_index + 1}, consider this: "
                f"{intervention_text}\n\n"
                f"Now proceed with step {step_index + 1} of {total_steps}."
            )
        elif previous_steps:
            prompt = (
                f"Continue to step {step_index + 1} of {total_steps}."
            )
        else:
            prompt = (
                f"Begin step 1 of {total_steps}. "
                f"Start by framing the problem and identifying key dimensions."
            )

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return self._call_api(system, messages)

    def generate_reasoning(
        self,
        task_prompt: str,
        n_steps: int = 4,
        intervention_per_step: Optional[dict] = None,
    ) -> list:
        """
        Orchestrate multi-step reasoning for a task.

        Args:
            task_prompt: The task to reason about.
            n_steps: Number of reasoning steps (default 4).
            intervention_per_step: Optional dict mapping step_index -> intervention text.

        Returns:
            List of step texts.
        """
        if intervention_per_step is None:
            intervention_per_step = {}

        steps = []
        for i in range(n_steps):
            intervention = intervention_per_step.get(i)
            step_text = self.generate_step(
                task_prompt=task_prompt,
                step_index=i,
                total_steps=n_steps,
                previous_steps=steps if steps else None,
                intervention_text=intervention,
            )
            steps.append(step_text)

        return steps
