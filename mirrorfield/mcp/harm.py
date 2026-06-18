"""
Optional harm classifier for the composed SEND / VERIFY / HOLD pipeline.

Wraps IBM Granite-Guardian-3.1-2B — a THIRD-PARTY safety classifier, NOT a
contribution of this project — to produce a harm probability for a
``(prompt, response)`` pair. Compose it with the validated wrongness gate via
``mirrorfield.send_hold_decision(..., harm_score=clf.score(prompt, response))``.

Heavy and optional: requires ``pip install "mirrorfield[harm]"`` (torch,
transformers, huggingface_hub) plus the Granite model (~2B params) cached locally.
``torch`` is imported lazily — only when you instantiate ``HarmClassifier`` — so
importing this module stays cheap.

Honest scope (WORK_MAP §4l–§4r): harm is delegated to Granite precisely because a
dedicated classifier (AUC ~0.87 on this project's tests) substantially beats the
wrongness gate's incidental harm signal. The gate judges *likely-wrong*; Granite
judges *harmful*; the pipeline keeps them separate and only HOLDs on harm when a
score is actually supplied.
"""

from __future__ import annotations

DEFAULT_MODEL = "ibm-granite/granite-guardian-3.1-2b"


def _resolve(model_id: str) -> str:
    """Prefer the cached local snapshot path. Passing a local directory to
    ``from_pretrained`` skips a network metadata probe that crashes some
    transformers versions under offline mode. Falls back to the repo id."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


class HarmClassifier:
    """Granite-Guardian harm scorer. Lazy: loads torch + the model on init.

    VRAM rule (this project, 8 GB): never co-resident with a large generator —
    load the HarmClassifier only AFTER the generating model has been freed.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        path = _resolve(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        # verdict-token ids (cover leading-space variants)
        self._yes = self._ids(["Yes", " Yes", "yes", " yes"])
        self._no = self._ids(["No", " No", "no", " no"])

    def _ids(self, variants: list[str]) -> list[int]:
        out: set[int] = set()
        for v in variants:
            ids = self.tok(v, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
        return sorted(out)

    def _chat_ids(self, prompt: str, response: str):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        try:
            return self.tok.apply_chat_template(
                msgs, guardian_config={"risk_name": "harm"},
                add_generation_prompt=True, return_tensors="pt")
        except TypeError:  # older tokenizer without guardian_config kwarg
            return self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")

    def score(self, prompt: str, response: str) -> float:
        """P(harmful) in [0, 1] for ``response`` given ``prompt``."""
        torch = self._torch
        ids = self._chat_ids(prompt, response or " ")[:, -2048:].to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        py = float(sum(p[i] for i in self._yes))
        pn = float(sum(p[i] for i in self._no))
        return py / (py + pn + 1e-12)
