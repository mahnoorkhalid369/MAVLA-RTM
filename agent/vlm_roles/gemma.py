"""The vision-language model behind every role: Gemma 4 E2B (vlm/model.py), or recorded answers in tests."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VLM(Protocol):
    def generate(self, text: str, image: np.ndarray | None = None, system: str | None = None,
                 max_new_tokens: int | None = None, thinking: bool | None = None) -> str: ...


class GemmaVLM:
    """The project's Gemma 4 (E4B by default; agent/settings.py GEMMA_MODEL) via vlm/model.py, loaded once."""

    def __init__(self, model_path: str | None = None) -> None:
        """model_path: a local model folder (agent/settings.py GEMMA_MODEL); None = vlm/config.py's default."""
        from vlm.config import VLMConfig
        from vlm.model import VisionLanguageModel

        self.model = VisionLanguageModel(VLMConfig() if model_path is None else VLMConfig(model_path=str(model_path)))

    def generate(self, text, image=None, system=None, max_new_tokens=None, thinking=None) -> str:
        out = self.model.generate(text, image=image, system=system, max_new_tokens=max_new_tokens, thinking=thinking)
        try:  # instrumentation only: text tokens in / out (image tokens are a fixed per-image cost, counted as images)
            tok = self.model.processor.tokenizer
            self.last_tokens = {"prompt": len(tok.encode((system or "") + text)), "answer": len(tok.encode(out)),
                                "images": int(image is not None)}
        except Exception:
            self.last_tokens = None
        return out


class RecordedVLM:
    """Replays recorded answers in order (tests: no GPU)."""

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.calls = list(answers), []

    def generate(self, text, image=None, system=None, max_new_tokens=None, thinking=None) -> str:
        self.calls.append(text)
        return self.answers.pop(0)
