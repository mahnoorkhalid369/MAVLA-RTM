"""Settings for the vision-language model (Gemma 4 E4B-it; E2B-it is still available for comparison).

Every field below can be overridden for one run with an environment variable (benchmarks switch configurations this
way without editing this file; unset = the default written here):

    VLM_QUANTIZATION=4bit|8bit|16bit   VLM_EMBEDDINGS_ON_CPU=1|0   VLM_MAX_NEW_TOKENS=<int>
    VLM_THINKING=1|0                  VLM_THINKING_TOKENS=<int>   VLM_IMAGE_MAX_SIDE=<int>
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/<package>/)
LOCAL_MODEL = ROOT / "models" / "gemma-4-E4B-it"
HUB_MODEL = "google/gemma-4-E4B-it"
# models/gemma-4-E2B-it (google/gemma-4-E2B-it): the smaller model used until 2026-10-26, kept for comparison


def _env(name: str, default, kind=str):
    """Factory for a field read from the environment at construction time (VLM_<NAME>), else the default."""
    def read():
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        if kind is bool:
            if raw.lower() in ("1", "true", "yes", "on"):
                return True
            if raw.lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"{name}={raw!r}: expected 1/0")
        return kind(raw)
    return field(default_factory=read)


@dataclass
class VLMConfig:
    # Local copy if downloaded (models/gemma-4-E4B-it), otherwise the Hugging Face id
    model_path: str = str(LOCAL_MODEL) if (LOCAL_MODEL / "config.json").exists() else HUB_MODEL
    # Weights are 16.0 GB in bf16 (E4B; E2B: 10.2 GB), too much next to Isaac Sim on a 12 GB GPU: "4bit" (default) |
    # "8bit" | "16bit" (bf16). Measured with E2B 4-bit (out/gpu_profile/2026-10-26): Gemma + YOLO ~2.3 GB next to Isaac
    # Sim's ~2.6 GB; loading briefly peaks ~6.8 GB higher, so the models load BEFORE the simulator starts (agent/run.py)
    quantization: str = _env("VLM_QUANTIZATION", "4bit")
    device_map: str = "cuda:0"
    # The per-layer embedding table (E4B 5.6 GB, E2B 4.7 GB; lookup only) can live in system RAM: saves that much
    # GPU memory at no measured speed cost (2026-11-05, 4bit E2B: 31 tokens/s on CPU vs 30 on GPU; E4B on the GPU
    # takes 9.5 GB, too much next to Isaac Sim). Needed to share the 12 GB GPU with Isaac Sim.
    per_layer_embeddings_on_cpu: bool = _env("VLM_EMBEDDINGS_ON_CPU", True, bool)
    max_new_tokens: int = _env("VLM_MAX_NEW_TOKENS", 1024, int)
    thinking: bool = _env("VLM_THINKING", True, bool)  # Gemma 4 reasoning mode: better plans, slower
    thinking_tokens: int = _env("VLM_THINKING_TOKENS", 3072, int)  # token budget when thinking (reasoning + answer)
    image_max_side: int = _env("VLM_IMAGE_MAX_SIDE", 1280, int)  # camera frames are downscaled to this first
    # Visual tokens per image (Gemma 4: 70 | 140 | 280 | 560 | 1120; the model's own default is 280). At 280 one token
    # covers ~48 x 48 px of a 1280 x 720 frame, more than a 30-40 px block: colours blurred into their neighbours
    # (live 2026-11-11: one yellow block reported as "three yellow blocks"). 1120 = 4x the detail, ~24 x 24 px.
    image_tokens: int = _env("VLM_IMAGE_TOKENS", 1120, int)
    audio_rate: int = 16000  # Hz, what the audio encoder expects

    def __post_init__(self) -> None:
        if self.quantization not in ("4bit", "8bit", "16bit"):
            raise ValueError(f"quantization={self.quantization!r}: expected 4bit, 8bit or 16bit")
        if self.image_tokens not in (70, 140, 280, 560, 1120):
            raise ValueError(f"image_tokens={self.image_tokens}: Gemma 4 takes 70, 140, 280, 560 or 1120")
