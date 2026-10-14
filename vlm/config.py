"""Settings for the vision-language model (Gemma 4 E2B-it)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL = ROOT / "models" / "gemma-4-E2B-it"
HUB_MODEL = "google/gemma-4-E2B-it"


@dataclass
class VLMConfig:
    # Local copy if downloaded (models/gemma-4-E2B-it), otherwise the Hugging Face id
    model_path: str = str(LOCAL_MODEL) if (LOCAL_MODEL / "config.json").exists() else HUB_MODEL
    # Weights are 10.2 GB in bf16, too much next to Isaac Sim on a 12 GB GPU:
    # "4bit" (~4 GB, default) | "8bit" (~6 GB) | "none" (bf16, ~11 GB)
    quantization: str = "4bit"
    device_map: str = "cuda:0"
    # The per-layer embedding table (4.7 GB, lookup only) can live in system RAM: saves ~4.7 GB of
    # GPU memory for a small speed cost. Needed to share the 12 GB GPU with Isaac Sim.
    per_layer_embeddings_on_cpu: bool = True
    max_new_tokens: int = 1024
    thinking: bool = True  # Gemma 4 reasoning mode: better plans, slower
    thinking_tokens: int = 3072  # token budget when thinking (reasoning + answer)
    image_max_side: int = 960  # camera frames are downscaled to this before the model sees them
    audio_rate: int = 16000  # Hz, what the audio encoder expects
