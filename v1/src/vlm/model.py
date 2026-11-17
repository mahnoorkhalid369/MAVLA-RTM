"""Gemma 4 E2B-it wrapper: one generate() call for any mix of text, image and audio.

No ROS, no Isaac Sim. Load once (several seconds), then call generate() as often as needed.

    vlm = VisionLanguageModel(VLMConfig())
    text = vlm.generate(text="What is on the table?", image=rgb_array)
    text = vlm.generate(text="Transcribe this.", audio=pcm_float32_16k)
"""

from __future__ import annotations

import logging
import os
import time

# bitsandbytes logs (logging, not warnings) on every 8-bit matmul that bfloat16 inputs are cast to float16;
# expected, so drop only that message.
logging.getLogger("bitsandbytes.autograd._functions").addFilter(
    lambda r: not r.getMessage().startswith("MatMul8bitLt: inputs will be cast"))

# Lets the CUDA allocator hand back the memory freed when the embedding table moves to the CPU
# (with the default allocator the process keeps ~4.6 GB reserved after loading).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig  # noqa: E402

from vlm.config import VLMConfig  # noqa: E402

PER_LAYER_EMBEDDINGS = "model.language_model.embed_tokens_per_layer"


class VisionLanguageModel:
    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        cfg = self.config
        quant = {
            "4bit": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "8bit": BitsAndBytesConfig(load_in_8bit=True),
            "16bit": None,
        }[cfg.quantization]
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(cfg.model_path)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            cfg.model_path, dtype=torch.bfloat16, device_map=cfg.device_map, quantization_config=quant
        )
        self.model.eval()
        if cfg.per_layer_embeddings_on_cpu:
            self._embeddings_to_cpu()
        self.load_seconds = time.monotonic() - start

    def _embeddings_to_cpu(self) -> None:
        """Move the 4.7 GB per-layer embedding table to system RAM (it is only used for lookups)."""
        table = self.model.get_submodule(PER_LAYER_EMBEDDINGS)
        gpu = next(p.device for p in self.model.parameters() if p.device.type == "cuda")
        table.to("cpu")
        table.register_forward_pre_hook(lambda _m, args: (args[0].to("cpu"),) + tuple(args[1:]))
        table.register_forward_hook(lambda _m, _args, out: out.to(gpu))
        torch.cuda.empty_cache()

    def _image(self, image: np.ndarray | Image.Image) -> Image.Image:
        img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)[..., :3])
        side = self.config.image_max_side
        if max(img.size) > side:
            scale = side / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
        return img

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        image: np.ndarray | Image.Image | None = None,
        audio: np.ndarray | None = None,
        system: str | None = None,
        max_new_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> str:
        """Run the model once. image: HxWx3 uint8 RGB; audio: mono float32 at config.audio_rate.

        thinking: override config.thinking for this call (e.g. off for quick transcription).
        Returns the final answer only (the reasoning, when thinking, is dropped).
        """
        think = self.config.thinking if thinking is None else thinking
        content: list[dict] = []
        if image is not None:  # Gemma: image before text
            content.append({"type": "image", "image": self._image(image)})
        if audio is not None:
            content.append({"type": "audio", "audio": np.asarray(audio, dtype=np.float32)})
        content.append({"type": "text", "text": text})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=think,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        budget = max_new_tokens or self.config.max_new_tokens
        if think:  # reasoning comes first and can use most of the budget
            budget = max(budget, self.config.thinking_tokens)
        outputs = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        raw = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = self.processor.parse_response(raw, prefix=inputs["input_ids"])
        if isinstance(parsed, dict):  # {"content": ..., "thinking": ...}
            return str(parsed.get("content") or "").strip()
        return str(parsed).strip()

    def memory_gb(self) -> float:
        """GPU memory currently held by the model's tensors (GB)."""
        return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
