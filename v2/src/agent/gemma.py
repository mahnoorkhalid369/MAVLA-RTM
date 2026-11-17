"""Gemma, the agent's one language model (Gemma 4 E4B, 4-bit, loaded once by vlm/model.py).

    gemma = Gemma(model_path)
    text = gemma.ask("question", images=[rgb], system="...", thinking=True)

ask() takes any number of images (shown before the text, as Gemma expects) and returns the answer only; with
thinking on, Gemma's reasoning is kept in gemma.last_thinking for the log. Decoding is greedy (same input, same
answer). VLMFailure is raised when an answer cannot be used after one retry.
"""

from __future__ import annotations

import numpy as np
import hashlib
import time
from PIL import Image
from agent.timing import timed
from agent.performance import Performance, inference_phase


class VLMFailure(RuntimeError):
    pass


class Gemma:
    def __init__(self, model_path: str | None = None) -> None:
        import torch
        from vlm.config import VLMConfig
        from vlm.model import VisionLanguageModel

        self.torch = torch
        self.vlm = VisionLanguageModel(VLMConfig() if model_path is None else VLMConfig(model_path=str(model_path)))
        self.last_thinking = ""
        self.call_count = 0

    @timed("gemma")
    def ask(self, text: str, images=(), system: str | None = None, thinking: bool = False,
            max_new_tokens: int = 400) -> str:
        start = time.monotonic()
        policy = Performance.from_env()
        phase = inference_phase.get()
        if phase == "placement":
            thinking = policy.placement_thinking
        proc, model = self.vlm.processor, self.vlm.model
        prepared = [self.vlm._image(_pil(i)) for i in images]
        self.call_count = getattr(self, "call_count", 0) + 1
        budget = max(max_new_tokens, self.vlm.config.thinking_tokens) if thinking else max_new_tokens
        if thinking and system and policy.reasoning_tokens is not None:
            budget = min(budget, policy.reasoning_tokens)
        log = getattr(self, "log", None)
        if log is not None:
            evidence = []
            for index, img in enumerate(prepared):
                path = None
                directory = getattr(self, "run_dir", None)
                if directory is not None:
                    folder = directory / "images/model_inputs"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{self.call_count:03d}_{index}.png"
                    img.save(path)
                evidence.append({"shape": [img.height, img.width, len(img.getbands())],
                                 "sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                                 "path": str(path) if path is not None else None})
            log("model_images", call_id=self.call_count, images=evidence, thinking=thinking, max_new_tokens=budget,
                requested_max_new_tokens=max_new_tokens)
        content = [{"type": "image", "image": img} for img in prepared]
        content.append({"type": "text", "text": text})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        processing = {"return_tensors": "pt"}
        if phase == "placement":
            processing["images_kwargs"] = {"max_soft_tokens": policy.wrist_image_tokens}
        inputs = proc.apply_chat_template(messages, tokenize=True, return_dict=True, processor_kwargs=processing,
                                          add_generation_prompt=True, enable_thinking=thinking).to(model.device)
        n = inputs["input_ids"].shape[-1]
        generated_at = time.monotonic()
        cached, cache_info = {}, {"hit": False, "tokens": 0}
        with self.torch.inference_mode():
            if policy.prefix_cache and system:
                from agent.prefix_cache import SystemPrefixCache
                if not hasattr(self, "prefix_cache"):
                    self.prefix_cache = SystemPrefixCache()
                cached, cache_info = self.prefix_cache.prepare(proc, model, inputs, system, thinking)
            out = model.generate(**inputs, **cached, max_new_tokens=budget, do_sample=False)
        # generate is synchronous through decoding below; no CUDA-only timing dependency.
        parsed = proc.parse_response(proc.decode(out[0][n:], skip_special_tokens=False), prefix=inputs["input_ids"])
        if isinstance(parsed, dict):
            self.last_thinking = str(parsed.get("thinking") or "").strip()
            answer = str(parsed.get("content") or "").strip()
        else:
            self.last_thinking = ""
            answer = str(parsed).strip()
        if log is not None:
            log("model_output", call_id=self.call_count, answer=answer, thinking=self.last_thinking,
                phase=phase, prompt_chars=len(text), system_chars=len(system or ""), input_tokens=int(n),
                image_count=len(prepared), generated_tokens=int(out.shape[-1]-n),
                thinking_chars=len(self.last_thinking),
                thinking_tokens=len(proc.tokenizer.encode(self.last_thinking, add_special_tokens=False)),
                thinking_tokens_method="retokenized_text",
                seconds=time.monotonic()-start, preprocessing_s=generated_at-start,
                generation_s=time.monotonic()-generated_at, budget=budget,
                prefix_cache=cache_info,
                budget_exhausted=int(out.shape[-1]-n) >= budget,
                image_tokens=policy.wrist_image_tokens if phase == "placement" else self.vlm.config.image_tokens)
        return answer


def _pil(image) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)[..., :3])

