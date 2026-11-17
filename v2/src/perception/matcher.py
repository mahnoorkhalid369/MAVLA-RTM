"""Image-text matching with SigLIP 2 (google/siglip2-base-patch16-224, loaded once): how well a picture of one object
matches each of several short names. No ROS, no Isaac Sim.

    matcher = Matcher()                                    # ~0.8 GB on the GPU in float16
    m = matcher.check(crop, "apple", ["apple", "milk carton", "blue tray"])
    m["ok"], m["top"], m["scores"]                         # True, "apple", {"apple": 0.93, ...}

It never chooses what the user meant: Gemma names the objects and chooses one; this only checks a close-up picture of
the chosen object against the names already in the catalogue (agent/catalogue.py). Scores are a softmax over the given
names, so they compare the names with each other (SigLIP's own per-pair scores are not calibrated across objects).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from agent import prompts

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL = ROOT / "models" / "siglip2-base-patch16-224"
HUB_MODEL = "google/siglip2-base-patch16-224"
MARGIN = 0.10  # the expected name may rank second when its score is within this of the first (near-synonyms)


class Matcher:
    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        path = model_path or (str(LOCAL_MODEL) if (LOCAL_MODEL / "config.json").exists() else HUB_MODEL)
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self._text_cache: dict[str, object] = {}

    def _text(self, names: list[str]):
        missing = [n for n in names if n not in self._text_cache]
        if missing:
            t = self.processor(text=[prompts.SIGLIP_TEXT.format(n) for n in missing], padding="max_length", max_length=64,
                               return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                f = self.model.get_text_features(**t)
            f = getattr(f, "pooler_output", f)
            f = f / f.norm(dim=-1, keepdim=True)
            for n, row in zip(missing, f):
                self._text_cache[n] = row
        return self.torch.stack([self._text_cache[n] for n in names])

    def _images(self, images: list[np.ndarray]):
        from PIL import Image
        pil = [Image.fromarray(np.asarray(i)[..., :3]) for i in images]
        x = self.processor(images=pil, return_tensors="pt").to(self.device)
        x["pixel_values"] = x["pixel_values"].to(self.model.dtype)
        with self.torch.inference_mode():
            f = self.model.get_image_features(**x)
        f = getattr(f, "pooler_output", f)
        return f / f.norm(dim=-1, keepdim=True)

    def scores(self, images: list[np.ndarray], names: list[str]) -> np.ndarray:
        """(images x names) softmax over the names, per image. Every image is encoded in one batch."""
        img, txt = self._images(images), self._text(names)
        with self.torch.inference_mode():
            logits = (img @ txt.T) * self.model.logit_scale.exp() + self.model.logit_bias
        return logits.float().softmax(dim=-1).cpu().numpy()

    def check(self, image: np.ndarray, expected: str, names: list[str]) -> dict:
        """Does a picture show `expected` rather than any other name in `names`? ok when `expected` ranks first, or
        second within MARGIN of the first."""
        names = list(dict.fromkeys([expected, *names]))  # unique, expected included
        p = self.scores([image], names)[0]
        order = np.argsort(-p)
        top, rank = names[order[0]], int(np.where(order == 0)[0][0])
        ok = rank == 0 or (rank == 1 and p[order[0]] - p[0] <= MARGIN)
        return {"ok": bool(ok), "expected": expected, "top": top, "rank": rank,
                "scores": {names[i]: round(float(p[i]), 3) for i in order[:5]}}
