"""Reproducible ablations; experimental inference/motion changes are opt-in."""

from dataclasses import asdict, dataclass
import os
from contextlib import contextmanager
from contextvars import ContextVar

inference_phase = ContextVar("inference_phase", default="default")


@contextmanager
def phase(name):
    token = inference_phase.set(name)
    try:
        yield
    finally:
        inference_phase.reset(token)


def flag(name, default=False):
    raw = os.environ.get(name, "1" if default else "0").lower()
    if raw not in {"0", "1", "true", "false"}:
        raise ValueError(f"{name}: expected 0/1")
    return raw in {"1", "true"}


@dataclass(frozen=True)
class Performance:
    reuse_placement: bool = False
    finish_plan: bool = False
    overlap_home: bool = False
    chain_transfers: bool = False
    placement_thinking: bool = True
    wrist_image_tokens: int = 1120
    reasoning_tokens: int | None = None
    carry_speed_scale: float = 1.0
    prefix_cache: bool = False

    @classmethod
    def from_env(cls):
        baseline = flag("V2_BASELINE")
        p = cls(reuse_placement=not baseline and flag("V2_REUSE_PLACEMENT"),
                finish_plan=not baseline and flag("V2_FINISH_PLAN"),
                overlap_home=not baseline and flag("V2_OVERLAP_HOME"),
                chain_transfers=not baseline and flag("V2_CHAIN_TRANSFERS"),
                placement_thinking=baseline or flag("V2_PLACEMENT_THINKING", True),
                wrist_image_tokens=1120 if baseline else int(os.environ.get("V2_WRIST_IMAGE_TOKENS", "1120")),
                reasoning_tokens=None if baseline else (int(os.environ["V2_REASONING_TOKENS"])
                                                       if "V2_REASONING_TOKENS" in os.environ else None),
                carry_speed_scale=1.0 if baseline else float(os.environ.get("V2_CARRY_SPEED_SCALE", "1")),
                prefix_cache=not baseline and flag("V2_PREFIX_CACHE"))
        if p.wrist_image_tokens not in {70, 140, 280, 560, 1120}:
            raise ValueError("invalid wrist visual token budget")
        if p.reasoning_tokens is not None and p.reasoning_tokens < 1:
            raise ValueError("reasoning token budget must be positive")
        if not 1 <= p.carry_speed_scale <= 1.5:
            raise ValueError("carry speed scale must be between 1 and 1.5")
        return p

    def manifest(self):
        return asdict(self)


def options(tools):
    policy = getattr(tools, "performance", None)
    return policy if policy is not None else Performance.from_env()
