"""Plan helpers: pure functions over plan steps (no robot, no model, no language rules)."""

from __future__ import annotations


def _ref_texts(args: dict) -> list[str]:
    out = []
    for k in ("object", "reference"):
        v = args.get(k)
        if isinstance(v, dict) and v.get("text"):
            out.append(v["text"])
    if args.get("target"):
        out.append(args["target"])
    return out
