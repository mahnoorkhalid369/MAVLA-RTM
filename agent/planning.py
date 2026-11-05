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


def bindings_text(bindings: dict) -> str:
    """The task's established bindings (the user's words -> the canonical #id), for every Gemma role later in the
    task, so no role resolves those words again (live 2026-11-04: decide bound the user's words to #9, the
    decomposition was not told and planned only to observe). Ids and keys of the form "#<n>" / "#held" are references already."""
    rows = [f'"{t}" = #{w}' for t, w in (bindings or {}).items() if isinstance(w, int) and not str(t).startswith("#")]
    if not rows:
        return ""
    return ("Established task bindings (use these #ids for those words; do not resolve the words again):\n"
            + "\n".join(rows))
