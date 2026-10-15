"""Request authority: what the USER's request permits the robot to do, decided once per instruction from the
user's own words and enforced before any capability runs (agent/workers.WorkerRegistry.dispatch).

Effect classes (CapabilitySpec.effect):
  none           read-only: observe, describe, answer, locate, plan
  sensing        robot motion that changes no object: wrist-camera inspection / exploration, return HOME
  object_change  grasp, carry, place: changes the world

Every request may use none + sensing (looking closer can serve a question; motion still passes the collision
checks and the safety layer). object_change is granted only when the request itself asks for a change to the
world. The decision comes from a dedicated classification of the request text alone (Roles.classify_request:
it never sees a plan), must quote verbatim evidence from the request, and falls back to least privilege on any
failure. Plans - including Gemma's decompositions - cannot change it: the supervisor only reads it, and the
registry refuses a capability whose effect is not granted, whatever the plan says.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

EFFECTS = ("none", "sensing", "object_change")
KINDS = ("question", "description", "action")


@dataclass(frozen=True)
class RequestAuthority:
    kind: str  # question | description | action
    grants: tuple  # effect classes allowed for this instruction
    evidence: str = ""  # verbatim words of the request that support `kind`
    source: str = "user_request"
    note: str = ""  # why least privilege was applied, if it was

    def allows(self, effect: str) -> bool:
        return effect in self.grants

    def to_dict(self) -> dict:
        return asdict(self)


LEAST = ("none", "sensing")


def least_privilege(note: str, kind: str = "question") -> RequestAuthority:
    return RequestAuthority(kind, LEAST, "", "default", note)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s#-]", " ", (s or "").lower())).strip()


def from_classification(instruction: str, kind: str, evidence: str) -> RequestAuthority:
    """Deterministic part of the decision: an "action" grants object_change only when its evidence is copied
    verbatim from the request (a classifier cannot grant authority with words the user did not say)."""
    if kind not in KINDS:
        return least_privilege(f"unknown request kind {kind!r}")
    if kind != "action":
        return RequestAuthority(kind, LEAST, evidence)
    ev = _norm(evidence)
    if not ev or ev not in _norm(instruction):
        return least_privilege(f"action claimed without verbatim evidence from the request ({evidence!r})", "action")
    return RequestAuthority("action", EFFECTS, evidence)


def classify(instruction: str, roles) -> RequestAuthority:
    """Authority for one instruction. Never raises: any failure means least privilege."""
    fn = getattr(roles, "classify_request", None)
    if fn is None:
        return least_privilege("no request classifier available")
    try:
        r = fn(instruction)
    except Exception as exc:  # the classifier failing must not grant anything
        return least_privilege(f"request classification failed: {exc!r}"[:200])
    return from_classification(instruction, r.kind, r.evidence)
