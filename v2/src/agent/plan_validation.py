"""Controller validation of transfer boundaries; no object/goal inference from model memory."""

import re


def requests_placement(message):
    return bool(re.search(r"\b(?:place|put|transfer|deposit)\b|\bmove\b.+\b(?:to|onto|into)\b", message, re.I))


def pick_only(message):
    return bool(re.match(r"(?:pick(?:\s+up)?|grasp|lift)\b", message, re.I)) and not requests_placement(message)


def complete_transfers(steps, request=None):
    """Join only an explicit adjacent pick/place pair for the same subject (or it/them).

    Different or ambiguous subjects require the master's bounded response repair, never a guessed
    merge. A genuine user pick-only request remains supported. An unpaired pickup in a transfer
    request must be repaired before any robot dispatch.
    """
    out, i = [], 0
    while i < len(steps):
        worker, message = steps[i]
        if worker == "manipulation" and pick_only(message) and i + 1 < len(steps):
            nw, nm = steps[i + 1]
            place = re.fullmatch(r"(?:place|put)\s+(.+?)\s+(?:on|in|into|onto|inside|next to|beside)\s+.+[.]?", nm, re.I)
            if nw == "manipulation" and place:
                subject = re.sub(r"^(?:pick(?:\s+up)?|grasp|lift)\s+(?:the\s+)?", "", message, flags=re.I).strip(" .").casefold()
                other = re.sub(r"^the\s+", "", place[1], flags=re.I).strip(" .").casefold()
                if other not in (subject, "it", "them"):
                    raise ValueError("split pickup/placement subjects differ; return one full transfer per object")
                out.append((worker, message.rstrip(" .") + " and " + nm[0].lower() + nm[1:]))
                i += 2
                continue
        if worker == "manipulation" and pick_only(message) and request and requests_placement(request):
            raise ValueError("transfer plan contains an unpaired pickup; include its destination in the same manipulation step")
        out.append((worker, message))
        i += 1
    return out
