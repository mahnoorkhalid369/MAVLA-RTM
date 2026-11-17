# Instruction flow

Open `instruction-flow.html` directly in a browser. It is a standalone, offline page with its own
styles; `instruction-flow-preview.html` is an identical compatibility copy. Reload an already-open
tab to see changes. No server, simulator or model is started by this animation.

The compact diagram highlights the active source, receiving node, and connecting arrows. Playback
is finite and supports pause, back/next, replay, and scrubbing. Reduced-motion preferences disable
moving packets and initial autoplay.

Paths cover a successful baseline transfer, optional tray clarification, bounded source-side
identity recovery and HOME, final-check rejection, and startup catalogue naming. Timing is
illustrative. Camera thumbnails are examples; the page is not connected to a live robot session.

The shared Gemma node is the single model used by the roles. The counter increments once per
generation, not for every prompt fragment or motion state. The successful baseline example shows
five calls: master plan, grounding, manipulation placement judgement, separate worker verification,
and the final original-request check. Startup naming is shown separately, one call per new object.
SigLIP is a separate image/text encoder; deterministic motion and sensor checks are not Gemma calls.

`View prompts` opens the full embedded templates from `src/agent/prompts.py`. Prompt names under
the current call are clickable. After editing that Python source, refresh the embedded snapshot:

```bash
python3 v2/visualizations/update_prompt_snapshot.py
```

This updates the viewer snapshot, not the running agent or its policy. The animation does not
claim the final-check interpretation problem or every occupied-tray planning case is resolved.
