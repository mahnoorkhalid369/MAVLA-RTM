"""Nested wall-clock spans. Evaluation assigns overlap to the innermost operation."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

skill_state = ContextVar("skill_state", default=None)


@contextmanager
def span(log, category, label, **fields):
    start = time.monotonic()
    ok = False
    try:
        yield
        ok = True
    finally:
        end = time.monotonic()
        if log is not None:
            log("timing", category=category, label=label, state=skill_state.get(),
                start_s=start, end_s=end, seconds=end-start, ok=ok, **fields)


def timed(category):
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            with span(getattr(self, "log", None), category, fn.__name__):
                return fn(self, *args, **kwargs)
        return call
    return decorate
