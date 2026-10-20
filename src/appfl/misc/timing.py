"""Scoped latency collection without passing state through intermediate calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional


@dataclass
class LatencyContext:
    """Elapsed seconds accumulated per key, retained after collection."""

    total_seconds: Dict[str, float] = field(default_factory=dict)


_current_latency_context: ContextVar[Optional[LatencyContext]] = ContextVar(
    "appfl_latency_context", default=None
)


@contextmanager
def collect_latencies(enabled: bool = True):
    """Collect measurements in this execution context; nested scopes are isolated.

    Create one scope per request. The scope follows synchronous calls and async
    tasks, but is not automatically propagated into newly created threads.
    When disabled, measurements are no-ops, including inside an outer scope.
    """
    context = LatencyContext() if enabled else None
    token = _current_latency_context.set(context)
    try:
        yield context
    finally:
        _current_latency_context.reset(token)


@contextmanager
def measure_latency(key: str):
    """Add a block's elapsed seconds to the named key in the active collection."""
    context = _current_latency_context.get()
    if context is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        context.total_seconds[key] = context.total_seconds.get(key, 0.0) + elapsed
