"""OpenTelemetry tracing — one span per graph node execution, exported OTLP.

Off by default; turned on by OTEL_EXPORTER_OTLP_ENDPOINT (compose sets it to the
Jaeger container). Spans carry the run_id plus whatever the node's events report
(tokens, verdicts, task ids), so the Jaeger flame view answers "which agent did
what, how long, at what token cost" — the observability requirement.

Honest scope note: parallel executor fan-out runs in worker threads, so those spans
may attach to the run root rather than nesting under their scheduler cycle. Fixing
that requires explicit context propagation through LangGraph's executor — noted as
future work; the per-agent timing/cost data is complete either way.
"""
from __future__ import annotations

import functools
import os

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        _tracer = False
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": "analyst"})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("analyst")
    except ImportError:
        _tracer = False  # otel extra not installed; tracing silently off
    return _tracer


def traced_node(name: str):
    """Wrap a LangGraph node: span per invocation, attributes from its events."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state, *a, **kw):
            tracer = _get_tracer()
            if not tracer:
                return fn(state, *a, **kw)
            with tracer.start_as_current_span(f"node.{name}") as span:
                if isinstance(state, dict):
                    span.set_attribute("run_id", str(state.get("run_id", "")))
                out = fn(state, *a, **kw)
                for e in (out or {}).get("events", []):
                    for k in ("tokens", "input_tokens", "output_tokens", "verdict",
                              "task", "status", "dispatch"):
                        if k in e:
                            span.set_attribute(f"event.{k}", str(e[k]))
                return out

        return wrapper

    return deco
