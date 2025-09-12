from typing import Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor


@pytest.fixture(scope="session")
def trace_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(trace_exporter: InMemorySpanExporter) -> None:
    trace_exporter.clear()


@pytest.fixture(autouse=True)
def instrument() -> Generator[AgnoInstrumentor, None, None]:
    instrumentor = AgnoInstrumentor()
    instrumentor.instrument()
    yield instrumentor
    instrumentor.uninstrument()
