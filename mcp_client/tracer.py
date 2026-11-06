import time
import uuid
from datetime import datetime


class TraceSpan:
    """
    Represents one operation inside a trace.

    Example:
        LLM tool selection
        MCP tool execution
        Guardrail
    """

    def __init__(self, trace_id: str, operation: str):
        self.trace_id = trace_id
        self.span_id = str(uuid.uuid4())
        self.operation = operation

        self.start_time = None
        self.end_time = None
        self.latency = None

        self.status = "running"
        self.metadata = {}

    def start(self):
        """Start the span."""

        self.start_time = time.perf_counter()

        return self

    def end(
        self,
        status: str = "success",
        metadata: dict | None = None
    ):
        """Finish the span and calculate latency."""

        self.end_time = time.perf_counter()

        if self.start_time is not None:
            self.latency = (
                self.end_time - self.start_time
            )

        self.status = status

        if metadata:
            self.metadata.update(metadata)

        return self

    def to_dict(self) -> dict:
        """Convert span information into a dictionary."""

        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_seconds": self.latency,
            "status": self.status,
            "metadata": self.metadata,
        }


class Trace:
    """
    Represents one complete agent request.
    """

    def __init__(self):
        self.trace_id = str(uuid.uuid4())

        self.started_at = datetime.now()

        self.start_time = time.perf_counter()
        self.end_time = None

        self.total_latency = None

        self.spans = []

        self.status = "running"

    def start_span(self, operation: str) -> TraceSpan:
        """
        Create and start a new span.
        """

        span = TraceSpan(
            trace_id=self.trace_id,
            operation=operation
        )

        span.start()

        self.spans.append(span)

        return span

    def finish(self, status: str = "success"):
        """
        Finish the complete trace.
        """

        self.end_time = time.perf_counter()

        self.total_latency = (
            self.end_time - self.start_time
        )

        self.status = status

    def summary(self) -> dict:
        """
        Return complete trace information.
        """

        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "total_latency_seconds": self.total_latency,
            "status": self.status,
            "span_count": len(self.spans),
            "spans": [
                span.to_dict()
                for span in self.spans
            ],
        }

    def print_summary(self):
        """
        Print a readable trace summary.
        """

        print("\n" + "=" * 70)
        print("TRACE SUMMARY")
        print("=" * 70)

        print(f"Trace ID      : {self.trace_id}")
        print(f"Status        : {self.status}")
        print(
            f"Total Latency : "
            f"{self.total_latency:.4f} sec"
        )
        print(
            f"Span Count    : "
            f"{len(self.spans)}"
        )

        print("\nSpans")
        print("-" * 70)

        for index, span in enumerate(
            self.spans,
            start=1
        ):

            print(
                f"{index}. {span.operation}"
            )

            print(
                f"   Span ID : {span.span_id}"
            )

            print(
                f"   Status  : {span.status}"
            )

            if span.latency is not None:
                print(
                    f"   Latency : "
                    f"{span.latency:.4f} sec"
                )

            if span.metadata:
                print(
                    f"   Metadata: "
                    f"{span.metadata}"
                )

        print("=" * 70)