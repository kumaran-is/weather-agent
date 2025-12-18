"""Structured Logging with Trace Context for Weather AI Agent.

Level 9: Comprehensive structured logging with:
- JSON format for log aggregation (Loki, ELK)
- Automatic trace context injection (trace_id, span_id)
- Request context propagation
- Log level filtering
- Performance-safe logging

Usage:
    >>> from backend.src.observability.logging import get_structured_logger
    >>> logger = get_structured_logger(__name__)
    >>> logger.info("Processing query", extra={"query": "weather in miami", "user_id": "123"})

Output (JSON format):
    {
        "timestamp": "2025-01-15T10:30:45.123Z",
        "level": "INFO",
        "logger": "backend.src.api.main",
        "message": "Processing query",
        "trace_id": "abc123def456",
        "span_id": "789xyz",
        "query": "weather in miami",
        "user_id": "123"
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

# Context variables for request-scoped data
request_context: ContextVar[dict[str, Any]] = ContextVar("request_context", default={})


class StructuredFormatter(logging.Formatter):
    """JSON formatter with trace context injection.

    Formats log records as JSON with automatic inclusion of:
    - OpenTelemetry trace context (trace_id, span_id)
    - Request context (user_id, session_id, request_id)
    - Standard log fields (timestamp, level, logger, message)
    """

    def __init__(
        self,
        include_trace_context: bool = True,
        include_request_context: bool = True,
    ):
        """Initialize structured formatter.

        Args:
            include_trace_context: Include trace_id and span_id from OpenTelemetry
            include_request_context: Include request context (user_id, etc.)
        """
        super().__init__()
        self.include_trace_context = include_trace_context
        self.include_request_context = include_request_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log structure
        log_dict: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace context if available and enabled
        if self.include_trace_context:
            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                log_dict["trace_id"] = format(span_context.trace_id, "032x")
                log_dict["span_id"] = format(span_context.span_id, "016x")

        # Add request context if available and enabled
        if self.include_request_context:
            ctx = request_context.get()
            if ctx:
                log_dict.update(ctx)

        # Add extra fields from log record
        if hasattr(record, "extra"):
            log_dict.update(record.extra)

        # Add any additional attributes passed via extra
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "extra",
                "message",
            }:
                if not key.startswith("_"):
                    log_dict[key] = value

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add source location for debugging
        if record.levelno >= logging.WARNING:
            log_dict["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_dict, default=str)


class StructuredLogger(logging.Logger):
    """Extended logger with structured logging support.

    Adds convenience methods for structured logging with automatic
    context propagation and trace injection.
    """

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with automatic context injection."""
        if extra is None:
            extra = {}

        # Merge with request context
        ctx = request_context.get()
        merged_extra = {**ctx, **extra}

        super()._log(level, msg, args, extra={"extra": merged_extra}, **kwargs)

    def info_structured(self, msg: str, **fields: Any) -> None:
        """Log INFO with structured fields.

        Args:
            msg: Log message
            **fields: Additional structured fields
        """
        self._log_with_context(logging.INFO, msg, (), extra=fields)

    def debug_structured(self, msg: str, **fields: Any) -> None:
        """Log DEBUG with structured fields."""
        self._log_with_context(logging.DEBUG, msg, (), extra=fields)

    def warning_structured(self, msg: str, **fields: Any) -> None:
        """Log WARNING with structured fields."""
        self._log_with_context(logging.WARNING, msg, (), extra=fields)

    def error_structured(self, msg: str, **fields: Any) -> None:
        """Log ERROR with structured fields."""
        self._log_with_context(logging.ERROR, msg, (), extra=fields)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger.

    Creates a logger configured for structured JSON output with
    trace context injection.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_structured_logger(__name__)
        >>> logger.info("Query received", extra={"query_type": "weather"})
    """
    # Set logger class
    logging.setLoggerClass(StructuredLogger)

    # Get logger
    logger = logging.getLogger(name)

    # Configure if not already done
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    return logger  # type: ignore


def add_trace_context(log_dict: dict[str, Any]) -> dict[str, Any]:
    """Add current trace context to a dictionary.

    Utility function to add trace_id and span_id to any dict,
    useful for custom logging or metrics.

    Args:
        log_dict: Dictionary to augment

    Returns:
        Dictionary with trace context added

    Example:
        >>> metrics = {"latency_ms": 150, "tokens": 500}
        >>> metrics_with_trace = add_trace_context(metrics)
        >>> # metrics_with_trace now includes trace_id and span_id
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        log_dict["trace_id"] = format(span_context.trace_id, "032x")
        log_dict["span_id"] = format(span_context.span_id, "016x")
    return log_dict


def set_request_context(
    user_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    **extra: Any,
) -> None:
    """Set request context for structured logging.

    Sets context variables that will be automatically included
    in all subsequent log messages within the same async context.

    Args:
        user_id: User identifier
        session_id: Session identifier
        request_id: Request identifier
        **extra: Additional context fields

    Example:
        >>> set_request_context(user_id="123", session_id="abc", tier="premium")
        >>> logger.info("Processing")  # Will include user_id, session_id, tier
    """
    ctx = {
        k: v
        for k, v in {
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            **extra,
        }.items()
        if v is not None
    }
    request_context.set(ctx)


def clear_request_context() -> None:
    """Clear request context after request completion."""
    request_context.set({})


# Configure root logger for structured output
def configure_structured_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure root logger for structured output.

    Should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format (True) or standard format (False)

    Example:
        >>> # In application startup
        >>> configure_structured_logging(level="INFO", json_format=True)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured handler
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)
