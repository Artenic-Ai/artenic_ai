"""Structured JSON logging with correlation-ID injection.

Call :func:`setup_logging` once at application startup to configure the
root logger with a JSON formatter and a filter that automatically
attaches the current request ID to every log record.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import traceback
from typing import Any

from artenic_ai_platform.middleware.correlation import get_request_id


class CorrelationIdFilter(logging.Filter):
    """Inject ``request_id`` into every log record.

    The value is read from the module-level
    :data:`~artenic_ai_platform.middleware.correlation.correlation_id_var`
    context variable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class PlatformJsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Output schema::

        {
            "timestamp": "2025-06-15T12:34:56.789012+00:00",
            "level": "INFO",
            "logger": "artenic_ai_platform.middleware.auth",
            "message": "Request authenticated",
            "request_id": "abc123",
            "exception": null
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_text: str | None = None
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = "".join(
                traceback.format_exception(*record.exc_info),
            )

        payload: dict[str, Any] = {
            "timestamp": (
                datetime.datetime.fromtimestamp(
                    record.created,
                    tz=datetime.UTC,
                ).isoformat()
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
            "exception": exc_text,
        }
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with JSON formatting.

    Parameters
    ----------
    level:
        Minimum log level (default ``logging.INFO``).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output ---------------
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(PlatformJsonFormatter())
    handler.addFilter(CorrelationIdFilter())

    root.addHandler(handler)
