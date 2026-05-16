
import logging
import sys
from typing import Any

try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    _STRUCTLOG_AVAILABLE = False


def configure_logging(log_level: str = "INFO") -> None:

    level = getattr(logging, log_level.upper(), logging.INFO)

    if _STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            stream=sys.stdout,
        )


def get_logger(name: str) -> Any:

    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)
