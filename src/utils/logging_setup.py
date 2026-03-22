"""
Logging configuration.

Provides structured JSON logging for experiment tracking.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
        # Include any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                log_obj[key] = value
        return json.dumps(log_obj, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = False,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to write logs to a file.
        use_json: If True, use JSON formatter; else use human-readable formatter.
        logger_name: Name for the logger. Uses root logger if None.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        if use_json:
            ch.setFormatter(JSONFormatter())
        else:
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

        # File handler
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            if use_json:
                fh.setFormatter(JSONFormatter())
            else:
                fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a named logger."""
    return logging.getLogger(name)
