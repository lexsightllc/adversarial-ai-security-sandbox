# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import json
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format logs as structured JSON for better parsing and analysis."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_record["stack_trace"] = record.stack_info

        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name", "levelname", "pathname", "lineno", "asctime",
                "message", "args", "exc_info", "exc_text", "stack_info",
                "levelno", "created", "msecs", "relativeCreated", "thread",
                "threadName", "process", "processName", "funcName", "filename",
                "module", "msg", "sinfo", "getMessage"
            ]:
                try:
                    log_record[key] = value
                except (TypeError, ValueError):
                    # Skip values that can't be serialized
                    pass

        return json.dumps(log_record, default=str)


class PlainFormatter(logging.Formatter):
    """Format logs as plain text for console output in development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as plain text."""
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)
        logger = record.name.ljust(25)
        message = record.getMessage()

        formatted = f"{timestamp} [{level}] {logger} - {message}"

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(service_name: str, log_dir: str = "/var/log/adversarial-sandbox") -> logging.Logger:
    """
    Configure logging for a service with both console and file output.

    Args:
        service_name: Name of the service for logging identification
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()  # json or plain

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Get logger
    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    if log_format == "json":
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
    else:
        formatter = PlainFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if os.getenv("LOG_TO_FILE", "false").lower() == "true":
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{service_name}.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),  # 10MB default
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ"))
        logger.addHandler(file_handler)

    return logger
