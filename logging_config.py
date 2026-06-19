import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _log_namer(default_name: str) -> str:
    """Rename rotated logs: app.log.2026-03-23 → app_2026-03-23.log"""
    # default_name = "/path/logs/app.log.2026-03-23"
    base = default_name.replace("app.log.", "app_")
    return base + ".log"


def setup_logging():
    """Configure root logger with daily rotating file + console handlers."""
    logger = logging.getLogger("paddler_ocr")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Daily rotating file handler (rotates at midnight, keep 90 days)
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        when="midnight",
        interval=1,
        backupCount=90,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = _log_namer
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the paddler_ocr namespace."""
    return logging.getLogger(f"paddler_ocr.{name}")
