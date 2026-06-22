"""Centralized logging for the whole booth pipeline + API.

Call ``setup_logging()`` once at process start (the API does this in main.py,
the CLI does it in pdf_hybrid_pipeline.py). It installs:

  * a rotating FILE handler at DEBUG -> ``$LOG_DIR/booth.log`` (deep logs:
    timestamp, level, logger name, file:line, function, message). Rotates at
    10 MB, keeps 10 backups.
  * a console handler at the level given by ``$LOG_CONSOLE_LEVEL`` (default
    INFO) so the terminal stays readable while the file keeps everything.

LOG_DIR defaults to ``/data/logs`` so logs land on the bind-mounted ./data
(persisting on the host). Override with the LOG_DIR env var. A few chatty
third-party libraries are capped at WARNING so our own DEBUG lines dominate.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

LOG_DIR = os.getenv("LOG_DIR", "/data/logs")
LOG_FILE = os.getenv("LOG_FILE", "booth.log")
_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()
_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()

_FMT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d %(funcName)s() | %(message)s"
)

# Libraries whose DEBUG output would drown our own logs.
_NOISY = ("urllib3", "httpx", "httpcore", "PIL", "matplotlib", "asyncio")


def _resolve_log_dir() -> str:
    """First writable dir among: $LOG_DIR, ./logs, $TMPDIR/booth_logs."""
    import tempfile
    candidates = [LOG_DIR, os.path.join(os.getcwd(), "logs"),
                  os.path.join(tempfile.gettempdir(), "booth_logs")]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            testfile = os.path.join(d, ".write_test")
            with open(testfile, "w") as fh:
                fh.write("")
            os.remove(testfile)
            return d
        except Exception:
            continue
    return ""  # no writable dir -> console-only


def setup_logging(force: bool = False) -> str:
    """Configure root logging once. Returns the log file path (or '' if file
    logging was not possible, in which case logging is console-only)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return getattr(setup_logging, "_log_path", "")

    fmt = logging.Formatter(_FMT)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter below

    # Drop any handlers a prior basicConfig()/run may have installed.
    for h in list(root.handlers):
        root.removeHandler(h)

    log_dir = _resolve_log_dir()
    log_path = os.path.join(log_dir, LOG_FILE) if log_dir else ""
    if log_path:
        try:
            file_handler = RotatingFileHandler(
                log_path, maxBytes=10 * 1024 * 1024, backupCount=10,
                encoding="utf-8")
            file_handler.setLevel(getattr(logging, _FILE_LEVEL, logging.DEBUG))
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except Exception:
            log_path = ""

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, _CONSOLE_LEVEL, logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    setup_logging._log_path = log_path
    if log_path:
        logging.getLogger(__name__).info(
            "Logging initialised -> %s (file=%s, console=%s)",
            log_path, _FILE_LEVEL, _CONSOLE_LEVEL)
    else:
        logging.getLogger(__name__).warning(
            "No writable log dir; logging to console only (console=%s)",
            _CONSOLE_LEVEL)
    return log_path
