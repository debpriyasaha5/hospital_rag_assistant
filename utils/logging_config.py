import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIRECTORY = PROJECT_ROOT / "logs"
LOG_PREFIX = "hospital-rag-assistant"
MAX_LOG_BYTES = 5_000_000
_HANDLER_MARKER = "hospital_rag_file_handler"
_CONSOLE_HANDLER_MARKER = "hospital_rag_console_handler"


class TimestampedRotatingFileHandler(RotatingFileHandler):
    def __init__(self, directory, prefix, max_bytes, encoding="utf-8"):
        self.directory = Path(directory)
        self.prefix = prefix
        self.directory.mkdir(parents=True, exist_ok=True)
        super().__init__(
            self._next_log_path(),
            maxBytes=max_bytes,
            backupCount=0,
            encoding=encoding,
        )

    def _next_log_path(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.directory / f"{self.prefix}_{timestamp}.log"
        suffix = 1
        while path.exists():
            path = self.directory / f"{self.prefix}_{timestamp}_{suffix}.log"
            suffix += 1
        return str(path.resolve())

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
        if self.stream.tell() == 0:
            return False
        return super().shouldRollover(record)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        self.baseFilename = self._next_log_path()
        if not self.delay:
            self.stream = self._open()


def configure_logging(logger_name=None):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in root_logger.handlers):
        file_handler = TimestampedRotatingFileHandler(
            LOG_DIRECTORY,
            LOG_PREFIX,
            max_bytes=MAX_LOG_BYTES,
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        root_logger.addHandler(file_handler)

    if not any(
        getattr(handler, _CONSOLE_HANDLER_MARKER, False)
        for handler in root_logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        setattr(console_handler, _CONSOLE_HANDLER_MARKER, True)
        root_logger.addHandler(console_handler)

    return logging.getLogger(logger_name)