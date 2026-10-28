"""Named destinations for measurements collected outside timed operations."""

import threading


class LoggerRegistry:
    """Route completed measurement rows to independently configured loggers.

    Register destinations before processing requests. Omitted columns are fixed
    atomically by the first dictionary write. Missing or disabled names
    silently skip writes. Schemas stay fixed for each registered destination.
    """

    def __init__(self):
        self._entries = {}
        self._lock = threading.RLock()

    def register(self, name, logger, *, columns=None):
        """Register an existing client/server file logger with a fixed schema."""
        columns = list(columns) if columns is not None else list(logger.titles)
        if not name or (columns and len(set(columns)) != len(columns)):
            raise ValueError("A name and unique columns are required")
        with self._lock:
            if name in self._entries:
                raise ValueError(f"Logger {name!r} is already registered")
            if any(entry["logger"] is logger for entry in self._entries.values()):
                raise ValueError("Each purpose requires its own logger instance")
            if logger.titles and list(logger.titles) != columns:
                raise ValueError("Logger schema does not match registered columns")
            if columns:
                logger.set_title(columns)
            self._entries[name] = {
                "logger": logger, "columns": columns, "enabled": True,
            }
        return logger

    def register_legacy(self, **destinations):
        """Adapt older constructor arguments; explicit registry entries win."""
        with self._lock:
            for name, logger in destinations.items():
                if logger is not None and name not in self._entries:
                    self.register(name, logger)

    def get(self, name):
        """Return an enabled destination, or None."""
        with self._lock:
            entry = self._entries.get(name)
            return entry["logger"] if entry and entry["enabled"] else None

    def disable(self, name):
        self._set_enabled(name, False)

    def enable(self, name):
        self._set_enabled(name, True)

    def _set_enabled(self, name, enabled):
        with self._lock:
            self._entries[name]["enabled"] = enabled

    def write(self, name, measurements):
        """Write a completed row; call after stopping measurement timers."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None or not entry["enabled"]:
                return
            logger = entry["logger"]
            if not entry["columns"]:
                if not isinstance(measurements, dict) or not measurements:
                    raise ValueError("First measurement must be a nonempty dictionary")
                entry["columns"] = list(measurements)
                logger.set_title(entry["columns"])
            if list(logger.titles) != entry["columns"]:
                raise ValueError("Registered logger schema changed")
            logger.log_content(measurements)
