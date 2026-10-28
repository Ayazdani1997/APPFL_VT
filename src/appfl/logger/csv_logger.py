import csv
import os
import threading
from typing import Dict, List, Optional, Union


class CSVLoggerMixin:
    """Add structured CSV output to an APPFL file logger."""

    def _init_csv_logger(self, csv_filepath: Optional[str]) -> None:
        self._csv_lock = threading.RLock()
        self.csv_filepath = csv_filepath
        self.titles = getattr(self, "titles", [])
        if csv_filepath is not None:
            os.makedirs(os.path.dirname(csv_filepath) or ".", exist_ok=True)

    def set_title(self, titles: List) -> None:
        if not self.titles:
            self.titles = list(titles)

    def _write_csv_row(self, contents: List) -> None:
        if self.csv_filepath is None:
            return

        with self._csv_lock:
            file_exists = os.path.exists(self.csv_filepath)
            file_has_content = file_exists and os.path.getsize(self.csv_filepath) > 0
            with open(self.csv_filepath, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                if not file_has_content:
                    writer.writerow(self.titles)
                writer.writerow(contents)

    def _normalize_content(self, contents: Union[Dict, List]) -> List:
        if not isinstance(contents, (dict, list)):
            raise ValueError("Contents must be a dictionary or list")
        if not self.titles:
            raise ValueError("Titles must be set before logging content")

        if isinstance(contents, dict):
            unknown_titles = set(contents) - set(self.titles)
            if unknown_titles:
                title = next(iter(unknown_titles))
                raise ValueError(f"Title {title} is not defined")
            return [contents.get(key, "") for key in self.titles]

        if len(contents) != len(self.titles):
            raise ValueError("Contents and titles must have the same length")
        return contents

    def get_csv_filepath(self) -> Optional[str]:
        """Return the CSV path, or ``None`` when file logging is disabled."""
        return self.csv_filepath
