"""I/O helpers for project data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write dictionaries to a JSONL file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_text_lines(path: str | Path) -> list[str]:
    """Read a UTF-8 text file as a list of lines without newline characters."""
    return Path(path).read_text(encoding="utf-8").splitlines()


def write_text_lines(lines: Iterable[str], path: str | Path) -> None:
    """Write text lines to a UTF-8 file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def write_json(data: dict, path: str | Path) -> None:
    """Write a dictionary as pretty JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
