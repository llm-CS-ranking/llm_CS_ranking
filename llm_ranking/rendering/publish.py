from __future__ import annotations

import shutil
from pathlib import Path


def write_text_to_many(tex: str, paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex, encoding="utf-8")


def copy_to_many(source: Path, destinations: list[Path]) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

