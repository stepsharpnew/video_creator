from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | os.PathLike | None = None) -> None:
    """
    Minimal .env loader without dependencies.
    Loads KEY=VALUE lines into os.environ (does not override existing vars).
    """
    path = Path(dotenv_path) if dotenv_path is not None else (Path(__file__).resolve().parent.parent / ".env")
    if not path.is_file():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

