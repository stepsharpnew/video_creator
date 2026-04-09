from __future__ import annotations

import re

from .config import TRANSCRIPT_NOISE_PATTERNS

_NOISE_RE: re.Pattern | None = None


def _get_noise_re() -> re.Pattern:
    global _NOISE_RE
    if _NOISE_RE is None:
        escaped = [re.escape(p) for p in TRANSCRIPT_NOISE_PATTERNS]
        _NOISE_RE = re.compile(
            r"\[?\(?\b(?:" + "|".join(escaped) + r")\b\)?\]?",
            re.IGNORECASE,
        )
    return _NOISE_RE


def is_noise_word(text: str) -> bool:
    """Check if a word/phrase is a transcript artefact that should be stripped."""
    cleaned = text.strip().strip("[]()").strip()
    if not cleaned:
        return True
    return bool(_get_noise_re().fullmatch(cleaned))


def filter_noise_words(words: list[dict]) -> list[dict]:
    """Remove transcript artefact words from a word list."""
    return [w for w in words if not is_noise_word(str(w.get("text", "")))]


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"

