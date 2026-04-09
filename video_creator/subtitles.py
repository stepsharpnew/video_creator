from __future__ import annotations

from .config import (
    PAUSE_GAP_MS,
    SUB_MAX_CHARS_PER_LINE,
    SUB_MAX_DURATION_MS,
    SUB_MAX_LINES,
    SUB_MIN_DURATION_MS,
    SUB_TARGET_WPM,
)
from .utils import filter_noise_words

# ── Russian punctuation that signals a syntactic boundary ──────────
_CLAUSE_ENDS = frozenset(".,!?;:—–…")
_SENTENCE_ENDS = frozenset(".!?…")


def make_srt(words: list[dict], start_ms: int, end_ms: int) -> str:
    """
    Build an .srt string for [start_ms, end_ms] using pause-aware + syntactic segmentation.

    Strategy (priority order):
      1. Split on pauses >= PAUSE_GAP_MS between words.
      2. Within a pause-segment, split on sentence-ending punctuation (.!?…).
      3. If a cue still exceeds max duration / max chars, split on clause punctuation (,;:—).
      4. Last resort: split by WPM-based word count.
    Each cue is then line-wrapped respecting SUB_MAX_CHARS_PER_LINE and SUB_MAX_LINES.
    Timing is clamped to [SUB_MIN_DURATION_MS, SUB_MAX_DURATION_MS].
    """
    clip_words = filter_noise_words(
        [w for w in words if int(w["start"]) >= start_ms and int(w["end"]) <= end_ms]
    )
    if not clip_words:
        return ""

    # Phase 1: split into pause-delimited groups
    groups = _split_by_pauses(clip_words)

    # Phase 2: further split groups that are too long
    cues: list[list[dict]] = []
    for grp in groups:
        cues.extend(_split_group(grp))

    # Phase 3: render SRT
    entries: list[str] = []
    for idx, cue_words in enumerate(cues, 1):
        if not cue_words:
            continue
        s = max(0, int(cue_words[0]["start"]) - start_ms)
        e = max(s + SUB_MIN_DURATION_MS, int(cue_words[-1]["end"]) - start_ms)
        text = _wrap_lines(" ".join(str(w["text"]) for w in cue_words))
        entries.append(f"{idx}\n{_ms_to_srt(s)} --> {_ms_to_srt(e)}\n{text}\n")

    return "\n".join(entries)


# ── internal helpers ───────────────────────────────────────────────


def _split_by_pauses(wds: list[dict]) -> list[list[dict]]:
    """Split word list on inter-word gaps >= PAUSE_GAP_MS."""
    if not wds:
        return []
    groups: list[list[dict]] = [[wds[0]]]
    for w in wds[1:]:
        prev_end = int(groups[-1][-1]["end"])
        if int(w["start"]) - prev_end >= PAUSE_GAP_MS:
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def _split_group(grp: list[dict]) -> list[list[dict]]:
    """
    Break a pause-segment into subtitle cues respecting duration and char limits.
    Prefers splitting at sentence ends, then clause punctuation, then by word count.
    """
    if _cue_ok(grp):
        return [grp]

    # Try sentence-end splits first
    parts = _split_at_punctuation(grp, _SENTENCE_ENDS)
    if len(parts) > 1:
        result: list[list[dict]] = []
        for p in parts:
            result.extend(_split_group(p))
        return result

    # Try clause-boundary splits
    parts = _split_at_punctuation(grp, _CLAUSE_ENDS)
    if len(parts) > 1:
        result = []
        for p in parts:
            result.extend(_split_group(p))
        return result

    # Fallback: split by target word count (WPM-based)
    return _split_by_word_count(grp)


def _cue_ok(wds: list[dict]) -> bool:
    if not wds:
        return True
    duration = int(wds[-1]["end"]) - int(wds[0]["start"])
    text_len = sum(len(str(w["text"])) for w in wds) + len(wds) - 1
    max_chars = SUB_MAX_CHARS_PER_LINE * SUB_MAX_LINES
    return duration <= SUB_MAX_DURATION_MS and text_len <= max_chars


def _split_at_punctuation(wds: list[dict], punct_set: frozenset) -> list[list[dict]]:
    """Split at words whose text ends with punctuation from punct_set."""
    split_indices: list[int] = []
    for i, w in enumerate(wds[:-1]):
        text = str(w["text"]).rstrip()
        if text and text[-1] in punct_set:
            split_indices.append(i + 1)

    if not split_indices:
        return [wds]

    parts: list[list[dict]] = []
    prev = 0
    for si in split_indices:
        if si > prev:
            parts.append(wds[prev:si])
        prev = si
    if prev < len(wds):
        parts.append(wds[prev:])

    return parts


def _split_by_word_count(wds: list[dict]) -> list[list[dict]]:
    """Split into chunks whose duration ≈ SUB_MAX_DURATION_MS based on WPM."""
    words_per_cue = max(2, int(SUB_TARGET_WPM * (SUB_MAX_DURATION_MS / 60_000)))
    parts: list[list[dict]] = []
    for i in range(0, len(wds), words_per_cue):
        chunk = wds[i : i + words_per_cue]
        if chunk:
            parts.append(chunk)
    return parts


def _wrap_lines(text: str) -> str:
    """
    Wrap text into at most SUB_MAX_LINES lines, each ≤ SUB_MAX_CHARS_PER_LINE.
    Tries to split at a balanced midpoint.
    """
    if len(text) <= SUB_MAX_CHARS_PER_LINE:
        return text

    if SUB_MAX_LINES < 2:
        return text[:SUB_MAX_CHARS_PER_LINE * 2]

    tokens = text.split()
    if len(tokens) <= 1:
        return text

    best_i, best_score = 1, 10**9
    for i in range(1, len(tokens)):
        left = " ".join(tokens[:i])
        right = " ".join(tokens[i:])
        score = abs(len(left) - len(right))
        if max(len(left), len(right)) > SUB_MAX_CHARS_PER_LINE * 1.3:
            score += 1000
        if score < best_score:
            best_score = score
            best_i = i

    return " ".join(tokens[:best_i]) + "\n" + " ".join(tokens[best_i:])


def _ms_to_srt(ms: int) -> str:
    ms = max(0, ms)
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1000
    r = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{r:03d}"
