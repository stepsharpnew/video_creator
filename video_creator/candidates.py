from __future__ import annotations

from .config import (
    CANDIDATE_CLUSTER_GAP_MS,
    CANDIDATE_NEXT_CONTEXT_MS,
    CANDIDATE_PREV_CONTEXT_MS,
    CANDIDATE_SETUP_MS,
    CANDIDATE_TAIL_MS,
    MAX_CLIP_SEC,
    MIN_CLIP_SEC,
    PAUSE_GAP_MS,
)
from .utils import filter_noise_words, fmt_time, log


# ── word-level helpers ──────────────────────────────────────


def _words_in_range(words: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    return filter_noise_words(
        [w for w in words if int(w["start"]) >= start_ms and int(w["end"]) <= end_ms]
    )


def _text_from_words(wds: list[dict]) -> str:
    return " ".join(str(w["text"]) for w in wds)


def _find_utterances(wds: list[dict]) -> list[dict]:
    """
    Split word sequence into utterances by pause gaps (>= PAUSE_GAP_MS).
    Each utterance: {start_ms, end_ms, text, word_count}.
    """
    if not wds:
        return []

    utts: list[dict] = []
    cur_words: list[dict] = [wds[0]]

    for w in wds[1:]:
        prev_end = int(cur_words[-1]["end"])
        cur_start = int(w["start"])
        if cur_start - prev_end >= PAUSE_GAP_MS:
            utts.append(_make_utterance(cur_words))
            cur_words = [w]
        else:
            cur_words.append(w)

    utts.append(_make_utterance(cur_words))
    return utts


def _make_utterance(wds: list[dict]) -> dict:
    return {
        "start_ms": int(wds[0]["start"]),
        "end_ms": int(wds[-1]["end"]),
        "text": _text_from_words(wds),
        "word_count": len(wds),
    }


def _find_pause_gaps(wds: list[dict]) -> list[dict]:
    """Return notable pauses (>= PAUSE_GAP_MS) between consecutive words."""
    gaps: list[dict] = []
    for i in range(1, len(wds)):
        prev_end = int(wds[i - 1]["end"])
        cur_start = int(wds[i]["start"])
        gap = cur_start - prev_end
        if gap >= PAUSE_GAP_MS:
            gaps.append({"at_ms": prev_end, "duration_ms": gap})
    return gaps


def _reaction_overlaps(reactions: list[dict], wds: list[dict]) -> list[dict]:
    """
    For each reaction, find words that overlap with the reaction window.
    Returns list of {reaction_start_ms, reaction_end_ms, overlapping_text}.
    """
    overlaps: list[dict] = []
    for r in reactions:
        rs, re = int(r["start_ms"]), int(r["end_ms"])
        ov_words = [w for w in wds if int(w["end"]) > rs and int(w["start"]) < re]
        overlaps.append({
            "reaction_start_ms": rs,
            "reaction_end_ms": re,
            "intensity": float(r["intensity"]),
            "overlapping_text": _text_from_words(ov_words) if ov_words else "",
        })
    return overlaps


# ── pipeline stages ─────────────────────────────────────────


def build_candidate_windows(
    reaction_zones: list[dict],
    words: list[dict],
) -> list[dict]:
    total_ms = int(words[-1]["end"]) if words else 0
    candidates: list[dict] = []

    for z in reaction_zones:
        r_start = int(z["start_ms"])
        r_end = int(z["end_ms"])
        intensity = float(z["intensity"])

        win_start = max(0, r_start - CANDIDATE_SETUP_MS)
        win_end = min(total_ms, r_end + CANDIDATE_TAIL_MS) if total_ms else r_end + CANDIDATE_TAIL_MS

        candidates.append({
            "win_start_ms": win_start,
            "win_end_ms": win_end,
            "reactions": [{"start_ms": r_start, "end_ms": r_end, "intensity": intensity}],
            "peak_intensity": intensity,
        })

    candidates.sort(key=lambda c: c["win_start_ms"])
    return candidates


def cluster_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    merged: list[dict] = [_copy_candidate(candidates[0])]

    for c in candidates[1:]:
        prev = merged[-1]
        if c["win_start_ms"] - prev["win_end_ms"] <= CANDIDATE_CLUSTER_GAP_MS:
            prev["win_end_ms"] = max(prev["win_end_ms"], c["win_end_ms"])
            prev["reactions"].extend(c["reactions"])
            prev["peak_intensity"] = max(prev["peak_intensity"], c["peak_intensity"])
        else:
            merged.append(_copy_candidate(c))

    return merged


def enrich_candidates(candidates: list[dict], words: list[dict]) -> list[dict]:
    """
    For each clustered candidate, build a rich metadata packet:
      - utterances, pause_gaps, reaction_overlaps
      - previous_context / next_context
      - word_count, estimated_wpm
      - reaction_at_ms (timestamp of strongest reaction)
    """
    total_ms = int(words[-1]["end"]) if words else 0
    enriched: list[dict] = []

    for c in candidates:
        win_start = c["win_start_ms"]
        win_end = c["win_end_ms"]
        duration_ms = win_end - win_start

        # clamp to [MIN_CLIP_SEC, MAX_CLIP_SEC]
        if duration_ms < MIN_CLIP_SEC * 1000:
            expand = (MIN_CLIP_SEC * 1000 - duration_ms) // 2
            win_start = max(0, win_start - expand)
            win_end = win_start + MIN_CLIP_SEC * 1000
            duration_ms = win_end - win_start
        if duration_ms > MAX_CLIP_SEC * 1000:
            win_end = win_start + MAX_CLIP_SEC * 1000
            duration_ms = MAX_CLIP_SEC * 1000

        clip_words = _words_in_range(words, win_start, win_end)
        if not clip_words:
            continue

        utterances = _find_utterances(clip_words)
        pause_gaps = _find_pause_gaps(clip_words)
        overlaps = _reaction_overlaps(c["reactions"], clip_words)
        transcript_chunk = _text_from_words(clip_words)
        word_count = len(clip_words)
        duration_sec = max(duration_ms / 1000, 0.1)
        estimated_wpm = round(word_count / (duration_sec / 60))

        # strongest reaction timestamp
        strongest = max(c["reactions"], key=lambda r: float(r["intensity"]))
        reaction_at_ms = int(strongest["start_ms"])

        # prev / next context snippets
        prev_start = max(0, win_start - CANDIDATE_PREV_CONTEXT_MS)
        prev_words = _words_in_range(words, prev_start, win_start)
        previous_context = _text_from_words(prev_words) if prev_words else ""

        next_end = min(total_ms, win_end + CANDIDATE_NEXT_CONTEXT_MS) if total_ms else win_end + CANDIDATE_NEXT_CONTEXT_MS
        next_words = _words_in_range(words, win_end, next_end)
        next_context = _text_from_words(next_words) if next_words else ""

        enriched.append({
            "win_start_ms": win_start,
            "win_end_ms": win_end,
            "reactions": c["reactions"],
            "peak_intensity": c["peak_intensity"],
            "reaction_at_ms": reaction_at_ms,
            "transcript_chunk": transcript_chunk,
            "utterances": utterances,
            "pause_gaps": pause_gaps,
            "reaction_overlaps": overlaps,
            "previous_context_5s": previous_context,
            "next_context_3s": next_context,
            "word_count": word_count,
            "estimated_wpm": estimated_wpm,
        })

    enriched.sort(key=lambda c: c["peak_intensity"], reverse=True)

    log("CAND", f"Кластеров-кандидатов: {len(enriched)}")
    for i, c in enumerate(enriched[:15], 1):
        dur = (c["win_end_ms"] - c["win_start_ms"]) // 1000
        n_react = len(c["reactions"])
        log(
            "CAND",
            f"  {i:2}. [{fmt_time(c['win_start_ms'])}–{fmt_time(c['win_end_ms'])}] "
            f"{dur}с  реакций: {n_react}  сила: {c['peak_intensity']:.0%}  "
            f"слов: {c['word_count']}  wpm: {c['estimated_wpm']}",
        )

    return enriched


def _copy_candidate(c: dict) -> dict:
    return {
        "win_start_ms": c["win_start_ms"],
        "win_end_ms": c["win_end_ms"],
        "reactions": list(c["reactions"]),
        "peak_intensity": c["peak_intensity"],
    }
