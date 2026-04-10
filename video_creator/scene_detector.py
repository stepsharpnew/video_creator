from __future__ import annotations

"""
Scene boundary detector using FFmpeg's built-in scene change filter.

FFmpeg selects=`gt(scene,THRESHOLD)` outputs one frame per detected cut.
We parse the pts_time from stderr/stdout metadata and return a sorted list
of {at_ms, score} dicts.

Usage in the pipeline:
    scenes = detect_scene_boundaries(video_path)
    # Then pass `scenes` to build_candidate_windows so that
    # window edges snap to the nearest scene cut instead of fixed offsets.
"""

import re
import subprocess

from .config import SCENE_THRESHOLD
from .utils import log


def detect_scene_boundaries(video_path: str) -> list[dict]:
    """
    Run FFmpeg scene-change detection on *video_path*.
    Returns list[{at_ms: int, score: float}] sorted by time.
    Always includes a boundary at t=0 (start of file).
    """
    log("SCENE", "Определяю монтажные границы (scene cuts)...")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',metadata=print:file=-",
        "-an",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    boundaries: list[dict] = [{"at_ms": 0, "score": 1.0}]

    # FFmpeg metadata=print outputs lines like:
    #   lavfi.scene_score=0.453421
    #   pts_time=12.344000
    current_score: float | None = None
    current_pts: float | None = None

    combined = proc.stderr + proc.stdout
    for line in combined.splitlines():
        line = line.strip()
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m:
            current_score = float(m.group(1))
            continue
        m = re.search(r"pts_time=([0-9.]+)", line)
        if m:
            current_pts = float(m.group(1))

        if current_score is not None and current_pts is not None:
            boundaries.append({
                "at_ms": int(current_pts * 1000),
                "score": round(current_score, 3),
            })
            current_score = None
            current_pts = None

    boundaries.sort(key=lambda b: b["at_ms"])
    boundaries = _deduplicate(boundaries, min_gap_ms=500)

    log("SCENE", f"Найдено {len(boundaries)} монтажных границ")
    for b in boundaries[:10]:
        log("SCENE", f"  {_fmt_ms(b['at_ms'])}  score={b['score']:.3f}")

    return boundaries


def find_nearest_boundary_before(boundaries: list[dict], at_ms: int, max_look_back_ms: int) -> int | None:
    """
    Find the latest scene boundary that is:
      - before *at_ms*
      - within *max_look_back_ms* distance

    Returns the boundary's at_ms, or None if nothing qualifies.
    """
    best: int | None = None
    for b in reversed(boundaries):
        bms = b["at_ms"]
        if bms >= at_ms:
            continue
        if at_ms - bms <= max_look_back_ms:
            best = bms
            break
    return best


def find_nearest_boundary_after(boundaries: list[dict], at_ms: int, max_look_ahead_ms: int) -> int | None:
    """
    Find the earliest scene boundary that is:
      - after *at_ms*
      - within *max_look_ahead_ms* distance

    Returns the boundary's at_ms, or None if nothing qualifies.
    """
    for b in boundaries:
        bms = b["at_ms"]
        if bms <= at_ms:
            continue
        if bms - at_ms <= max_look_ahead_ms:
            return bms
    return None


# ── helpers ──────────────────────────────────────────────────


def _deduplicate(boundaries: list[dict], min_gap_ms: int) -> list[dict]:
    """Keep only boundaries separated by at least min_gap_ms."""
    result: list[dict] = []
    for b in boundaries:
        if not result or b["at_ms"] - result[-1]["at_ms"] >= min_gap_ms:
            result.append(b)
    return result


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"
