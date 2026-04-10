from __future__ import annotations

import array
import subprocess

from .config import (
    ANALYSIS_HOP_SEC,
    ANALYSIS_SAMPLE_RATE,
    ANALYSIS_WINDOW_SEC,
    REACTION_MERGE_GAP,
    REACTION_THRESHOLD,
)
from .reaction_classifier import classify_zones_bulk
from .utils import fmt_time, log


def analyze_audience_reactions(audio_path: str) -> list[dict]:
    """
    Detect loud reaction zones by RMS/mean-abs analysis, then classify
    each zone's type (laugh / applause / music_hit / crowd_noise / silence).
    """
    log("REACT", "Анализирую аудио — ищу реакции зала...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg: не удалось декодировать аудио для анализа")

    samples = array.array("h")
    samples.frombytes(proc.stdout)

    sr = ANALYSIS_SAMPLE_RATE
    win = int(sr * ANALYSIS_WINDOW_SEC)
    hop = int(sr * ANALYSIS_HOP_SEC)

    energies: list[tuple[int, float]] = []
    for i in range(0, len(samples) - win, hop):
        chunk = samples[i : i + win]
        mean_abs = sum(abs(x) for x in chunk) / len(chunk)
        t_ms = int(i * 1000 / sr)
        energies.append((t_ms, mean_abs))

    if not energies:
        log("REACT", "Аудио слишком короткое для анализа")
        return []

    values = sorted(v for _, v in energies)
    median = values[len(values) // 2]

    threshold = median * REACTION_THRESHOLD
    peaks = [(t, v) for t, v in energies if v > threshold]

    if len(peaks) < 3:
        threshold = median * 1.3
        peaks = [(t, v) for t, v in energies if v > threshold]

    if not peaks:
        log("REACT", "Не найдено выраженных пиков громкости")
        return []

    max_val = max(v for _, v in peaks)

    zones: list[dict] = []
    z_start, z_end, z_max = peaks[0][0], peaks[0][0], peaks[0][1]

    for t, v in peaks[1:]:
        if t - z_end <= REACTION_MERGE_GAP:
            z_end = t
            z_max = max(z_max, v)
        else:
            zones.append(_make_zone(z_start, z_end, z_max, max_val))
            z_start, z_end, z_max = t, t, v

    zones.append(_make_zone(z_start, z_end, z_max, max_val))
    zones.sort(key=lambda z: z["intensity"], reverse=True)

    # Classify reaction types using spectral analysis
    log("REACT", "Классифицирую типы реакций...")
    zones = classify_zones_bulk(audio_path, zones, sample_rate=ANALYSIS_SAMPLE_RATE)

    log("REACT", f"Найдено {len(zones)} зон реакции зала (топ-15):")
    for i, z in enumerate(zones[:15], 1):
        dur = (z["end_ms"] - z["start_ms"]) // 1000
        rtype = z.get("reaction_type", "?")
        conf = z.get("type_confidence", 0)
        log(
            "REACT",
            f"  {i:2}. [{fmt_time(z['start_ms'])}] {dur}с  "
            f"сила: {z['intensity']:.0%}  тип: {rtype} ({conf:.0%})",
        )

    return zones


def reactions_to_text(reaction_zones: list[dict], limit: int = 30) -> str:
    lines: list[str] = []
    for z in reaction_zones[:limit]:
        lines.append(
            f"  [{fmt_time(int(z['start_ms']))} — {fmt_time(int(z['end_ms']))}]  "
            f"сила: {float(z['intensity']):.0%}"
        )
    return "\n".join(lines) if lines else "  (не найдено)"


def _make_zone(start: int, end: int, peak: float, max_val: float) -> dict:
    return {
        "start_ms": start,
        "end_ms": end + int(ANALYSIS_WINDOW_SEC * 1000),
        "intensity": round(peak / max_val, 2) if max_val > 0 else 0,
    }
