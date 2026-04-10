from __future__ import annotations

"""
Rule-based spectral reaction classifier.

Operates purely on raw PCM samples (s16le) and a sample rate.
No external ML dependencies — uses only stdlib math.

Reaction types
--------------
laugh        — human laughter: irregular bursts, energy concentrated 1–4 kHz,
               fast amplitude modulation, duration typically 0.5–3 s.
applause     — clapping: broadband, sustained, moderate temporal regularity.
music_hit    — musical accent/stab: wide-band, very short, harmonic-ish envelope.
crowd_noise  — general crowd noise that doesn't fit the above patterns.
silence      — below noise floor, probably a transition artefact.

Strategy: compute per-band energy ratios + temporal features on the zone's samples.
"""

import array
import math
import subprocess
from typing import Literal

ReactionType = Literal["laugh", "applause", "music_hit", "crowd_noise", "silence"]

# Frequency band edges (Hz) for the 8 kHz mono signal → max freq 4 kHz
# We split into 4 bands:  0–500, 500–1500, 1500–3000, 3000–4000
_BAND_EDGES_HZ = [(0, 500), (500, 1500), (1500, 3000), (3000, 4000)]


def classify_reaction_zone(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    sample_rate: int = 8000,
) -> tuple[ReactionType, float]:
    """
    Classify the audio content in [start_ms, end_ms] of audio_path.
    Returns (reaction_type, confidence 0..1).
    """
    samples = _load_samples(audio_path, start_ms, end_ms, sample_rate)
    if not samples:
        return "silence", 1.0

    features = _extract_features(samples, sample_rate)
    return _classify(features)


def classify_zones_bulk(
    audio_path: str,
    zones: list[dict],
    sample_rate: int = 8000,
) -> list[dict]:
    """
    Load the entire audio once and classify all zones.
    More efficient than calling classify_reaction_zone per zone.
    Returns zones with 'reaction_type' and 'type_confidence' fields added.
    """
    all_samples = _load_all_samples(audio_path, sample_rate)
    if not all_samples:
        for z in zones:
            z["reaction_type"] = "silence"
            z["type_confidence"] = 1.0
        return zones

    result: list[dict] = []
    for z in zones:
        s_start = int(z["start_ms"] * sample_rate / 1000)
        s_end = int(z["end_ms"] * sample_rate / 1000)
        chunk = all_samples[s_start : s_end]
        if not chunk:
            z = dict(z, reaction_type="silence", type_confidence=1.0)
        else:
            features = _extract_features(chunk, sample_rate)
            rtype, conf = _classify(features)
            z = dict(z, reaction_type=rtype, type_confidence=round(conf, 2))
        result.append(z)
    return result


# ── feature extraction ───────────────────────────────────────


class _Features:
    energy: float          # mean absolute amplitude (normalised 0..1)
    band_ratios: list[float]  # energy fraction per band (4 bands, sum ≈ 1)
    burst_rate: float      # short-term energy bursts per second (laugh proxy)
    centroid: float        # spectral centroid in Hz (bright = high)
    duration_ms: int


def _extract_features(samples: list[int], sr: int) -> _Features:
    n = len(samples)
    f = _Features()
    f.duration_ms = int(n * 1000 / sr)

    # ── global energy ──
    mean_abs = sum(abs(x) for x in samples) / n if n else 0
    f.energy = mean_abs / 32768.0

    # ── spectral bands via simple DFT on a short window ──
    # Use up to 4096 samples centred in the zone for spectral analysis
    win_size = min(4096, n)
    offset = (n - win_size) // 2
    window = samples[offset : offset + win_size]

    fft_mag = _simple_fft_magnitudes(window, sr)
    band_energies = _band_energies(fft_mag, sr, win_size)
    total_e = sum(band_energies) or 1.0
    f.band_ratios = [e / total_e for e in band_energies]

    # ── spectral centroid (Hz) ──
    freq_step = sr / win_size
    num_bins = len(fft_mag)
    weighted_sum = sum(fft_mag[i] * (i * freq_step) for i in range(num_bins))
    mag_sum = sum(fft_mag) or 1.0
    f.centroid = weighted_sum / mag_sum

    # ── burst rate (amplitude modulation indicator, laugh proxy) ──
    # Divide signal into 50 ms frames, count frames that are "hot" (> 60% of zone peak)
    frame_len = max(1, int(sr * 0.05))
    peak = max(abs(x) for x in samples) or 1
    threshold = peak * 0.60
    hot_frames = 0
    total_frames = 0
    for i in range(0, n, frame_len):
        frame = samples[i : i + frame_len]
        if frame:
            total_frames += 1
            if max(abs(x) for x in frame) > threshold:
                hot_frames += 1
    # Burst rate = fraction of hot frames × total frames per second
    f.burst_rate = (hot_frames / total_frames) if total_frames else 0

    return f


def _classify(f: _Features) -> tuple[ReactionType, float]:
    # silence
    if f.energy < 0.02:
        return "silence", 0.95

    # band_ratios: [low 0–500Hz, low-mid 500–1500, mid-hi 1500–3000, hi 3000–4000]
    low, lo_mid, mid_hi, hi = f.band_ratios

    # ── music_hit ──
    # Short, very transient, high total energy, balanced spectrum with lo-mid emphasis
    if f.duration_ms < 1500 and f.energy > 0.15 and lo_mid > 0.30 and f.burst_rate > 0.7:
        conf = min(0.9, f.energy * 2 + lo_mid)
        return "music_hit", round(conf, 2)

    # ── laugh ──
    # Irregular bursts, strong mid-hi + hi bands, moderate duration, high burst_rate
    if f.burst_rate < 0.80 and mid_hi + hi > 0.45 and f.centroid > 1500:
        score = (mid_hi + hi) * 0.5 + (1 - f.burst_rate) * 0.3 + min(1.0, f.centroid / 3000) * 0.2
        if score > 0.55:
            return "laugh", round(min(score, 0.95), 2)

    # ── applause ──
    # Sustained, broadband (all bands fairly even), longer duration, steady
    band_evenness = 1 - max(f.band_ratios)  # how uniform distribution is
    if band_evenness > 0.50 and f.duration_ms > 1000 and f.burst_rate > 0.50:
        conf = band_evenness * 0.6 + f.burst_rate * 0.4
        return "applause", round(min(conf, 0.90), 2)

    # ── crowd_noise (fallback) ──
    return "crowd_noise", 0.60


# ── FFT helpers (no external deps) ──────────────────────────


def _simple_fft_magnitudes(samples: list[int], sr: int) -> list[float]:
    """
    Compute magnitude spectrum via Cooley-Tukey radix-2 FFT.
    Only positive frequencies (first N//2 bins).
    Falls back to DFT for non-power-of-2 sizes by zero-padding.
    """
    n = len(samples)
    if n == 0:
        return []

    # Zero-pad to next power of 2
    size = 1
    while size < n:
        size <<= 1
    padded = [float(samples[i]) if i < n else 0.0 for i in range(size)]

    spectrum = _fft_recursive(padded)
    # Return magnitudes for first half (positive frequencies only)
    return [math.sqrt(c.real ** 2 + c.imag ** 2) / size for c in spectrum[: size // 2]]


def _fft_recursive(x: list[float]) -> list[complex]:
    """Cooley-Tukey in-place recursive FFT (real input as floats)."""
    n = len(x)
    if n <= 1:
        return [complex(v) for v in x]
    even = _fft_recursive(x[::2])
    odd = _fft_recursive(x[1::2])
    factor = [math.e ** (-2j * math.pi * k / n) * odd[k % (n // 2)] for k in range(n // 2)]
    return [even[k] + factor[k] for k in range(n // 2)] + [even[k] - factor[k] for k in range(n // 2)]


def _band_energies(magnitudes: list[float], sr: int, fft_size: int) -> list[float]:
    """Sum magnitudes^2 in each band."""
    freq_step = sr / fft_size
    energies = [0.0] * len(_BAND_EDGES_HZ)
    for i, mag in enumerate(magnitudes):
        freq = i * freq_step
        for band_idx, (lo, hi) in enumerate(_BAND_EDGES_HZ):
            if lo <= freq < hi:
                energies[band_idx] += mag * mag
                break
    return energies


# ── audio loading helpers ────────────────────────────────────


def _load_samples(audio_path: str, start_ms: int, end_ms: int, sr: int) -> list[int]:
    duration_s = max(0, (end_ms - start_ms) / 1000)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ms / 1000),
        "-t", str(duration_s),
        "-i", audio_path,
        "-ar", str(sr),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return []
    buf = array.array("h")
    buf.frombytes(proc.stdout)
    return list(buf)


def _load_all_samples(audio_path: str, sr: int) -> list[int]:
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", str(sr),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return []
    buf = array.array("h")
    buf.frombytes(proc.stdout)
    return list(buf)
