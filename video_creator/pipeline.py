from __future__ import annotations

import json
import os

from .audio import extract_audio
from .candidates import build_candidate_windows, cluster_candidates, enrich_candidates
from .reactions import analyze_audience_reactions
from .render_ffmpeg import process_clip
from .scoring import select_top_highlights
from .select_highlights_openai import score_candidates
from .transcribe_assemblyai import poll_transcription, start_transcription, upload_to_assemblyai
from .utils import log


def run_pipeline(*, video_path: str, output_dir: str, count: int, skip_asr_path: str | None = None) -> list[str]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "_audio.mp3")

    # Step 0: extract audio
    if not os.path.exists(audio_path):
        extract_audio(video_path, audio_path)
    else:
        log("SKIP", f"Аудио уже есть: {audio_path}")

    # Step 1: audience reaction zones
    reactions_path = os.path.join(output_dir, "reactions.json")
    if os.path.exists(reactions_path):
        log("SKIP", f"Загружаю кэш реакций: {reactions_path}")
        with open(reactions_path, encoding="utf-8") as f:
            reaction_zones = json.load(f)
    else:
        reaction_zones = analyze_audience_reactions(audio_path)
        with open(reactions_path, "w", encoding="utf-8") as f:
            json.dump(reaction_zones, f, ensure_ascii=False, indent=2)
        log("SAVE", f"Реакции сохранены: {reactions_path}")

    # Step 2: transcript
    transcript_path = skip_asr_path or os.path.join(output_dir, "transcript.json")
    if os.path.exists(transcript_path):
        log("SKIP", f"Загружаю транскрипт: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
    else:
        upload_url = upload_to_assemblyai(audio_path)
        transcript_id = start_transcription(upload_url)
        transcript = poll_transcription(transcript_id)
        t_path = os.path.join(output_dir, "transcript.json")
        with open(t_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        log("SAVE", f"Транскрипт сохранён: {t_path}")

    words = transcript.get("words", [])
    total_ms = int(words[-1]["end"]) if words else 0

    # Step 3: build candidate windows → cluster → enrich
    raw_candidates = build_candidate_windows(reaction_zones, words)
    clustered = cluster_candidates(raw_candidates)
    candidates = enrich_candidates(clustered, words)

    cand_path = os.path.join(output_dir, "candidates.json")
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2, default=str)
    log("SAVE", f"Кандидаты сохранены: {cand_path}")

    if not candidates:
        log("WARN", "Не найдено кандидатов — нечего отправлять в GPT.")
        return []

    # Step 4: GPT scores every candidate (no selection — just evaluation)
    scored = score_candidates(candidates)

    scored_path = os.path.join(output_dir, "scored.json")
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    log("SAVE", f"Оценки сохранены: {scored_path}")

    # Step 5: deterministic selection — score, penalize, greedy dedup, top-N
    highlights = select_top_highlights(scored, candidates, total_ms, count)

    h_path = os.path.join(output_dir, "highlights.json")
    with open(h_path, "w", encoding="utf-8") as f:
        json.dump(highlights, f, ensure_ascii=False, indent=2)
    log("SAVE", f"Моменты сохранены: {h_path}")

    if not highlights:
        log(
            "WARN",
            "Не осталось валидных клипов после отбора. "
            "Попробуй снизить порог реакции (REACTION_THRESHOLD) или увеличить --count.",
        )
        return []

    # Step 6: render clips
    results: list[str] = []
    for i, h in enumerate(highlights, 1):
        clip = process_clip(video_path, h, words, output_dir, i, len(highlights))
        if clip:
            results.append(clip)

    return results
