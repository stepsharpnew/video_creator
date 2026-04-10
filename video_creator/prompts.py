from __future__ import annotations

from .config import MAX_CLIP_SEC, MIN_CLIP_SEC
from .utils import fmt_time


def build_score_candidates_prompt(*, candidates_block: str) -> str:
    return f"""Ты — редактор-оценщик коротких вертикальных видео (TikTok / Reels / Shorts).

Тебе даны автоматически сформированные кандидаты-клипы из видео (сериал, шоу, выступление и т.д.).
Каждый кандидат построен вокруг зоны аудиторной реакции, определённой по аудио.

Ты НЕ выбираешь финальные клипы — только оцениваешь каждый.
Финальный отбор, дедупликацию и ранжирование делает программа на основе твоих оценок.

═══ ШКАЛЫ ОЦЕНКИ (1–10 для каждого кандидата) ═══

reaction_score (1–10):
  Насколько реакция аудитории уместна и усиливает момент?
  Учитывай: reaction_strength — это сила по аудио, но ты оцениваешь
  качество момента в контексте того, ЧТО вызвало реакцию.
  Фоновый шум или случайный всплеск = 1–3.
  Яркая, точно попавшая реакция (смех на шутку, ахание на поворот) = 8–10.

hook_score (1–10):
  Зацепят ли первые 1–3 секунды клипа зрителя, листающего ленту?
  Посмотри на первое utterance: оно должно мгновенно ставить вопрос или создавать интригу.
  Вялый вход, незавершённая фраза с предыдущей сцены, невнятный звук = 1–3.
  Чёткий, провокационный или смешной вход = 8–10.

self_contained_score (1–10):
  Понятна ли сцена ЛЮБОМУ зрителю, который НЕ смотрел остальной эпизод?
  Если для понимания нужно знать, кто персонажи или что было раньше = 1–4.
  Если ситуация считывается из самого клипа = 7–10.
  Особенно учитывай: есть ли в previous_context_5s критическая подводка,
  которая не вошла в окно? Если да — балл ниже.

transcript_clarity_score (1–10):
  Чёткость и понятность речи.
  WPM > 220 или < 100 = минус. Заглушенная реакцией речь (reaction_overlaps
  содержит текст одновременно с реакцией) = минус.
  Чистый, внятный диалог с нормальным темпом = 8–10.

title_quality_score (1–10):
  Насколько придуманный тобой title цепляет и точно описывает момент?
  Ставь себе честную оценку. Если title банальный / общий — снижай.

═══ ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ ═══

has_payoff (true/false):
  Есть ли в клипе завершённый payoff ДО конца?
  Payoff = реакция на шутку, развязка ситуации, неожиданный поворот, яркая финальная реплика.
  Если клип обрывается на середине мысли без разрешения — false.

novelty_summary (строка, 5–15 слов):
  Кратко опиши суть момента: что происходит, кто говорит, о чём.
  Программа использует это для обнаружения дублей между кандидатами.
  Будь конкретен: не "смешной диалог", а "Макс врёт шефу про заказ, тот не верит".

duplicate_group (строка):
  Если два+ кандидата — по сути одна и та же сцена (близкий reaction_at_ms,
  похожий текст), присвой одинаковый duplicate_group.
  Уникальным кандидатам — уникальный duplicate_group.

═══ ГРАНИЦЫ КЛИПА ═══
suggested_start_ms, suggested_end_ms:
  • Начало: логически понятная точка входа. Если previous_context содержит
    важную подводку — сдвинь start раньше, чтобы её захватить.
  • Конец: после payoff / реакции / на логической паузе.
  • Длина: от {MIN_CLIP_SEC} до {MAX_CLIP_SEC} секунд.

═══ КАНДИДАТЫ ═══
{candidates_block}

═══ ФОРМАТ ОТВЕТА ═══
Верни JSON-объект с ключом "scored_candidates" — массив.
Оцени ВСЕ кандидаты без исключения.

{{
  "scored_candidates": [
    {{
      "candidate_id": 1,
      "reaction_score": 8,
      "hook_score": 7,
      "self_contained_score": 6,
      "transcript_clarity_score": 9,
      "title_quality_score": 7,
      "has_payoff": true,
      "novelty_summary": "Макс врёт шефу про заказ, тот не верит",
      "duplicate_group": "макс_и_шеф_заказ",
      "suggested_start_ms": 11500,
      "suggested_end_ms": 46000,
      "title": "Короткое название кириллицей",
      "hook": "Зацепка — зачем смотреть (1 предложение)"
    }}
  ]
}}
"""


def format_candidates_block(candidates: list[dict]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(candidates, 1):
        reactions_str = ", ".join(
            f"{fmt_time(int(r['start_ms']))}–{fmt_time(int(r['end_ms']))} ({float(r['intensity']):.0%})"
            for r in c["reactions"]
        )

        overlaps_str = "; ".join(
            f"[{fmt_time(ov['reaction_start_ms'])}] «{ov['overlapping_text'][:60]}»"
            for ov in c.get("reaction_overlaps", [])
            if ov.get("overlapping_text")
        ) or "(нет речи во время реакции)"

        utterances_str = "\n    ".join(
            f"[{fmt_time(u['start_ms'])}–{fmt_time(u['end_ms'])}] {u['text']}"
            for u in c.get("utterances", [])
        )

        pauses_str = ", ".join(
            f"{fmt_time(p['at_ms'])} ({p['duration_ms']}мс)"
            for p in c.get("pause_gaps", [])
        ) or "нет заметных пауз"

        rtype = c.get("reaction_type", "crowd_noise")
        blocks.append(
            f"── Кандидат {i} ──\n"
            f"Окно: {fmt_time(c['win_start_ms'])}–{fmt_time(c['win_end_ms'])} "
            f"({(c['win_end_ms'] - c['win_start_ms']) // 1000}с)\n"
            f"reaction_at_ms: {c.get('reaction_at_ms', '?')}\n"
            f"reaction_strength: {c['peak_intensity']:.0%}\n"
            f"reaction_type: {rtype}\n"
            f"Реакции: {reactions_str}\n"
            f"reaction_overlaps: {overlaps_str}\n"
            f"word_count: {c.get('word_count', '?')}  estimated_wpm: {c.get('estimated_wpm', '?')}\n"
            f"previous_context_5s: {c.get('previous_context_5s') or '—'}\n"
            f"utterances:\n    {utterances_str}\n"
            f"pause_gaps: {pauses_str}\n"
            f"next_context_3s: {c.get('next_context_3s') or '—'}\n"
        )

    return "\n".join(blocks)
