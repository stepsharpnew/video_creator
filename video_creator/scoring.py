from __future__ import annotations

import re

from .config import (
    MAX_CLIP_SEC,
    MIN_CLIP_SEC,
    MIN_HOOK_SCORE,
    NOVELTY_DECAY_PER_SIMILAR,
    OVERLAP_REJECT_RATIO,
    PAYOFF_PENALTY,
    REACTION_AT_PROXIMITY_MS,
    SCORE_W_CLARITY,
    SCORE_W_HOOK,
    SCORE_W_NOVELTY,
    SCORE_W_REACTION,
    SCORE_W_SELF_CONTAINED,
    SCORE_W_TITLE,
    TRANSCRIPT_JACCARD_REJECT,
)
from .utils import fmt_time, log


# ── public API ──────────────────────────────────────────────


def select_top_highlights(
    scored: list[dict],
    candidates: list[dict],
    total_ms: int,
    count: int,
) -> list[dict]:
    """
    Deterministic final selection:
      1. Merge GPT scores with original candidate metadata
      2. Compute base_score (weighted sum)
      3. Apply payoff / hook penalties
      4. Clamp boundaries
      5. Greedy loop: pick best → reject similar/overlapping → repeat
    """
    merged = _merge_scored_with_candidates(scored, candidates)
    items = _compute_base_scores(merged)
    items = _apply_penalties(items)
    items = _clamp_boundaries(items, total_ms)

    if not items:
        log("SELECT", "Нет валидных кандидатов после подготовки.")
        return []

    selected = _greedy_select(items, count)

    log("SELECT", f"Итого клипов: {len(selected)} (из {len(scored)} оценённых, запрошено ≤{count})")
    for i, it in enumerate(selected, 1):
        dur = (it["end_ms"] - it["start_ms"]) // 1000
        log(
            "SELECT",
            f"  {i:2}. [{fmt_time(it['start_ms'])}] {dur}с  "
            f"final={it['final_score']:.2f}  base={it['base_score']:.2f}  "
            f"dup={it['duplicate_group']}  — {it['title']}",
        )

    return [
        {
            "start_ms": it["start_ms"],
            "end_ms": it["end_ms"],
            "title": it["title"],
            "hook": it["hook"],
            "final_score": it["final_score"],
            "base_score": it["base_score"],
            "scores": it["scores"],
            "duplicate_group": it["duplicate_group"],
            "novelty_summary": it.get("novelty_summary", ""),
            "has_payoff": it.get("has_payoff", True),
        }
        for it in selected
    ]


# ── merge GPT output with enriched candidates ───────────────


def _merge_scored_with_candidates(scored: list[dict], candidates: list[dict]) -> list[dict]:
    cand_by_idx: dict[int, dict] = {}
    for i, c in enumerate(candidates, 1):
        cand_by_idx[i] = c

    merged: list[dict] = []
    for s in scored:
        cid = int(s.get("candidate_id", 0))
        orig = cand_by_idx.get(cid, {})
        merged.append({
            **s,
            "transcript_chunk": orig.get("transcript_chunk", ""),
            "reaction_at_ms": orig.get("reaction_at_ms", s.get("suggested_start_ms", 0)),
            "peak_intensity": orig.get("peak_intensity", 0),
        })
    return merged


# ── score computation ────────────────────────────────────────


def _compute_base_scores(items: list[dict]) -> list[dict]:
    result: list[dict] = []
    for s in items:
        reaction = _clamp(s.get("reaction_score", 1))
        hook = _clamp(s.get("hook_score", 1))
        self_cont = _clamp(s.get("self_contained_score", 1))
        clarity = _clamp(s.get("transcript_clarity_score", 1))
        title_q = _clamp(s.get("title_quality_score", 1))

        base = (
            reaction * SCORE_W_REACTION
            + hook * SCORE_W_HOOK
            + self_cont * SCORE_W_SELF_CONTAINED
            + clarity * SCORE_W_CLARITY
            + title_q * SCORE_W_TITLE
        )

        result.append({
            "candidate_id": s.get("candidate_id"),
            "start_ms": int(s.get("suggested_start_ms", 0)),
            "end_ms": int(s.get("suggested_end_ms", 0)),
            "title": s.get("title", "clip"),
            "hook": s.get("hook", ""),
            "duplicate_group": str(s.get("duplicate_group", s.get("candidate_id", "?"))),
            "novelty_summary": str(s.get("novelty_summary", "")),
            "has_payoff": bool(s.get("has_payoff", True)),
            "transcript_chunk": s.get("transcript_chunk", ""),
            "reaction_at_ms": int(s.get("reaction_at_ms", 0)),
            "peak_intensity": float(s.get("peak_intensity", 0)),
            "base_score": round(base, 3),
            "final_score": round(base, 3),
            "scores": {
                "reaction": reaction,
                "hook": hook,
                "self_contained": self_cont,
                "clarity": clarity,
                "title_quality": title_q,
            },
        })

    result.sort(key=lambda x: x["base_score"], reverse=True)
    return result


def _apply_penalties(items: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for it in items:
        score = it["base_score"]

        if it["scores"]["hook"] < MIN_HOOK_SCORE:
            reason = f"hook={it['scores']['hook']} < {MIN_HOOK_SCORE}"
            log("PENALTY", f"  #{it['candidate_id']} пропущен — {reason}")
            continue

        if not it["has_payoff"]:
            score *= PAYOFF_PENALTY
            log(
                "PENALTY",
                f"  #{it['candidate_id']} has_payoff=false → "
                f"score {it['base_score']:.2f} → {score:.2f}",
            )

        it["final_score"] = round(score, 3)
        valid.append(it)

    return valid


# ── boundary clamping ────────────────────────────────────────


def _clamp_boundaries(items: list[dict], total_ms: int) -> list[dict]:
    valid: list[dict] = []
    for it in items:
        start = it["start_ms"]
        end = it["end_ms"]
        if end <= start:
            continue

        if end - start < MIN_CLIP_SEC * 1000:
            end = start + MIN_CLIP_SEC * 1000
        if end - start > MAX_CLIP_SEC * 1000:
            end = start + MAX_CLIP_SEC * 1000

        if total_ms > 0 and end > total_ms:
            end = total_ms
            start = min(start, max(0, end - MIN_CLIP_SEC * 1000))

        if end - start < MIN_CLIP_SEC * 1000:
            continue

        it["start_ms"] = start
        it["end_ms"] = end
        valid.append(it)

    return valid


# ── greedy selection with multi-signal rejection ─────────────


def _greedy_select(pool: list[dict], count: int) -> list[dict]:
    """
    Pick the best clip, reject everything too similar/overlapping, repeat.
    Each iteration re-applies novelty decay to remaining candidates relative
    to what's already been selected.
    """
    remaining = list(pool)
    selected: list[dict] = []

    while remaining and len(selected) < count:
        remaining.sort(key=lambda x: x["final_score"], reverse=True)
        best = remaining.pop(0)
        selected.append(best)

        next_remaining: list[dict] = []
        for it in remaining:
            reason = _rejection_reason(it, selected)
            if reason:
                log(
                    "REJECT",
                    f"  #{it['candidate_id']} [{fmt_time(it['start_ms'])}] "
                    f"отброшен — {reason}",
                )
                continue

            novelty_mult = _novelty_multiplier(it, selected)
            if novelty_mult < 1.0:
                old = it["final_score"]
                it["final_score"] = round(it["base_score"] * novelty_mult, 3)
                if it["final_score"] != old:
                    log(
                        "NOVELTY",
                        f"  #{it['candidate_id']} novelty decay "
                        f"{old:.2f} → {it['final_score']:.2f}",
                    )

            next_remaining.append(it)

        remaining = next_remaining

    return selected


def _rejection_reason(candidate: dict, selected: list[dict]) -> str | None:
    """Hard rejection: returns reason string or None if candidate is ok."""
    cid = candidate["candidate_id"]

    for sel in selected:
        # 1. duplicate_group
        if candidate["duplicate_group"] == sel["duplicate_group"]:
            return f"duplicate_group={candidate['duplicate_group']} (уже взят #{sel['candidate_id']})"

        # 2. temporal overlap > threshold
        overlap_ms = _overlap_ms(candidate, sel)
        if overlap_ms > 0:
            shorter_dur = min(
                candidate["end_ms"] - candidate["start_ms"],
                sel["end_ms"] - sel["start_ms"],
            )
            if shorter_dur > 0 and overlap_ms / shorter_dur > OVERLAP_REJECT_RATIO:
                pct = overlap_ms / shorter_dur
                return (
                    f"overlap {pct:.0%} с #{sel['candidate_id']} "
                    f"(>{OVERLAP_REJECT_RATIO:.0%})"
                )

        # 3. reaction_at_ms too close
        delta = abs(candidate["reaction_at_ms"] - sel["reaction_at_ms"])
        if delta < REACTION_AT_PROXIMITY_MS:
            return (
                f"reaction_at_ms {fmt_time(candidate['reaction_at_ms'])} "
                f"в {delta}мс от #{sel['candidate_id']} "
                f"({fmt_time(sel['reaction_at_ms'])})"
            )

        # 4. transcript Jaccard similarity
        jacc = _word_jaccard(candidate["transcript_chunk"], sel["transcript_chunk"])
        if jacc > TRANSCRIPT_JACCARD_REJECT:
            return (
                f"transcript Jaccard={jacc:.2f} с #{sel['candidate_id']} "
                f"(>{TRANSCRIPT_JACCARD_REJECT})"
            )

        # 5. title near-duplicate
        title_jacc = _word_jaccard(candidate["title"], sel["title"])
        if title_jacc > 0.7:
            return f"title слишком похож на #{sel['candidate_id']}: «{sel['title']}»"

    return None


def _novelty_multiplier(candidate: dict, selected: list[dict]) -> float:
    """
    Soft penalty: for each selected clip that is somewhat similar,
    multiply by NOVELTY_DECAY_PER_SIMILAR.
    Returns a multiplier in (0, 1].
    """
    mult = 1.0
    cand_summary = candidate.get("novelty_summary", "")

    for sel in selected:
        sel_summary = sel.get("novelty_summary", "")
        sim = _word_jaccard(cand_summary, sel_summary)
        if sim > 0.25:
            mult *= NOVELTY_DECAY_PER_SIMILAR

    novelty_component = mult * 10.0
    base_without_novelty = candidate["base_score"]
    return (base_without_novelty + SCORE_W_NOVELTY * novelty_component) / (
        base_without_novelty + SCORE_W_NOVELTY * 10.0
    ) if (base_without_novelty + SCORE_W_NOVELTY * 10.0) > 0 else 1.0


# ── similarity helpers ───────────────────────────────────────


_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _word_jaccard(a: str, b: str) -> float:
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _overlap_ms(a: dict, b: dict) -> int:
    start = max(a["start_ms"], b["start_ms"])
    end = min(a["end_ms"], b["end_ms"])
    return max(0, end - start)


def _clamp(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 1.0
    return max(1.0, min(10.0, f))
