from __future__ import annotations

import json

from .config import HTTP_TIMEOUT, OPENAI_API_KEY, OPENAI_BASE_URL
from .http import HTTP
from .prompts import build_score_candidates_prompt, format_candidates_block
from .utils import log


def score_candidates(candidates: list[dict]) -> list[dict]:
    """
    Send all candidates to OpenAI for evaluation.
    Returns raw scored_candidates — no selection, just per-candidate scores.
    """
    candidates_block = format_candidates_block(candidates)
    prompt = build_score_candidates_prompt(candidates_block=candidates_block)

    log("GPT", f"Отправляю {len(candidates)} кандидатов в OpenAI для оценки ...")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }

    resp = HTTP.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    data = json.loads(content)
    scored = data.get("scored_candidates", [])

    log("GPT", f"Оценено кандидатов: {len(scored)}/{len(candidates)}")
    for s in scored[:12]:
        cid = s.get("candidate_id", "?")
        rs = s.get("reaction_score", 0)
        hs = s.get("hook_score", 0)
        sc = s.get("self_contained_score", 0)
        cl = s.get("transcript_clarity_score", 0)
        tq = s.get("title_quality_score", 0)
        pf = s.get("has_payoff", "?")
        dg = s.get("duplicate_group", "?")
        log(
            "GPT",
            f"  #{cid}: react={rs} hook={hs} self={sc} clarity={cl} title={tq} "
            f"payoff={pf} dup={dg}",
        )

    return scored
