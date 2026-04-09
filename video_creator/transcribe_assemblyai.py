from __future__ import annotations

import time

import requests

from .config import ASSEMBLYAI_API_KEY, HTTP_TIMEOUT, LANGUAGE
from .http import HTTP
from .utils import log


def upload_to_assemblyai(audio_path: str) -> str:
    log("AAI", "Загружаю аудио в AssemblyAI ...")
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            with open(audio_path, "rb") as f:
                resp = HTTP.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers={"Authorization": ASSEMBLYAI_API_KEY},
                    data=f,
                    timeout=HTTP_TIMEOUT,
                )
            resp.raise_for_status()
            url = resp.json()["upload_url"]
            log("AAI", f"Загружено: {url[:60]}...")
            return url
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt >= 5:
                break
            sleep_s = 2 ** (attempt - 1)
            log("AAI", f"Сетевая ошибка (попытка {attempt}/5): {e}. Повтор через {sleep_s}с...")
            time.sleep(sleep_s)
    raise RuntimeError(f"Не удалось загрузить аудио после 5 попыток: {last_err}")


def start_transcription(upload_url: str) -> str:
    log("AAI", "Запускаю транскрибацию...")
    payload = {
        "audio_url": upload_url,
        "language_code": LANGUAGE,
        "speaker_labels": True,
        "speech_models": ["universal-3-pro", "universal-2"],
    }
    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}
    resp = HTTP.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    if not resp.ok:
        log("AAI", f"Ошибка {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    tid = resp.json()["id"]
    log("AAI", f"Транскрипция запущена, ID: {tid}")
    return tid


def poll_transcription(transcript_id: str) -> dict:
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}
    while True:
        resp = HTTP.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        st = data["status"]
        if st == "completed":
            log("AAI", f"Готово! Слов: {len(data.get('words', []))}")
            return data
        if st == "error":
            raise RuntimeError(f"AssemblyAI ошибка: {data.get('error')}")
        log("AAI", f"Статус: {st} — жду 15 сек...")
        time.sleep(15)

