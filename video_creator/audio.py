from __future__ import annotations

import subprocess

from .utils import log


def extract_audio(video_path: str, audio_path: str) -> str:
    log("AUDIO", f"Извлекаю аудио из {video_path} ...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "64k",
        audio_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg ошибка:\n{r.stderr[-400:]}")
    log("AUDIO", f"Аудио сохранено: {audio_path}")
    return audio_path

