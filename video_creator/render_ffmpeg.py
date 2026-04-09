from __future__ import annotations

import os
import subprocess

from .config import OUTPUT_HEIGHT, OUTPUT_WIDTH, SUB_FONT_NAME, SUB_FONTSIZE
from .subtitles import make_srt
from .utils import fmt_time, log


def process_clip(
    video_path: str,
    highlight: dict,
    words: list[dict],
    output_dir: str,
    index: int,
    total: int,
) -> str | None:
    safe = "".join(c if (c.isalnum() or c in "_ ") else "_" for c in str(highlight.get("title", "clip")))
    safe = safe.replace(" ", "_")[:35]
    base = f"{index:02d}_{safe}"
    raw = os.path.join(output_dir, f"{base}_raw.mp4")
    srt_f = os.path.join(output_dir, f"{base}.srt")
    final = os.path.join(output_dir, f"{base}.mp4")

    start_s = int(highlight["start_ms"]) / 1000
    duration = (int(highlight["end_ms"]) - int(highlight["start_ms"])) / 1000

    log(
        "FFMPEG",
        f"Клип {index}/{total}: [{fmt_time(int(highlight['start_ms']))}] {duration:.0f}с — {highlight.get('title', '')}",
    )

    vf = (
        f"[0:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=25:5[bg];"
        f"[0:v]scale={OUTPUT_WIDTH}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-filter_complex",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        raw,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("FFMPEG", f"Ошибка trim: {r.stderr[-300:]}")
        return None

    srt_content = make_srt(words, int(highlight["start_ms"]), int(highlight["end_ms"]))
    if srt_content:
        with open(srt_f, "w", encoding="utf-8") as f:
            f.write(srt_content)

        style = (
            f"FontName={SUB_FONT_NAME},FontSize={SUB_FONTSIZE},Bold=0,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H40000000,"
            f"Outline=1,Shadow=1,ShadowColour=&H80000000,"
            f"Alignment=2,MarginV=30"
        )
        srt_esc = srt_f.replace("\\", "/").replace(":", "\\:")
        cmd_sub = [
            "ffmpeg",
            "-y",
            "-i",
            raw,
            "-vf",
            f"subtitles={srt_esc}:force_style='{style}'",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-c:a",
            "copy",
            final,
        ]
        r2 = subprocess.run(cmd_sub, capture_output=True, text=True)
        if r2.returncode != 0:
            log("FFMPEG", "Субтитры не наложились — сохраняю без них")
            os.rename(raw, final)
        else:
            os.remove(raw)
    else:
        os.rename(raw, final)

    size_mb = os.path.getsize(final) / 1_048_576
    log("DONE", f"{os.path.basename(final)} ({size_mb:.1f} MB)")
    return final

