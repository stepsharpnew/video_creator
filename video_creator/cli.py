from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import HIGHLIGHTS_COUNT
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video Highlights Pipeline")
    p.add_argument(
        "video",
        help=(
            "Видео: путь или имя файла. "
            "Если передано только имя, файл ищется в папке videos/"
        ),
    )
    p.add_argument("--output", default="highlights", help="Папка для результатов")
    p.add_argument("--count", type=int, default=HIGHLIGHTS_COUNT)
    p.add_argument("--skip-asr", help="Путь к готовому transcript.json")
    return p


def _normalize_video_arg(raw_video: str) -> str:
    # Handle accidental extra shell quotes around the argument.
    v = raw_video.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
        v = v[1:-1].strip()
    return v


def _resolve_video_path(video_arg: str) -> str:
    video_arg = _normalize_video_arg(video_arg)
    candidate = Path(video_arg).expanduser()

    # Absolute/relative path explicitly provided.
    if candidate.exists():
        return str(candidate)

    # If looks like a path but does not exist, return as-is for clear error in pipeline.
    if candidate.parent != Path("."):
        return str(candidate)

    # Resolve plain filename from project videos/ folder.
    project_root = Path(__file__).resolve().parent.parent
    videos_dir = project_root / "videos"
    direct = videos_dir / candidate.name
    if direct.exists():
        return str(direct)

    # If extension omitted, try to match by stem.
    if candidate.suffix == "" and videos_dir.exists():
        matches = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.stem == candidate.name)
        if len(matches) == 1:
            return str(matches[0])

    return str(direct)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = _resolve_video_path(args.video)

    print("\n" + "=" * 58)
    print("  VIDEO HIGHLIGHTS PIPELINE")
    print(f"  Входной файл  : {video_path}")
    print(f"  Клипов        : {args.count}")
    print(f"  Выходная папка: {args.output}")
    print("=" * 58 + "\n")

    results = run_pipeline(
        video_path=video_path,
        output_dir=args.output,
        count=args.count,
        skip_asr_path=args.skip_asr,
    )

    print("\n" + "=" * 58)
    print(f"  ГОТОВО! Клипов: {len(results)}/{args.count}")
    print(f"  Папка: {os.path.abspath(args.output)}")
    print("=" * 58)
    for r in results:
        print(f"  {os.path.basename(r)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

