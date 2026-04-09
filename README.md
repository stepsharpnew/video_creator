# Video Creator (highlights)

Пайплайн: концерт → N вертикальных клипов 9:16 с субтитрами.

## Требования

- Python 3.12+
- `ffmpeg` в PATH
- Ключи:
  - `ASSEMBLYAI_API_KEY`
  - `OPENAI_API_KEY`

## Установка

Если у тебя уже есть `venv/` — пропусти создание окружения.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка ключей

Создай файл `.env` в корне проекта:

```bash
ASSEMBLYAI_API_KEY=...
OPENAI_API_KEY=...
```

## Запуск

Из корня проекта:

```bash
source venv/bin/activate
python3 -m video_creator "concert.mp4" --count 10 --output highlights/
```

## Кэш/артефакты

В `--output` будут сохраняться:

- `_audio.mp3`
- `reactions.json`
- `transcript.json`
- `highlights.json`
- итоговые клипы `*.mp4`

