from __future__ import annotations

import os

from .env import load_dotenv


load_dotenv()

# API keys / endpoints
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "YOUR_ASSEMBLYAI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Defaults
HIGHLIGHTS_COUNT = 10
MIN_CLIP_SEC = 30
MAX_CLIP_SEC = 60
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
LANGUAGE = "ru"

# Audience reaction analysis
ANALYSIS_SAMPLE_RATE = 8000
ANALYSIS_WINDOW_SEC = 1.0
ANALYSIS_HOP_SEC = 0.5
REACTION_MERGE_GAP = 3000  # ms
REACTION_THRESHOLD = 2.0   # peak = N × median loudness

# Candidate windows around reactions (ms)
CANDIDATE_SETUP_MS = 10_000
CANDIDATE_TAIL_MS = 3_000
CANDIDATE_CLUSTER_GAP_MS = 10_000
CANDIDATE_PREV_CONTEXT_MS = 5_000
CANDIDATE_NEXT_CONTEXT_MS = 3_000

# Pause detection in transcript
PAUSE_GAP_MS = 400

# Subtitle segmentation
SUB_MIN_DURATION_MS = 800
SUB_MAX_DURATION_MS = 5_000
SUB_MAX_CHARS_PER_LINE = 32
SUB_MAX_LINES = 2
SUB_TARGET_WPM = 180
SUB_FONTSIZE = 12
SUB_FONT_NAME = "DejaVu Sans"

# Words / phrases to strip from transcript (ASR artefacts)
TRANSCRIPT_NOISE_PATTERNS: list[str] = [
    "реакция зала",
    "аплодисменты",
    "смех в зале",
    "музыка",
    "иностранная музыка",
    "неразборчиво",
    "звук",
    "шум",
]

# ── Score aggregation weights (must sum to 1.0) ──
SCORE_W_REACTION = 0.30
SCORE_W_HOOK = 0.20
SCORE_W_SELF_CONTAINED = 0.20
SCORE_W_NOVELTY = 0.15
SCORE_W_CLARITY = 0.10
SCORE_W_TITLE = 0.05

# ── Greedy selection thresholds ──
OVERLAP_REJECT_RATIO = 0.25       # reject if temporal overlap > 25% of shorter clip
TRANSCRIPT_JACCARD_REJECT = 0.40  # reject if word-set Jaccard similarity > 40%
REACTION_AT_PROXIMITY_MS = 5_000  # reject if reaction_at_ms within 5s of already-selected
MIN_HOOK_SCORE = 4                # hard floor: skip candidates with hook < this
PAYOFF_PENALTY = 0.85             # multiply base_score if has_payoff == false
NOVELTY_DECAY_PER_SIMILAR = 0.70  # multiply novelty_score per semantically similar selected clip

# Network
HTTP_TIMEOUT = (10, 300)
