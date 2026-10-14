"""Central configuration. Secrets come from env vars / .env — never hard-code keys."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# --- API -------------------------------------------------------------------
API_BASE = os.getenv("DEADLOCK_API_BASE", "https://api.deadlock-api.com")
API_KEY = os.getenv("DEADLOCK_API_KEY")  # optional; raises per-IP limit 200->400 req/min

# Rate limits (req/min): 200 per IP, 400 with key, 2000 global. Stay comfortably under.
REQ_PER_MIN = 380 if API_KEY else 190

# --- Data lake (docs/data_dumps.md) ----------------------------------------
LAKE_BASE = os.getenv("DEADLOCK_LAKE_BASE", "https://data.deadlock-api.com")
MANIFEST_URL = f"{LAKE_BASE}/v1/manifest.json"

# Souls every player holds at t=0: net worth minus all summed gold sources at the first
# snapshot is 600 for the large majority of Ranked players (checked 2026-10-13), and
# stats.level is 1 at that net worth. Used as the start-of-game state before the first
# snapshot (180 s), instead of borrowing a snapshot from after the purchase.
START_SOULS = 600
START_LEVEL = 1

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # cached match metadata (git-ignored)
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"  # parquet feature tables
MODELS_DIR = REPO_ROOT / "models"       # git-ignored

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
