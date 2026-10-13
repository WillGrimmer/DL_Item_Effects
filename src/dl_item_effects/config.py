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

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # cached match metadata (git-ignored)
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"  # parquet feature tables
MODELS_DIR = REPO_ROOT / "models"       # git-ignored

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
