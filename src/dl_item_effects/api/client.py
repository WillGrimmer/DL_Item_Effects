"""Thin, rate-limited client for deadlock-api.com.

Only the endpoints this project needs are wrapped. Field notes live in PLAN.md §2.
Endpoints verified against the live API on 2026-07-22.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from .. import config


class DeadlockAPI:
    def __init__(self, base: str | None = None, api_key: str | None = None,
                 req_per_min: int | None = None, timeout: float = 30.0):
        self.base = (base or config.API_BASE).rstrip("/")
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.req_per_min = req_per_min or config.REQ_PER_MIN
        self.timeout = timeout
        self._min_interval = 60.0 / max(self.req_per_min, 1)
        self._last_call = 0.0
        self._session = requests.Session()
        if self.api_key:
            # Header name per deadlock-api docs; adjust if the API expects a query param.
            self._session.headers["X-API-Key"] = self.api_key

    # -- core ---------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, path: str, **params: Any) -> Any:
        self._throttle()
        r = self._session.get(f"{self.base}{path}", params=params or None,
                              timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # -- matches ------------------------------------------------------------
    def recently_fetched(self) -> list[dict]:
        """Recent matches: {match_id, start_time, duration_s, match_mode, game_mode,
        average_badge_team0/1, players[]}. Handy match-ID seed source."""
        return self.get("/v1/matches/recently-fetched")

    def match_metadata(self, match_id: int) -> dict:
        """THE key endpoint. players[].items[].game_time_s = purchase timestamps;
        players[].stats[] = economy series @180s. See PLAN.md §2 for the full schema."""
        return self.get(f"/v1/matches/{match_id}/metadata")

    def player_match_history(self, account_id: int) -> Any:
        return self.get(f"/v1/players/{account_id}/match-history")

    # -- assets (reference / decoding) -------------------------------------
    def assets_items(self) -> list[dict]:
        return self.get("/v1/assets/items")

    def assets_heroes(self) -> list[dict]:
        return self.get("/v1/assets/heroes")

    def assets_ranks(self) -> list[dict]:
        return self.get("/v1/assets/ranks")

    def patches(self) -> list[dict]:
        """Changelog threads: ``title`` is the patch date (e.g. "09-29-2026" or
        "09-16-2026 Update"); ``pub_date`` moves when hotfixes are appended."""
        return self.get("/v1/patches")
