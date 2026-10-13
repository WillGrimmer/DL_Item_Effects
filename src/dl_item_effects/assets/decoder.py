"""Item / asset decoding (PLAN.md Phase 1).

Turns raw ``item_id`` ints (as they appear in match metadata ``players[].items[]``)
into structured info, and — crucially — separates the two things that array conflates:

* ``type == "upgrade"``  -> a real **shop purchase** (the counterfactual-relevant decision).
                            Atomic: has slot (weapon/spirit/vitality), tier (1-5), cost.
* ``type == "ability"``  -> a **hero ability-point upgrade** (levelling a skill). These
                            repeat several times per match and are NOT purchases.
* ``type == "weapon"``   -> hero base weapon entry.

Verified against the live ``/v1/assets/items`` payload (726 assets: 86 weapon, 389 ability,
251 upgrade) on 2026-07-22. Deadlock shop items do **not** combine from components, so
inventory reconstruction is simply "the shop purchases before time t" (see decode_purchases).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import config

# Asset "type" values that correspond to buyable shop items vs hero abilities.
SHOP_TYPE = "upgrade"
ABILITY_TYPE = "ability"
SLOTS = ("weapon", "spirit", "vitality")


@dataclass(frozen=True)
class ItemInfo:
    id: int
    name: str
    type: str                 # "upgrade" | "ability" | "weapon"
    slot: str | None          # weapon | spirit | vitality (shop items only)
    tier: int | None          # 1-5 (shop items only)
    cost: int | None
    is_active: bool | None
    shopable: bool | None
    class_name: str | None = None

    @property
    def is_shop_item(self) -> bool:
        return self.type == SHOP_TYPE

    @property
    def is_ability(self) -> bool:
        return self.type == ABILITY_TYPE


@dataclass(frozen=True)
class Purchase:
    """One decoded shop purchase from a player's items[] list."""
    item_id: int
    name: str
    slot: str | None
    tier: int | None
    cost: int | None
    is_active: bool | None
    game_time_s: int          # exact purchase time (the field the whole project relies on)
    sold_time_s: int          # 0 = still held at match end
    held_at_end: bool


class ItemDecoder:
    """Lookup table over all Deadlock assets. Build via :meth:`load` (cached) or
    :meth:`from_records` (e.g. a test fixture)."""

    def __init__(self, records: Iterable[dict[str, Any]]):
        self._by_id: dict[int, ItemInfo] = {}
        for r in records:
            iid = r.get("id")
            if iid is None:
                continue
            self._by_id[int(iid)] = ItemInfo(
                id=int(iid),
                name=r.get("name") or r.get("class_name") or f"item_{iid}",
                type=r.get("type", ""),
                slot=r.get("item_slot_type"),
                tier=r.get("item_tier"),
                cost=r.get("cost"),
                is_active=r.get("is_active_item"),
                shopable=r.get("shopable"),
                class_name=r.get("class_name"),
            )

    # -- construction -------------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "ItemDecoder":
        return cls(records)

    @classmethod
    def from_json(cls, path: str | Path) -> "ItemDecoder":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def load(cls, api=None, refresh: bool = False,
             cache_path: str | Path | None = None) -> "ItemDecoder":
        """Load from local cache; fetch from the API and cache if missing/refresh.

        ``api`` is a :class:`dl_item_effects.api.DeadlockAPI` (imported lazily so the
        decoder is usable offline / in tests without network).
        """
        cache = Path(cache_path or (config.INTERIM_DIR / "assets_items.json"))
        if refresh or not cache.exists():
            if api is None:
                from ..api import DeadlockAPI
                api = DeadlockAPI()
            records = api.assets_items()
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(records, fh)
        else:
            with open(cache, "r", encoding="utf-8") as fh:
                records = json.load(fh)
        return cls(records)

    # -- lookups ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, item_id: int) -> ItemInfo | None:
        return self._by_id.get(int(item_id))

    def name(self, item_id: int) -> str:
        info = self.get(item_id)
        return info.name if info else f"unknown_{item_id}"

    def is_shop_item(self, item_id: int) -> bool:
        info = self.get(item_id)
        return bool(info and info.is_shop_item)

    def is_ability(self, item_id: int) -> bool:
        info = self.get(item_id)
        return bool(info and info.is_ability)

    def shop_items(self) -> list[ItemInfo]:
        return [i for i in self._by_id.values() if i.is_shop_item]

    # -- the useful bit: decode a player's items[] list ---------------------
    def decode_purchases(self, items: list[dict[str, Any]],
                         shopable_only: bool = True) -> list[Purchase]:
        """Filter a match player's ``items[]`` to real shop purchases (dropping ability
        upgrades and weapon-base entries) and return them decoded, sorted by time."""
        out: list[Purchase] = []
        for it in items:
            info = self.get(it.get("item_id"))
            if info is None or not info.is_shop_item:
                continue
            if shopable_only and info.shopable is False:
                continue
            sold = it.get("sold_time_s", 0) or 0
            out.append(Purchase(
                item_id=info.id, name=info.name, slot=info.slot, tier=info.tier,
                cost=info.cost, is_active=info.is_active,
                game_time_s=it.get("game_time_s", 0), sold_time_s=sold,
                held_at_end=(sold == 0),
            ))
        out.sort(key=lambda p: p.game_time_s)
        return out

    def ability_upgrades(self, items: list[dict[str, Any]]) -> list[tuple[int, str, int]]:
        """The ability-point spends from items[]: (game_time_s, ability_name, item_id).
        Kept separate — useful later as an ability-build-order feature, not a purchase."""
        out = []
        for it in items:
            info = self.get(it.get("item_id"))
            if info and info.is_ability:
                out.append((it.get("game_time_s", 0), info.name, info.id))
        out.sort()
        return out

    def inventory_at(self, items: list[dict[str, Any]], t: int) -> list[Purchase]:
        """Shop items owned at game time ``t`` (bought at/before t and not yet sold)."""
        return [p for p in self.decode_purchases(items)
                if p.game_time_s <= t and (p.sold_time_s == 0 or p.sold_time_s > t)]
