"""Phase 1 — item/hero/rank decoding.

Turns raw item_id ints into structured info and separates real shop purchases
(type="upgrade": slot/tier/cost, atomic — no component recipes) from ability-point
upgrades (type="ability") in a match's players[].items[] array. See PLAN.md §5 Phase 1.
"""
from .decoder import ItemDecoder, ItemInfo, Purchase

__all__ = ["ItemDecoder", "ItemInfo", "Purchase"]
