"""Offline tests for the item-asset decoder (Phase 1).

Uses the committed fixture tests/fixtures/assets_items_sample.json (real records pulled
from /v1/assets/items) — no network required.
"""
import json
from pathlib import Path

import pytest

from dl_item_effects.assets import ItemDecoder, Purchase

FIXTURE = Path(__file__).parent / "fixtures" / "assets_items_sample.json"

# ids from the fixture
CARD_TRICK = 1999680326      # ability
FULL_AUTO = 1842576017       # ability
GOLDEN_GOOSE = 2462046703    # upgrade, spirit, tier 1
MONSTER_ROUNDS = 1009965641  # upgrade, weapon, tier 1
QUICKSILVER = 84321454       # upgrade, spirit, tier 2
CULTIST = 709540378          # upgrade, weapon, tier 3


@pytest.fixture(scope="module")
def dec() -> ItemDecoder:
    return ItemDecoder.from_json(FIXTURE)


def test_loads_all_records(dec):
    assert len(dec) == 8


def test_classifies_ability_vs_shop(dec):
    assert dec.is_ability(CARD_TRICK)
    assert not dec.is_shop_item(CARD_TRICK)
    assert dec.is_shop_item(GOLDEN_GOOSE)
    assert not dec.is_ability(GOLDEN_GOOSE)


def test_shop_item_fields(dec):
    gg = dec.get(GOLDEN_GOOSE)
    assert gg.name == "Golden Goose Egg"
    assert gg.slot == "spirit"
    assert gg.tier == 1
    assert gg.cost == 800


def test_unknown_id_is_safe(dec):
    assert dec.get(123) is None
    assert dec.name(123).startswith("unknown_")
    assert not dec.is_shop_item(123)


def _sample_items():
    """Mimic a real players[].items[] array: ability upgrades repeat; shop items are atomic."""
    return [
        {"item_id": CARD_TRICK, "game_time_s": 23, "sold_time_s": 0},
        {"item_id": MONSTER_ROUNDS, "game_time_s": 185, "sold_time_s": 575},  # sold later
        {"item_id": CARD_TRICK, "game_time_s": 300, "sold_time_s": 0},        # ability re-level
        {"item_id": GOLDEN_GOOSE, "game_time_s": 63, "sold_time_s": 821},
        {"item_id": QUICKSILVER, "game_time_s": 364, "sold_time_s": 0},
        {"item_id": FULL_AUTO, "game_time_s": 209, "sold_time_s": 0},          # ability
        {"item_id": CULTIST, "game_time_s": 575, "sold_time_s": 0},
    ]


def test_decode_purchases_drops_abilities_and_sorts(dec):
    purchases = dec.decode_purchases(_sample_items())
    # 4 shop items (goose, monster rounds, quicksilver, cultist); abilities dropped
    assert [p.item_id for p in purchases] == [GOLDEN_GOOSE, MONSTER_ROUNDS, QUICKSILVER, CULTIST]
    assert all(isinstance(p, Purchase) for p in purchases)
    # sorted by game_time_s
    assert [p.game_time_s for p in purchases] == [63, 185, 364, 575]


def test_held_at_end_flag(dec):
    purchases = {p.item_id: p for p in dec.decode_purchases(_sample_items())}
    assert purchases[MONSTER_ROUNDS].held_at_end is False   # sold at 575
    assert purchases[CULTIST].held_at_end is True            # sold_time_s == 0


def test_ability_upgrades_kept_separate(dec):
    ab = dec.ability_upgrades(_sample_items())
    names = [n for _, n, _ in ab]
    assert names.count("Card Trick") == 2
    assert "Full Auto" in names
    assert "Golden Goose Egg" not in names  # shop items excluded


def test_inventory_at_respects_buy_and_sell_times(dec):
    items = _sample_items()
    # at t=400: goose bought@63 sold@821 (held), monster rounds bought@185 sold@575 (held),
    # quicksilver bought@364 (held), cultist bought@575 (not yet)
    inv = {p.item_id for p in dec.inventory_at(items, 400)}
    assert inv == {GOLDEN_GOOSE, MONSTER_ROUNDS, QUICKSILVER}
    # at t=600: monster rounds sold@575 -> gone; cultist@575 -> now present
    inv2 = {p.item_id for p in dec.inventory_at(items, 600)}
    assert inv2 == {GOLDEN_GOOSE, QUICKSILVER, CULTIST}
