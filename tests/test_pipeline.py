"""Dev/confirm split (PLAN.md §3.1)."""
import numpy as np

from dl_item_effects.pipeline import is_dev


def test_dev_split_is_a_stable_fifth():
    ids = np.arange(116_000_000, 116_200_000, dtype=np.int64)
    dev = is_dev(ids)
    assert abs(dev.mean() - 0.2) < 0.005
    # pinned values: changing the hash would silently move matches between dev and confirm
    assert is_dev([116000024, 116000127, 116392572, 1]).tolist() == [False, True, False, True]


def test_min_badge_keeps_only_high_lobbies(tmp_path):
    import pandas as pd

    from dl_item_effects.pipeline import POPULATIONS, load_hero_window

    ev = pd.DataFrame(dict(match_id=[1, 1, 2, 3], account_id=[10, 11, 12, 13],
                           hero_id=[13, 7, 13, 13], team=[0, 1, 0, 0], item_id=[5, 5, 5, 5],
                           game_time_s=[100, 100, 100, 100], sold_time_s=[0, 0, 0, 0],
                           badge_level=[49, 49, 48, 60]))
    ev.to_parquet(tmp_path / "part-001.parquet")
    w = load_hero_window("x", 13, events_dir=tmp_path, min_badge=POPULATIONS["oracle_plus"])
    assert sorted(w.events["match_id"]) == [1, 3]           # Archon 6 (48) is out
    assert set(w.match_events["match_id"]) == {1, 3}        # enemies only from kept matches
    assert len(load_hero_window("x", 13, events_dir=tmp_path).events) == 3
