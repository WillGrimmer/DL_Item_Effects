"""Dev/confirm split (PLAN.md §3.1)."""
import numpy as np

from dl_item_effects.pipeline import is_dev


def test_dev_split_is_a_stable_fifth():
    ids = np.arange(116_000_000, 116_200_000, dtype=np.int64)
    dev = is_dev(ids)
    assert abs(dev.mean() - 0.2) < 0.005
    # pinned values: changing the hash would silently move matches between dev and confirm
    assert is_dev([116000024, 116000127, 116392572, 1]).tolist() == [False, True, False, True]
