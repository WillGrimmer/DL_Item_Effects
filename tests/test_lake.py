"""Manifest handling, partition selection by date, and patch windows. Offline."""
from datetime import date

import pandas as pd

from dl_item_effects.ingest import extract, lake
from dl_item_effects.ingest.patches import parse_patches, patch_window


def test_base_files_ignore_deltas():
    man = {"tables": {"match_player": {"files": [
        {"key": "v1/tables/match_player/g2/base/part-7/a.parquet", "kind": "base", "partition": 7},
        {"key": "v1/tables/match_player/g2/delta/b.parquet", "kind": "delta"},
    ]}}}
    assert lake.base_files(man, base_url="https://x.test/") == {
        7: "https://x.test/v1/tables/match_player/g2/base/part-7/a.parquet"}


def test_partitions_for_window(tmp_path):
    con = extract.connect()
    days = {1: ["2026-09-10", "2026-09-14"], 2: ["2026-09-14", "2026-09-20"],
            3: ["2026-09-20", "2026-09-25"], 4: ["2026-09-25", "2026-10-01"]}
    files = {}
    for p, ds in days.items():
        f = tmp_path / f"part-{p}.parquet"
        df = pd.DataFrame({"start_time": [pd.Timestamp(d + " 23:30", tz="UTC") for d in ds]})
        con.register("df", df)
        con.execute(f"COPY df TO '{f.as_posix()}' (FORMAT PARQUET)")
        con.unregister("df")
        files[p] = f.as_posix()
    cache = tmp_path / "dates.json"
    # 23:30 UTC stays on the same UTC date: the session time zone is UTC
    assert lake.partition_dates(con, files, [2], cache)[2] == (date(2026, 9, 14), date(2026, 9, 20))
    assert lake.partitions_for_window(con, files, date(2026, 9, 17), date(2026, 9, 28),
                                      cache) == [2, 3, 4]
    assert lake.partitions_for_window(con, files, date(2026, 9, 21), date(2026, 9, 24),
                                      cache) == [3]


def test_parse_patches_and_windows():
    recs = [{"title": "09-29-2026"}, {"title": "09-16-2026 Update"},
            {"title": "08-22-2026 Update"}, {"title": "Developer Q&A"}]
    patches = parse_patches(recs)
    assert [n for n, _ in patches] == ["08-22-2026", "09-16-2026", "09-29-2026"]
    assert patch_window(patches, "09-16-2026") == (date(2026, 9, 17), date(2026, 9, 28))
    assert patch_window(patches, "09-29-2026") == (date(2026, 9, 30), None)
