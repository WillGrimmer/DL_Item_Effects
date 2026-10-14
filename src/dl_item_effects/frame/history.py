"""Player-history covariates from a window's ``_history.parquet`` (PLAN.md §8, 2026-10-14).

Each player-match gets features of the same account's Ranked matches, ordered by
(start_time, match_id). Only strictly earlier matches count; the current one is excluded:

* ``prev_won``: the previous match's result (equal to the extractor's ``nc_prev_won``).
* ``wr_2_6``, ``wr_2_11``: win rate over matches 2 to 6 and 2 to 11 back, among those
  available; NaN if there are none. They start two back so that the §4.1 control (whose
  outcome is the previous result) can use them without seeing its own outcome.
* ``n_hist``: number of earlier matches in the history file, capped at 11.
* ``games_24h``: Ranked matches started in the 24 hours before this one (strictly earlier
  start times; a match at the same second is not counted).

``nc_next_won``, the result of the account's next Ranked match, is a diagnostic outcome and
is never a covariate. ``prev_gap_h`` (from the extractor) joins the history covariates in
:data:`HISTORY_COLUMNS`. History covers about two weeks before each window, so ``n_hist``
reaches its cap for most rows.
"""
from __future__ import annotations

import duckdb
import pandas as pd

HISTORY_COLUMNS = ["prev_won", "wr_2_6", "wr_2_11", "n_hist", "games_24h", "prev_gap_h"]
N_HIST_CAP = 11

_SQL = f"""
SELECT account_id, match_id,
    lag(won) OVER w AS prev_won,
    avg(won::DOUBLE) OVER (PARTITION BY account_id ORDER BY start_time, match_id
                           ROWS BETWEEN 6 PRECEDING AND 2 PRECEDING) AS wr_2_6,
    avg(won::DOUBLE) OVER (PARTITION BY account_id ORDER BY start_time, match_id
                           ROWS BETWEEN 11 PRECEDING AND 2 PRECEDING) AS wr_2_11,
    least(row_number() OVER w - 1, {N_HIST_CAP}) AS n_hist,
    count(*) OVER (PARTITION BY account_id ORDER BY start_time
                   RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW)
      - count(*) OVER (PARTITION BY account_id ORDER BY start_time
                       RANGE BETWEEN CURRENT ROW AND CURRENT ROW) AS games_24h,
    lead(won) OVER w AS nc_next_won
FROM hist
WINDOW w AS (PARTITION BY account_id ORDER BY start_time, match_id)
"""


def history_features(hist: str | pd.DataFrame, accounts=None,
                     con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Features for every (account_id, match_id) in ``hist`` (a parquet path or a frame with
    account_id, match_id, start_time, won). ``accounts`` restricts to those accounts first."""
    con = con or duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    src = (f"read_parquet('{hist}')" if isinstance(hist, str) else "hist_df")
    if not isinstance(hist, str):
        con.register("hist_df", hist)
    where = ""
    if accounts is not None:
        con.register("acc_df", pd.DataFrame({"account_id": pd.unique(pd.Series(accounts))}))
        where = "WHERE account_id IN (SELECT account_id FROM acc_df)"
    con.execute(f"CREATE OR REPLACE TEMP TABLE hist AS SELECT account_id, match_id, "
                f"start_time, won FROM {src} {where}")
    out = con.execute(_SQL).df()
    for c in ("prev_won", "nc_next_won"):
        out[c] = out[c].astype("float64")
    out["n_hist"] = out["n_hist"].astype("int64")
    out["games_24h"] = out["games_24h"].astype("int64")
    out["account_id"] = out["account_id"].astype("int64")
    out["match_id"] = out["match_id"].astype("int64")
    return out


def add_player_history(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Join history features onto a decision frame by (match_id, account_id). A player-match
    missing from the history gets NaN features and ``n_hist`` 0."""
    cols = [c for c in HISTORY_COLUMNS if c != "prev_gap_h"] + ["nc_next_won"]
    f = feats[["match_id", "account_id"] + cols]
    left = df.drop(columns=[c for c in cols if c in df.columns])
    key = left[["match_id", "account_id"]].astype("int64")
    m = key.merge(f, on=["match_id", "account_id"], how="left")
    out = left.copy()
    for c in cols:
        out[c] = m[c].to_numpy()
    out["n_hist"] = out["n_hist"].fillna(0)
    out["games_24h"] = out["games_24h"].fillna(0)
    out.attrs = {**df.attrs,
                 "history_cols": [c for c in HISTORY_COLUMNS if c in out.columns]}
    return out
