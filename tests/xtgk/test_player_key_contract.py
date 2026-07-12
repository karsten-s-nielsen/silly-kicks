"""W3 (4.45.0): the GK-domain keeper-id data contract.

Convention: GK-domain consumers use the RESOLVED `player_key`, never raw `player_id` (null for goal-kicks
by SPADL design -- the reason `acting_gk_from_frames` exists). Analysis loaders source resolved fields from
GOLD marts, not raw `bronze.spadl_actions`. These guards catch a wrong-column read (the bug the earlier
investigation hit) AND a resolver regression.
"""

import pytest


def test_xtgk_loader_sources_keeper_from_gold_player_key_not_bronze():
    # pure SQL-source guard (CI): the resolved keeper comes from fct_action_context (c.player_key),
    # never from raw bronze.spadl_actions (which is null for goal-kicks).
    import scripts._loader_databricks as L

    assert "c.player_key" in L._XTGK_ACTIONS_SQL
    assert "s.player_key" not in L._XTGK_ACTIONS_SQL  # never the raw-bronze keeper source


@pytest.mark.e2e
def test_player_key_non_null_at_least_99pct_on_gk_distribution():
    # live data-contract: catches a wrong-column read AND a resolver regression (current live value 99.9%).
    import sys

    sys.path.insert(0, "scripts")
    from _loader_databricks import _connect, _query_param  # type: ignore[import-not-found]

    conn = _connect()
    try:
        r = _query_param(
            conn.cursor(),
            """
            SELECT SUM(CASE WHEN is_gk_distribution THEN 1 ELSE 0 END) AS n,
                   SUM(CASE WHEN is_gk_distribution AND player_key IS NOT NULL THEN 1 ELSE 0 END) AS n_key
            FROM soccer_analytics.dev_gold.fct_action_context
            """,
            {},
        )
    finally:
        conn.close()
    n, n_key = int(r["n"].iloc[0]), int(r["n_key"].iloc[0])
    assert n > 0 and n_key / n >= 0.99, f"player_key non-null {n_key}/{n} < 99% -- wrong column or resolver regression"
