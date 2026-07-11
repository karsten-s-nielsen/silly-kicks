"""rho loader/trainer GK-distribution domain: self-adapting is_gk_distribution + NULL coalescing."""

import numpy as np
import pandas as pd


def test_retention_sql_is_unconditional_and_probe_helpers_gone():
    # Part B: the transitional self-adapting probe is collapsed -- is_gk_distribution is now a HARD
    # dependency (lakehouse F1 materialized it), selected unconditionally; the probe helpers are removed.
    import scripts._loader_databricks as L

    assert "c.is_gk_distribution" in L._RETENTION_SQL
    assert "{is_gk_distribution_select}" not in L._RETENTION_SQL  # no template hole
    assert "gk_was_distributing" not in L._RETENTION_SQL
    assert not hasattr(L, "should_select_is_gk_distribution")
    assert not hasattr(L, "_build_retention_sql")
    assert not hasattr(L, "_IS_GK_DISTRIBUTION_PROBE")
    assert not hasattr(L, "_RETENTION_SQL_TEMPLATE")


def _domain_actions(is_gk_col=None):
    # 6 GK-distribution rows EARLY (3 goalkick + 3 GK-pass by player 1) + outfield filler spanning
    # to t=60 so the early rows' 10s retains() windows are fully observed (finite labels). is_gk_col
    # (when given) marks the 3 GK-passes True; goalkicks are covered by the actor-independent type term.
    n = 31
    type_id = [22, 22, 22, 0, 0, 0] + [0] * (n - 6)  # 3 goalkick, 3 GK-pass, rest outfield pass
    player_id = [1, 1, 1, 1, 1, 1] + [10] * (n - 6)
    df = pd.DataFrame(
        {
            "game_id": [1] * n,
            "action_id": range(n),
            "period_id": [1] * n,
            "time_seconds": np.arange(n, dtype=float) * 2.0,  # 0..60s
            "team_id": [5] * n,
            "player_id": player_id,
            "type_id": type_id,
            "result_id": [1] * n,
            "start_x": np.linspace(5, 50, n),
            "start_y": [34.0] * n,
            "end_x": np.linspace(20, 70, n),
            "end_y": [34.0] * n,
            "pressure": [0.3] * n,
        }
    )
    if is_gk_col is not None:
        marks = [False, False, False, True, True, True] + [False] * (n - 6)
        df["is_gk_distribution"] = pd.Series(marks, dtype=is_gk_col) if isinstance(is_gk_col, str) else marks
    return df


def test_domain_present_nonnull_broadens_beyond_goalkicks():
    from scripts.train_gk_retention import prepare_retention_training_data

    # is_gk_distribution True for the 3 GK-passes -> domain = 3 goalkicks + 3 GK-passes (all early,
    # windows observed). Broader than the 3 goalkicks alone.
    X_full, _, _ = prepare_retention_training_data(_domain_actions("boolean"))
    X_goalkicks, _, _ = prepare_retention_training_data(_domain_actions(None))
    assert len(X_full) > len(X_goalkicks)  # the GK-passes broadened the domain
    assert len(X_full) >= 4


def test_domain_present_null_coalesces_to_false_goalkicks_only():
    from scripts.train_gk_retention import prepare_retention_training_data

    # The rollout population: column exists but is NULL everywhere -> coalesced to False -> goalkicks-only.
    null_actions = _domain_actions(None)
    null_actions["is_gk_distribution"] = pd.Series([pd.NA] * len(null_actions), dtype="boolean")
    X_null, *_ = prepare_retention_training_data(null_actions)
    X_absent, *_ = prepare_retention_training_data(_domain_actions(None))
    assert len(X_null) == len(X_absent)  # NULL == absent (goalkicks-only), NOT dropped/corrupted


def test_dropped_column_is_not_a_feature():
    # Loader-drop safety: removing gk_was_distributing can't break inference (it's a domain input,
    # never a model feature). The invariant is asserted UNCONDITIONALLY against the always-available
    # feature-name list (so a weights-less CI still exercises the guarantee), with the bundled default
    # model as an ADDITIONAL layer when present.
    from silly_kicks.xtgk._retention_features import RETENTION_FEATURE_NAMES

    assert "gk_was_distributing" not in RETENTION_FEATURE_NAMES  # unconditional core guarantee
    assert "is_gk_distribution" not in RETENTION_FEATURE_NAMES

    from silly_kicks.xtgk._retention import GkRetentionModel

    try:
        m = GkRetentionModel.from_variant("default")
    except FileNotFoundError:
        return  # weights not bundled -> the unconditional assertions above still ran
    assert "gk_was_distributing" not in m.feature_names
    assert list(m.feature_names) == RETENTION_FEATURE_NAMES
