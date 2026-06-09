"""Owner-gated e2e: KDE-vs-Singh held-out transition-NLL on the real gold action-values mart.

Permanent, reproducible triangulation of the SK-xT-1 ~4% KDE win (4.17.0 ran it as a non-committed
one-off). Runs only where the owner Databricks credentials + databricks-sql-connector are reachable
(public CI skips). Thin orchestrator over unit-tested seams: scripts._loader_databricks (read+shape),
tests._xthreat_helpers.nll_relative_win / kde_clears_tripwire (verdict). Scores PASSES-ONLY (matches
the StatsBomb sibling + the lakehouse's published "Held-out NLL (passes)" 3.789->3.748). Hard asserts
on the FULL corpus only. See ADR-021 and docs/superpowers/specs/2026-06-09-xt-nll-lakehouse-e2e-design.md.
"""

import importlib.util
import os

import pytest

import scripts._loader_databricks as L
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import (
    GridSpec,
    KDEParams,
    compute_holdout_nll,
    holdout_split,
    kde_smoothed_transition_matrix,
    singh_transition_matrix,
)
from tests._xthreat_helpers import kde_clears_tripwire, nll_relative_win

_DBX_ENV = ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")


def _connector_available() -> bool:
    # find_spec("databricks.sql") imports the parent `databricks` to read its __path__, so it RAISES
    # ModuleNotFoundError (not returns None) when the connector is absent — guard it.
    try:
        return importlib.util.find_spec("databricks.sql") is not None
    except ModuleNotFoundError:
        return False


# Conservative floor for the KDE(4.0)-over-Singh relative held-out-NLL win. The 4.17.0 one-off
# measured ~4% at bandwidth>=4 on the full mart; this floor sits well below that so the tripwire
# tracks a real regression without flaking as the mart grows. See spec / ADR-021.
_MIN_RELATIVE_WIN = 0.015
_PROD_BANDWIDTH = 4.0  # held-out-optimal multiplier is >=4 on ~8.9M actions (full mart only)
_MIN_MAPPED = 0.95  # live-path coverage guard against mart-vocabulary drift

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in _DBX_ENV),
        reason="owner-tier Databricks credentials (DATABRICKS_HOST/HTTP_PATH/TOKEN)",
    ),
    pytest.mark.skipif(
        not _connector_available(),
        reason="databricks-sql-connector not importable (install in an isolated env, NOT the main .venv)",
    ),
]


def _nlls(train, holdout_passes, grid):
    singh = singh_transition_matrix(train, grid)
    kde4 = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=_PROD_BANDWIDTH))
    kde1 = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=1.0))
    return {
        "singh": compute_holdout_nll(singh, holdout_passes, grid=grid),
        "kde4": compute_holdout_nll(kde4, holdout_passes, grid=grid),
        "kde1": compute_holdout_nll(kde1, holdout_passes, grid=grid),
    }


def test_kde_beats_singh_on_holdout_nll_real_mart(capsys):
    max_matches = os.environ.get("XT_NLL_E2E_MAX_MATCHES")
    subsampled = max_matches is not None
    raw = L.fetch_action_values(max_matches=int(max_matches) if subsampled else None)
    assert len(raw) > 0, "gold mart returned no rows"

    actions = L.shape_action_values(raw)
    # Coverage guard: fail loud on mart-vocabulary drift instead of silently dropping every move.
    type_cov = actions["type_id"].notna().mean()
    result_cov = actions["result_id"].notna().mean()
    assert type_cov > _MIN_MAPPED, f"action_type vocab drift: only {type_cov:.1%} mapped to a SPADL type_id"
    assert result_cov > _MIN_MAPPED, f"action_result vocab drift: only {result_cov:.1%} mapped to a SPADL result_id"

    # Drop the <=5% unmapped rows so every downstream ==-mask is NA-free: a nullable-boolean <NA>
    # mask can raise ValueError on older pandas (the owner's <2.3.0 env); CI's 2.3.3 tolerates it,
    # so CI is blind to this path. Dropping here is correct regardless of the exact raising boundary.
    n_raw = len(actions)
    actions = actions.dropna(subset=["type_id", "result_id"])
    n_dropped = n_raw - len(actions)

    train, holdout = holdout_split(actions, holdout_fraction=0.15)
    # Score PASSES-ONLY (sibling + published-reference parity); fit on the full train.
    pass_id = spadlconfig.actiontype_id["pass"]
    holdout_passes = holdout[holdout["type_id"] == pass_id]
    assert len(train) > 0 and len(holdout_passes) > 0

    grid_default, grid_lakehouse = GridSpec(16, 12), GridSpec(12, 8)
    nll_d = _nlls(train, holdout_passes, grid_default)
    nll_l = _nlls(train, holdout_passes, grid_lakehouse)

    with capsys.disabled():
        print("\n=== xT held-out transition-NLL cross-check (gold mart; scored=passes,success) ===")
        print(
            f"n_actions={len(actions)} (dropped {n_dropped} unmapped)  "
            f"n_train_matches={train['game_id'].nunique()}  "
            f"n_holdout_pass={len(holdout_passes)}  subsampled={subsampled}"
        )
        for label, d in (("16x12 (default)", nll_d), ("12x8 (lakehouse)", nll_l)):
            print(
                f"[{label}] singh={d['singh']:.5f}  "
                f"kde@1.0={d['kde1']:.5f} ({nll_relative_win(d['singh'], d['kde1']) * 100:+.2f}%)  "
                f"kde@4.0={d['kde4']:.5f} ({nll_relative_win(d['singh'], d['kde4']) * 100:+.2f}%)"
            )
        print("Lakehouse published reference (12x8, passes): singh 3.78924 -> kde ~3.748")

    if subsampled:
        pytest.skip(
            "XT_NLL_E2E_MAX_MATCHES set: bandwidth=4.0 is tuned for the full mart; "
            "subsampled run is log-only (see printed block above)."
        )

    # Hard tripwire — FULL corpus, silly-kicks default resolution (16x12), passes-only. Two checks:
    #  (1) the TUNED KDE(4.0) clears the sensitivity floor (the primary, most-sensitive tripwire);
    #  (2) the SHIPPED-DEFAULT KDE(1.0) STRICTLY beats Singh (its documented contract). No floor on
    #      (2): the default's margin erodes as the mart grows (the held-out-optimal bandwidth
    #      multiplier rises with corpus size, so a fixed 1.0 under-smooths more), e.g. ~+8.7% on a
    #      300-match smoke -> ~+3.0% on the full ~9.6M-action mart. A floor would trip on benign
    #      growth; strict-beat (floor=0.0) catches only a real "default stops beating Singh" regression.
    assert kde_clears_tripwire(nll_d["singh"], nll_d["kde4"], floor=_MIN_RELATIVE_WIN), (
        f"KDE(4.0) failed the tripwire at 16x12 (need strict-beat AND rel>={_MIN_RELATIVE_WIN}): {nll_d}"
    )
    assert kde_clears_tripwire(nll_d["singh"], nll_d["kde1"], floor=0.0), (
        f"shipped-default KDE(1.0) no longer strictly beats Singh at 16x12 (full mart): {nll_d}"
    )
