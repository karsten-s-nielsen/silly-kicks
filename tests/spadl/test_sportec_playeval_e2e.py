"""Owner-gated e2e: the native sportec play_evaluation success-allowlist on the real 7 IDSSE matches.

Runs the NATIVE silly_kicks.spadl.sportec converter on Databricks bronze.idsse_events (the only path
that surfaces the raw play_evaluation token to the converter this PR changes -- pining's IDSSE loader
parses via the kloppy gateway, which never exposes it). Asserts the success-allowlist is warn-silent
on real DFL data (allowlist u {unsuccessful} covers the vocabulary -> byte-identical to 4.20.1) and
the BUG-2 mechanism is still live (goalkick fail-rate in a plausible band). Skips in public CI; needs
the owner Databricks credentials + databricks-sql-connector (install in an isolated env, NOT the main
.venv -- the connector pins pandas<2.3.0). See the design spec under docs/superpowers/specs/ (2026-06-09).
"""

import importlib.util
import os
import warnings

import pytest

import scripts._loader_databricks as L
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl import sportec as sportec_spadl

# NOTE: deliberately NO `silly_kicks.tracking` import -- it transitively pulls xgboost + numba +
# sklearn (verified), which a SPADL-completion e2e must not drag in. ET is dropped inline below.

_DBX_ENV = ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")
_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_FAIL = spadlconfig.result_id["fail"]


def _connector_available() -> bool:
    try:
        return importlib.util.find_spec("databricks.sql") is not None
    except ModuleNotFoundError:
        return False


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


def test_play_evaluation_allowlist_on_real_idsse(capsys):
    raw = L.fetch_idsse_events()
    assert len(raw) > 0, "bronze.idsse_events returned no rows"
    assert "play_evaluation" in raw.columns, "bronze.idsse_events missing play_evaluation"

    gk_total = 0
    gk_fail = 0
    caught: list[str] = []
    for _match_id, ev in raw.groupby("match_id"):
        # Defensive ET drop (Bundesliga has no ET, but the native converter RAISES on ET-without-flag,
        # ADR-010). Inline + dtype-robust (drops periods 3/4 whether int or str) -> no tracking import.
        ev = ev[~ev["period"].astype(str).isin(["3", "4"])]
        if ev.empty:
            continue
        # home_team_id is orientation-only; result_id from play_evaluation is orientation-independent.
        home = str(ev["team"].dropna().mode().iloc[0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actions, _ = sportec_spadl.convert_to_actions(ev, home_team_id=home, home_team_start_left=True)
        caught += [str(x.message) for x in w if "unexpected play_evaluation" in str(x.message)]
        gk = actions[actions["type_id"] == _GOALKICK]
        gk_total += len(gk)
        gk_fail += int((gk["result_id"] == _FAIL).sum())

    fail_rate = gk_fail / gk_total if gk_total else float("nan")
    with capsys.disabled():
        print("\n=== sportec play_evaluation allowlist e2e (bronze IDSSE) ===")
        print(f"matches={raw['match_id'].nunique()}  goalkicks={gk_total}  goalkick_fail_rate={fail_rate:.3f}")
        print(f"observed raw play_evaluation values: {sorted(set(raw['play_evaluation'].dropna().astype(str)))}")

    # (1) Warn-silent: allowlist u {unsuccessful} covers the real vocabulary (byte-identical condition).
    assert not caught, f"unexpected play_evaluation token(s) on real IDSSE: {caught}"
    # (2) Liveness band: BUG-2 mechanism live (fails exist) AND not an all-fail regression.
    assert gk_total > 0, "no goalkicks found"
    assert 0.05 <= fail_rate <= 0.60, f"goalkick fail-rate {fail_rate:.3f} outside [0.05, 0.60]"
