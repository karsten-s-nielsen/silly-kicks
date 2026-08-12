import warnings

import pandas as pd
import pytest

import silly_kicks.calibration._features as _features_module
from silly_kicks.calibration._features import (
    _TRIAL_DEPENDENT_COLS,
    ALL_FEATURES,
    _compute_das,
    enrich_full,
    enrich_invariant,
    patch_trial_columns,
)

_CP = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}


def test_invariant_sets_trial_cols_nan_and_others_present(synth, frozen_xt):
    actions, frames, _home = synth
    base, links = enrich_invariant(actions=actions, frames=frames, xt=frozen_xt.xt, carrier_params=_CP)
    for col in _TRIAL_DEPENDENT_COLS:
        assert base[col].isna().all(), f"{col} must be a NaN placeholder in the invariant"
    # A non-trial tracking feature must be materialised (not all-NaN) for at least some rows.
    assert base["pressure_on_actor__andrienko_oval"].notna().any()
    assert "frame_id" in links.columns
    # Every model feature is present after the invariant pass.
    assert [c for c in ALL_FEATURES if c not in base.columns] == []


def test_patch_overwrites_exactly_the_trial_cols(synth, frozen_xt):
    actions, frames, home = synth
    base, links = enrich_invariant(actions=actions, frames=frames, xt=frozen_xt.xt, carrier_params=_CP)
    invariant_snapshot = base.drop(columns=_TRIAL_DEPENDENT_COLS).copy()
    patched = patch_trial_columns(
        base_actions=base,
        frames=frames,
        links=links,
        home_team_id=home,
        k3=2.0,
        pre_seconds=2.0,
        min_displacement_m=4.0,
    )
    # Trial cols are now populated...
    assert patched["pressure_on_actor__link_zones"].notna().any()
    assert patched["n_off_ball_runners_pre_window"].notna().any()
    # ...and NO invariant column changed.
    pd.testing.assert_frame_equal(
        patched.drop(columns=_TRIAL_DEPENDENT_COLS)[invariant_snapshot.columns],
        invariant_snapshot,
    )


def test_line_break_columns_are_not_features():
    assert not any("line_break" in c or "lines_broken" in c for c in ALL_FEATURES)


def test_all_features_count_matches_spec():
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in ALL_FEATURES


def test_enrich_full_is_independent_and_populates_trial_cols(synth, frozen_xt):
    actions, frames, home = synth
    full = enrich_full(
        actions=actions,
        frames=frames,
        xt=frozen_xt.xt,
        home_team_id=home,
        carrier_params=_CP,
        k3=1.5,
        pre_seconds=2.0,
        min_displacement_m=4.0,
    )
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in full.columns
    # The always-computed trial outputs are populated inline (NOT NaN placeholders). The
    # displacement/speed run cols can legitimately be all-NaN when no off-ball run is detected
    # in a short fixture — the column-parity test (full == invariant+patch) covers those.
    assert full["pressure_on_actor__link_zones"].notna().any()
    assert full["n_off_ball_runners_pre_window"].notna().any()


# ---------------------------------------------------------------------------
# ADR-043 -- the das_ok workaround is gone; DAS provenance is public
# ---------------------------------------------------------------------------


def _legacy_compute_das(actions, frames, links, carrier_params):
    """The pre-ADR-043 inline DAS computation, verbatim, as an oracle.

    ``_compute_das`` now routes through the public ``add_das`` instead of re-implementing
    the lookup. This oracle proves that routing moved no DAS value: the frames are
    pre-restricted to the linked ``(period_id, frame_id)`` pairs, so the direction
    ``add_das`` pins is inferred on exactly the frame set the library would otherwise have
    inferred it on.
    """
    import numpy as np

    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._das import DasUnscoreableError, get_individual_das

    carrier = infer_ball_carrier(
        frames,
        tolerance_m=carrier_params["tolerance_m"],
        beta=carrier_params["beta"],
        gamma=carrier_params["gamma"],
    )
    frames_with_tip = derive_team_in_possession(frames, carrier)
    linked = links[["action_id", "frame_id"]].dropna(subset=["frame_id"])
    linked = linked.merge(actions[["action_id", "period_id"]], on="action_id", how="left")
    linked_frame_ids = linked[["period_id", "frame_id"]].drop_duplicates()
    das_frames = frames_with_tip.merge(linked_frame_ids, on=["period_id", "frame_id"], how="inner")
    try:
        das_result = get_individual_das(das_frames, use_progress_bar=False, chunk_size=10)
    except DasUnscoreableError:
        # The public routing DEGRADES this exact class to NaN (das_source='unscoreable_call',
        # ADR-043); the oracle must degrade identically or the parity check compares a raise
        # to a degrade instead of value to value. All-NaN DAS makes `valid_rows` empty below,
        # which is precisely the all-NaN result `add_das` returns. Note this branch makes the
        # comparison VACUOUS -- the caller must prove a finite DAS exists before trusting it.
        das_result = das_frames.assign(DAS=float("nan"))
    player_rows = das_result[das_result["is_ball"] != True]  # noqa: E712
    valid_rows = player_rows.dropna(subset=["DAS"])
    das_lookup: dict[tuple, dict] = {}
    for (_pid, fid, tid), grp in valid_rows.groupby(["period_id", "frame_id", "team_id"]):
        das_lookup.setdefault((_pid, fid), {})[tid] = float(grp["DAS"].sum())
    pointer_lookup = links.set_index("action_id")
    team_vals = np.full(len(actions), np.nan)
    opp_vals = np.full(len(actions), np.nan)
    for pos, row in enumerate(actions.itertuples()):
        aid = row.action_id
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        key = (row.period_id, int(float(str(fid_raw))))
        if key not in das_lookup:
            continue
        team_id = row.team_id
        team_vals[pos] = das_lookup[key].get(team_id, np.nan)
        opp = [v for k, v in das_lookup[key].items() if k != team_id]
        if opp:
            opp_vals[pos] = opp[0]
    out = actions.copy()
    out["das_team"] = team_vals
    out["das_opponent"] = opp_vals
    out["das_diff"] = team_vals - opp_vals
    return out


def test_compute_das_values_are_unchanged_by_the_public_routing(synth):
    """Deleting the das_ok workaround must not move a single DAS number."""
    pytest.importorskip("accessible_space")
    from silly_kicks.tracking import link_actions_to_frames

    actions, frames, _home = synth
    links, _report = link_actions_to_frames(actions, frames)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = _compute_das(actions, frames, links, _CP)
        old = _legacy_compute_das(actions, frames, links, _CP)

    cols = ["das_team", "das_opponent", "das_diff"]
    pd.testing.assert_frame_equal(new[cols], old[cols])

    # Non-vacuity, asserted AFTER the comparison so the NaN-parity above still runs: an
    # all-NaN pair agrees trivially and proves nothing about the routing. Whether the
    # fixture is scoreable at all is an ENVIRONMENT property -- accessible-space cannot
    # simulate it once pandas infers `str` for the ball-carrier column -- so the honest
    # outcome there is a skip with the reason attached, never a silently vacuous pass.
    if not new["das_team"].notna().any():
        pytest.skip(
            "accessible-space scored no frame in this environment "
            f"(das_source={sorted(set(new['das_source']))}); the value comparison would be vacuous"
        )


def test_compute_das_emits_the_public_provenance_instead_of_a_private_flag(synth):
    """M8 is now served by das_source; _compute_das no longer returns a bool."""
    pytest.importorskip("accessible_space")
    from silly_kicks.tracking import DAS_SOURCE_VALUES, link_actions_to_frames

    actions, frames, _home = synth
    links, _report = link_actions_to_frames(actions, frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _compute_das(actions, frames, links, _CP)

    assert isinstance(out, pd.DataFrame), "_compute_das must return the frame alone (no das_ok tuple)"
    assert "das_source" in out.columns
    assert set(out["das_source"]) <= set(DAS_SOURCE_VALUES)


def test_das_ok_workaround_is_deleted():
    """The private flag must not come back: no `das_ok` IDENTIFIER in the calibration package.

    AST-based, not a substring scan: the surviving prose mentions of ``das_ok`` are the
    docstrings explaining that it is gone, and those must not trip the guard (a substring
    scan would force the explanation to be deleted along with the flag).
    """
    import ast
    import pathlib

    pkg = pathlib.Path(_features_module.__file__).parent
    offenders: list[str] = []
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == "das_ok") or (
                isinstance(node, ast.arg) and node.arg == "das_ok"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"das_ok resurrected at {offenders}; read das_source instead (ADR-043)"
