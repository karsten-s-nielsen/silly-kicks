import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import compute_bravery


def _final_actions(rows):
    base: dict[str, object] = dict(
        game_id="g1",
        period_id=1,
        bodypart_id=spadlconfig.bodypart_id["foot"],
        start_x=90.0,
        start_y=34.0,
        end_x=100.0,
        end_y=40.0,
    )
    # A benign team-20 pass so BOTH teams are present in the match (the two-team opponent
    # inference needs team 20 to appear somewhere; it is not a shot/cross so it never counts).
    rows = [*list(rows), dict(type_name="pass", result_name="success", team_id=20, player_id=99)]
    out = []
    for i, r in enumerate(rows):
        d: dict[str, object] = dict(base)
        d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[str(d.pop("type_name"))]
        d["result_id"] = spadlconfig.result_id[str(d.pop("result_name", "fail"))]
        d.setdefault("shot_blocked", pd.NA)
        d.setdefault("cross_blocked", pd.NA)
        # ADR-133: the provider's own label. Default: the native label equals the given type (a native cross is
        # a cross); a test re-types rows explicitly to separate the two.
        d.setdefault("cross_native", _native_default(d["type_id"]))
        out.append(d)
    df = pd.DataFrame(out)
    df["shot_blocked"] = pd.array(df["shot_blocked"].tolist(), dtype="boolean")
    df["cross_blocked"] = pd.array(df["cross_blocked"].tolist(), dtype="boolean")
    df["cross_native"] = pd.array(df["cross_native"].tolist(), dtype="boolean")
    return df


_CROSSED = {spadlconfig.actiontype_id[t] for t in ("cross", "corner_crossed", "freekick_crossed")}
_SHORT = {spadlconfig.actiontype_id[t] for t in ("pass", "corner_short", "freekick_short")}


def _native_default(type_id):
    if type_id in _CROSSED:
        return True
    if type_id in _SHORT:
        return False
    return pd.NA


def test_bravery_worked_example_shots_only():
    # team 20 faces 5 shots by team 10, 4 blocked -> bravery_shots = 0.8
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=(i < 4)) for i in range(5)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]  # the DEFENDING team
    assert row["bravery_shots"] == pytest.approx(0.8)
    assert row["n_shots_faced"] == 5


def test_set_piece_crosses_are_exposed_not_dropped():
    rows = [
        dict(type_name="shot", team_id=10, player_id=1, shot_blocked=True),
        dict(type_name="cross", team_id=10, player_id=2, cross_blocked=True),  # open-play
        dict(type_name="corner_crossed", team_id=10, player_id=3, cross_blocked=pd.NA),  # set-piece
        dict(type_name="freekick_crossed", team_id=10, player_id=4, cross_blocked=pd.NA),
    ]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_set_piece_crosses"])  # NaN, never 0
    assert row["n_set_piece_crosses_faced"] == 2  # the gap is exposed
    # headline is over the KNOWN domain (1 shot + 1 open-play cross, both blocked = 1.0),
    # UNCHANGED by the set-piece crosses:
    assert row["bravery_pct_known_domain"] == pytest.approx(1.0)


def test_all_na_cross_column_yields_nan_open_play_component():
    rows = [dict(type_name="cross", team_id=10, player_id=2, cross_blocked=pd.NA) for _ in range(3)]
    out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_open_play_crosses"])  # unknown -> NaN, not 0


def _retyped(native_rows, retype):
    """The same native actions after the geometric re-typing: ``retype`` maps row index -> new type name."""
    df = _final_actions(native_rows)
    for i, name in retype.items():
        df.loc[i, "type_id"] = spadlconfig.actiontype_id[name]
    return df


_NATIVE_ROWS = [
    dict(type_name="shot", team_id=10, player_id=1, shot_blocked=True),
    dict(type_name="shot", team_id=10, player_id=1, shot_blocked=False),
    dict(type_name="cross", team_id=10, player_id=2, cross_blocked=True),  # native cross, blocked
    dict(type_name="cross", team_id=10, player_id=2, cross_blocked=False),  # native cross, delivered
    dict(type_name="cross", team_id=10, player_id=2, cross_blocked=False),  # native cross, short delivery
    dict(type_name="pass", team_id=10, player_id=5),  # native pass that the geometry calls a cross
    dict(type_name="corner_crossed", team_id=10, player_id=3),
    dict(type_name="corner_short", team_id=10, player_id=3),  # native short corner delivered into the box
]


def test_bravery_unchanged_under_retyping():
    # ADR-133: bravery reads the provider's own label (cross_native), so the geometric re-typing of the SPADL
    # type moves no bravery output.
    before = compute_bravery(_final_actions(_NATIVE_ROWS))
    after = compute_bravery(_retyped(_NATIVE_ROWS, {2: "pass", 4: "pass", 5: "cross"}))  # open play only
    pd.testing.assert_frame_equal(before, after)
    row = after[after["team_id"] == 20].iloc[0]
    assert row["n_open_play_crosses_faced"] == 3
    assert row["n_set_piece_crosses_faced"] == 1
    assert row["bravery_open_play_crosses"] == pytest.approx(1 / 3)


def test_bravery_atomic_matches_standard():
    from silly_kicks.atomic.tracking.features import compute_bravery as atomic_compute_bravery

    actions = _retyped(_NATIVE_ROWS, {2: "pass", 5: "cross"})
    std = compute_bravery(actions)
    atomic = atomic_compute_bravery(actions)
    shared = [c for c in std.columns if c not in ("bravery_set_piece_crosses", "n_set_piece_crosses_faced")]
    pd.testing.assert_frame_equal(std[shared], atomic[shared])


@pytest.mark.parametrize("atomic", [False, True])
def test_bravery_requires_cross_native(atomic):
    actions = _final_actions(_NATIVE_ROWS).drop(columns="cross_native")
    if atomic:
        from silly_kicks.atomic.tracking.features import compute_bravery as fn
    else:
        fn = compute_bravery
    with pytest.raises(ValueError, match="cross_native"):
        fn(actions)


def test_both_columns_na_yields_nan_headline_and_warns():
    rows = [dict(type_name="shot", team_id=10, player_id=1, shot_blocked=pd.NA)]
    with pytest.warns(UserWarning, match="bravery"):
        out = compute_bravery(_final_actions(rows))
    row = out[out["team_id"] == 20].iloc[0]
    assert np.isnan(row["bravery_pct_known_domain"])


def _defending(out):
    """The defending team's (team 20's) bravery row."""
    return out[out["team_id"] == 20].iloc[0]


def _bravery_fn(atomic):
    if atomic:
        from silly_kicks.atomic.tracking.features import compute_bravery as fn

        return fn
    return compute_bravery


_UNLABELLED_ROWS = [
    dict(type_name="shot", team_id=10, player_id=1, shot_blocked=True),
    dict(type_name="shot", team_id=10, player_id=1, shot_blocked=False),
    dict(type_name="cross", team_id=10, player_id=2, cross_blocked=True),
    dict(type_name="cross", team_id=10, player_id=2, cross_blocked=False),
    dict(type_name="pass", team_id=10, player_id=5),
]


@pytest.mark.parametrize("atomic", [False, True])
def test_open_play_count_is_unknown_without_a_provider_label(atomic):
    # ADR-133 + ADR-027: a game whose open-play pass-class rows carry no known cross_native (SkillCorner) has an
    # unknown open-play cross count: <NA> and NaN, never 0, and the headline counts the shots only.
    unlabelled = _final_actions(_UNLABELLED_ROWS)
    unlabelled["cross_native"] = pd.array([pd.NA] * len(unlabelled), dtype="boolean")
    row = _defending(_bravery_fn(atomic)(unlabelled))
    assert pd.isna(row["n_open_play_crosses_faced"])
    assert np.isnan(row["bravery_open_play_crosses"])
    assert row["bravery_pct_known_domain"] == pytest.approx(0.5)  # 1 of 2 shots blocked; no cross counted
    assert row["n_shots_faced"] == 2
    # The other side: the same rows with the provider's label keep their count.
    labelled = _defending(_bravery_fn(atomic)(_final_actions(_UNLABELLED_ROWS)))
    assert labelled["n_open_play_crosses_faced"] == 2
    assert labelled["bravery_open_play_crosses"] == pytest.approx(0.5)
    assert labelled["bravery_pct_known_domain"] == pytest.approx(0.5)  # 2 of 4 shots and crosses blocked
    # A labelled set piece does not make the open-play label known: the open-play count stays unknown.
    corner = _final_actions([*_UNLABELLED_ROWS, dict(type_name="corner_crossed", team_id=10, player_id=3)])
    open_play = corner["type_id"].isin([spadlconfig.actiontype_id["pass"], spadlconfig.actiontype_id["cross"]])
    corner.loc[open_play, "cross_native"] = pd.NA
    assert corner["cross_native"].notna().any()  # precondition: the corner keeps its label
    assert pd.isna(_defending(_bravery_fn(atomic)(corner))["n_open_play_crosses_faced"])


def test_the_open_play_label_is_judged_per_game():
    # A corpus of a labelled and an unlabelled game: only the unlabelled game's count is unknown.
    g1 = _final_actions(_UNLABELLED_ROWS)
    g2 = _final_actions(_UNLABELLED_ROWS).assign(game_id="g2")
    g2["cross_native"] = pd.array([pd.NA] * len(g2), dtype="boolean")
    out = compute_bravery(pd.concat([g1, g2], ignore_index=True))
    assert _defending(out[out["game_id"] == "g1"])["n_open_play_crosses_faced"] == 2
    assert pd.isna(_defending(out[out["game_id"] == "g2"])["n_open_play_crosses_faced"])


@pytest.mark.parametrize("label", [pd.NA, True, False], ids=["no-label", "labelled-crossed", "labelled-short"])
def test_set_piece_crosses_faced_are_counted_by_type(label):
    # Set pieces are not re-typed (ADR-133), so the type is the provider's label: SkillCorner and Metrica
    # set-piece rows carry cross_native <NA> and still count, as before ADR-133; a short set piece never counts.
    rows = [
        dict(type_name="corner_crossed", team_id=10, player_id=3, cross_native=label),
        dict(type_name="freekick_crossed", team_id=10, player_id=4, cross_native=label),
        dict(type_name="corner_short", team_id=10, player_id=3, cross_native=label),
        dict(type_name="freekick_short", team_id=10, player_id=4, cross_native=label),
    ]
    row = _defending(compute_bravery(_final_actions(rows)))
    assert row["n_set_piece_crosses_faced"] == 2
    assert np.isnan(row["bravery_set_piece_crosses"])
