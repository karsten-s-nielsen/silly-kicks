"""W2: the columnar spearman frame index and the batched combine (spec 2026-11-08 tracking-batch §4.2, plan Task 7).

Every surface is bit-identical (``scripts._bitwise.assert_bits_equal``, per field) to the per-frame public call
``compute_pitch_control`` on the request's frame, in the same test run (its own ``compute_spearman`` path), on both
backends: ``SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY`` unset (numba) and ``"1"`` (numpy reference). The frozen
pre-change oracle ended with ruling C5 (spec 2026-11-08 tracking-batch, section 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks._frame_index import group_rows
from silly_kicks.tracking.pitch_control import _dispatch, _spearman_batch, compute_pitch_control
from silly_kicks.tracking.pitch_control._params import SpearmanParams
from tests._perf_structural import call_counter
from tests._prod_fixtures import make_production_frames

_KEYS = ("game_id", "period_id", "frame_id")
_FIELDS = (
    "grid_x",
    "grid_y",
    "surface",
    "method",
    "attacking_team_id",
    "per_player_influence",
    "player_ids",
    "player_team_ids",
)


@pytest.fixture(params=["numba", "numpy"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setenv("SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY", "1")
    else:
        monkeypatch.delenv("SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY", raising=False)
    return request.param


def _with_velocity(frames: pd.DataFrame) -> pd.DataFrame:
    """Seeded ``vx``/``vy`` (spearman needs velocity), with about 2 % NaN to exercise the NaN -> 0.0 rule."""
    rng = np.random.default_rng(20261111)
    n = len(frames)
    vx, vy = rng.normal(0.0, 2.5, n), rng.normal(0.0, 2.5, n)
    vx[rng.random(n) < 0.02] = np.nan
    return frames.assign(
        vx=pd.Series(vx, index=frames.index).astype(frames["x"].dtype),
        vy=pd.Series(vy, index=frames.index).astype(frames["y"].dtype),
    )


def _frames(*, mixed_ids: bool = False, ball: bool = True, n_frames: int = 12) -> pd.DataFrame:
    frames = _with_velocity(make_production_frames(n_frames=n_frames, n_periods=2, mixed_ids=mixed_ids))
    if not ball:
        frames = frames[~frames["is_ball"].astype(bool)].reset_index(drop=True)
    return frames


def _requests(frames: pd.DataFrame, decompose: bool) -> list:
    keys = frames[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)
    teams = [t for t in pd.unique(frames["team_id"].dropna())]
    return [(tuple(k), team, decompose) for k in keys for team in teams]


def assert_surfaces_equal(new: list, old: list) -> None:
    assert len(new) == len(old) > 0
    for n, o in zip(new, old, strict=True):
        for field in _FIELDS:
            a, b = getattr(n, field), getattr(o, field)
            if a is None or b is None:
                assert a is None and b is None, field
            elif isinstance(a, np.ndarray):
                assert_bits_equal(a, b)
            else:
                assert type(a) is type(b) and a == b, (field, a, b)


def _per_frame(frames: pd.DataFrame, requests: list, params: SpearmanParams) -> list:
    """The reference: the per-frame public call on each request's frame slice (rows in frame order)."""
    by_key = frames.groupby(list(_KEYS), sort=False, observed=True)
    return [
        compute_pitch_control(by_key.get_group(tuple(key)), team, method="spearman", params=params, decompose=dec)
        for key, team, dec in requests
    ]


def _new_and_old(frames: pd.DataFrame, requests: list):
    groups = group_rows(frames, _KEYS)
    params = SpearmanParams()
    new = _dispatch._batch_from_groups(groups, requests, method="spearman", params=params)
    return new, _per_frame(frames, requests, params)


@pytest.mark.parametrize("decompose", [False, True])
@pytest.mark.parametrize("ball", [True, False])
def test_batch_bit_identical_production(backend, decompose, ball):
    frames = _frames(ball=ball)
    new, old = _new_and_old(frames, _requests(frames, decompose))
    assert_surfaces_equal(new, old)


def _edge_frames() -> pd.DataFrame:
    """Frames 1-4 of one game: an empty frame (ball only), one team only (no defenders for team 5, no attackers
    for team 6), a normal frame, and a frame whose players all have NaN positions."""
    base = _with_velocity(make_production_frames(n_frames=4, n_periods=1))
    fids = sorted(base["frame_id"].unique())
    is_ball = base["is_ball"].astype(bool)
    empty = base[(base["frame_id"] == fids[0]) & is_ball]
    one_team = base[(base["frame_id"] == fids[1]) & (is_ball | (base["team_id"] == 5))]
    normal = base[base["frame_id"] == fids[2]]
    nan_pos = base[base["frame_id"] == fids[3]].copy()
    nan_pos.loc[~nan_pos["is_ball"].astype(bool), ["x", "y"]] = np.nan
    return pd.concat([empty, one_team, normal, nan_pos], ignore_index=True)


@pytest.mark.parametrize("decompose", [False, True])
def test_batch_edge_frames(backend, decompose):
    frames = _edge_frames()
    keys = [tuple(k) for k in frames[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)]
    requests = [(k, team, decompose) for k in keys for team in (5, 6)]
    requests += [requests[2], requests[3], requests[2]]  # duplicate requests, both teams on one frame again
    new, old = _new_and_old(frames, requests)
    assert_surfaces_equal(new, old)


@pytest.mark.parametrize("decompose", [False, True])
def test_batch_mixed_id_content(backend, decompose):
    """Frames whose team ids are ints next to frames whose team ids are strings (``mixed_ids``): ``ids_match``
    takes its raw path on some requests and its canonical path on others, in one batch."""
    frames = _frames(mixed_ids=True)
    keys = [tuple(k) for k in frames[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)]
    requests = [(k, team, decompose) for k in keys for team in (5, "6")]
    new, old = _new_and_old(frames, requests)
    assert_surfaces_equal(new, old)


def test_slice_index_two_slices_one_key(backend):
    """The cover-shadow pattern: a frame and its keeper-removed counterfactual share one frame key and are scored
    as two distinct slices (never deduplicated by frame id)."""
    frames = _frames()
    fid = frames["frame_id"].iloc[0]
    frame = frames[frames["frame_id"] == fid].reset_index(drop=True)
    no_gk = frame[~frame["is_goalkeeper"].astype(bool)].reset_index(drop=True)
    slices = [frame, no_gk, frame]
    teams = [5, 5, 6]
    decs = [True, True, False]
    balls = [_dispatch._resolve_ball_position(s, None) for s in slices]
    params = SpearmanParams()
    new = _spearman_batch.compute_spearman_batch(slices, teams, decs, balls, params=params)
    old = [
        compute_pitch_control(s, team, method="spearman", params=params, decompose=dec, ball_position=ball)
        for s, team, dec, ball in zip(slices, teams, decs, balls, strict=True)
    ]
    assert_surfaces_equal(new, old)
    assert not np.array_equal(new[0].surface, new[1].surface)  # non-vacuity: the counterfactual differs


def test_categorical_team_id_identical(backend):
    """Review Focus 3: a ``category`` team id (ADR-106 storage) gives the same surfaces."""
    frames = _frames()
    cat = frames.assign(team_id=frames["team_id"].astype("category"))
    requests = _requests(frames, True)
    new, _ = _new_and_old(cat, requests)
    _, old = _new_and_old(frames, requests)
    assert_surfaces_equal(new, old)


def test_dispatch_key_exact_type(backend):
    """R2 detection: requests for team ``1`` and team ``"1"`` on one frame are two requests, each surface reporting
    its own ``attacking_team_id`` (the old ``str()`` dedup key collapsed them)."""
    frames = _frames().assign(team_id=lambda f: f["team_id"].map({5: 1, 6: 2}).astype(object))
    key = tuple(frames[list(_KEYS)].iloc[0])
    groups = group_rows(frames, _KEYS)
    surfaces = _dispatch._batch_from_groups(groups, [(key, 1, False), (key, "1", False)], method="spearman")
    assert [type(s.attacking_team_id) for s in surfaces] == [int, str]
    assert surfaces[0].attacking_team_id == 1 and surfaces[1].attacking_team_id == "1"


# ---- structural guards (per chunk) ----


def test_structure_tti_once_influence_constant_extraction_per_frame(backend, monkeypatch):
    from silly_kicks.tracking.pitch_control import _index

    sizes: list[int] = []
    real_ids_match = _spearman_batch.ids_match

    def counting_ids_match(series, value):
        sizes.append(len(series))
        return real_ids_match(series, value)

    monkeypatch.setattr(_spearman_batch, "ids_match", counting_ids_match)
    for n_frames in (6, 12):
        frames = _frames(n_frames=n_frames)
        requests = _requests(frames, True) + _requests(frames, True)[:3]  # duplicates are computed once
        sizes.clear()
        tti = call_counter(monkeypatch, _spearman_batch, "compute_tti")
        infl = call_counter(monkeypatch, _spearman_batch, "_influence_rows")
        build = call_counter(monkeypatch, _index._SpearmanFrameIndex, "from_groups")
        _dispatch._batch_from_groups(group_rows(frames, _KEYS), requests, method="spearman")
        assert tti["n"] == 1
        assert infl["n"] == 1  # one elementwise call per chunk, whatever the frame count (decompose included)
        assert build["n"] == 1  # the frames are extracted once, into one index
        index = _index._SpearmanFrameIndex.from_groups(group_rows(frames, _KEYS))
        assert index.n_frames == frames[list(_KEYS)].drop_duplicates().shape[0]  # one entry per distinct frame
        # Owner ruling 2026-11-18 (TBC-PERF-03): one ids_match per (index part, team), over that team's requested
        # rows, when the part's team ids take one path for every frame. Each requested row is read once per team.
        assert len(sizes) == 2  # one index part, two teams
        assert sum(sizes) == 2 * sum(index.n_rows(f) for f in range(index.n_frames))


def _content_frames(content: str) -> pd.DataFrame:
    """The production frames with the team id column in one storage form (players never have a missing team id)."""
    if content == "mixed":
        return _frames(mixed_ids=True)
    frames = _frames()
    team = frames["team_id"].astype(object)
    if content == "int64":
        return frames.assign(team_id=team.astype("Int64"))
    if content == "str_object":
        return frames.assign(team_id=team.map({5: "5", 6: "6"}).astype(object))
    return frames  # "category": the ADR-106 storage form


@pytest.mark.parametrize("content", ["category", "int64", "str_object", "mixed"])
def test_ids_match_once_per_part_and_team_unless_mixed(backend, monkeypatch, content):
    """Owner ruling 2026-11-18 (TBC-PERF-03, the spec W2.4 change): ``ids_match`` runs once per (index part, team)
    over that team's requested rows when its path cannot depend on the frame it sees: a non-object column, or an
    object column that holds only strings. An object column with non-strings (``mixed``) keeps one call per request.
    Every surface stays bit-identical to the per-frame public call."""
    frames = _content_frames(content)
    requests = _requests(frames, True)
    if content == "mixed":  # an int team next to a string team, as in the mixed-content production frames
        keys = frames[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)
        requests = [(tuple(k), team, True) for k in keys for team in (5, "6")]
    old = _per_frame(frames, requests, SpearmanParams())  # before the spy: the reference's own calls are not counted
    sizes: list[int] = []
    real_ids_match = _spearman_batch.ids_match

    def counting_ids_match(series, value):
        sizes.append(len(series))
        return real_ids_match(series, value)

    monkeypatch.setattr(_spearman_batch, "ids_match", counting_ids_match)
    new = _dispatch._batch_from_groups(group_rows(frames, _KEYS), requests, method="spearman", params=SpearmanParams())
    assert_surfaces_equal(new, old)
    n_teams = len(pd.unique(frames["team_id"].dropna()))
    assert len(sizes) == (len(set(requests)) if content == "mixed" else n_teams)
