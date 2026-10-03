"""Native-DAS corpus parity + performance driver (owner-run, DGX; ADR-107/108, spec 7.2).

Runs FOUR legs over the owner-tier pining corpus and writes an AGGREGATES-ONLY artifact
(``docs/research/das_native_parity/metrics.json``) -- never owner-tier rows (ADR-038):

* ``reference``  -- ``accessible-space`` 2.0.15, per ``(game, period)``, on float64 inputs, with the
  native direction supplied through ``attacking_direction_col`` (the golden-generator recipe). Imported
  LAZILY (the ``das-reference`` dev extra); never installed in CI. Stubbed by the golden reference
  outputs in the driver's local reduce-path test, so the full map+reduce runs offline.
* ``native numpy (reference quadrature)`` -- the parity target versus the library (delta ~ 1e-12).
* ``native numba (reference quadrature)`` -- engine-vs-engine (delta ~ 1e-10; ADR-076 precedent).
* ``native production (periodic quadrature)`` -- the shipped engine; its difference from the reference
  leg is the quadrature shift the ADR/CHANGELOG quote (ADR-108).

The corpus map is ``for_each`` (ADR-052: per-match shards, resumable, resume-before-load over
``list_match_refs``, conserving); the parity percentiles are CORPUS statistics computed in the REDUCE
over ALL shards. The clean-tree guard runs FIRST (ADR-037); the input contract declares which symbols
the numbers depend on (ADR-056).

The commit-2 artifact GATE (``tests/tracking/test_das_parity_artifact.py``) checks the spec 4.2 / 7.2
bounds against the owner run; this driver's own test pins the reduce PATH + schema + conservation.

Usage (owner):
    python scripts/validate_das_native_parity.py --out docs/research/das_native_parity
    python scripts/validate_das_native_parity.py --list-matches
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import pandas as pd

from scripts._das_reference_leg import _check_reference_env, _id_key
from scripts._input_contract import declare_inputs

# Corpus seam (owner-ratified reuse; ADR-052 D14). ``pining_source`` lists refs (resume-before-load
# over ``list_match_refs``) and loads one full match per item; exposed at module scope so a test can
# inject a fake corpus through it (tests/scripts/_fake_corpus.install_fake_corpus patches this name).
from scripts._loader_pining import pining_source, resolve_cache_dir
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._das_pack import PackedFrames, Reason
from silly_kicks.tracking._das_params import DAS_PARAMS
from silly_kicks.tracking._gk_resolve import resolve_defended_goals

_FRAME_KEYS = ("game_id", "period_id", "frame_id")
_X_OFFSET = 52.5

#: The velocity-bearing tracking providers scored by default (spec 7.2). ``statsbomb`` is NOT here: its
#: SB360 freeze-frames are velocity-less (ADR-063) and structurally unscoreable, recorded in the
#: population block. The owner overrides with ``--providers`` for a subset or to add statsbomb.
_DEFAULT_PROVIDERS = ("skillcorner", "gradientsports", "idsse")

#: The reference quadrature profile (parity-only; the shipped default is periodic, ADR-108).
_REFERENCE_PARAMS = dataclasses.replace(DAS_PARAMS, quadrature="reference")

#: The direction column the driver writes onto the frames before every leg, so all four legs score the
#: SAME per-frame direction (spec 7.2 "same direction"). Never a caller-facing name.
_DIR_COL = "_das_parity_dir"

#: numba thread counts profiled for the ms/frame table (spec 7.2). Only the ones <= os.cpu_count() run.
_THREAD_SWEEP = (1, 2, 4, 8, 16, 20)

#: das-native spec 4.1 golden-fixture bounds, applied PER ROW on the corpus (np.allclose, rtol = atol).
_GOLDEN_TOL_NUMPY = 1e-12
_GOLDEN_TOL_NUMBA = 1e-10

#: Pack reason -> shard column, so the reduce reports each divergence / degrade class (spec 7.2).
_REASON_COLS = {
    0: "reason_ok",
    1: "reason_no_possession",
    2: "reason_no_ball",
    3: "reason_no_players",
    4: "reason_ball_nan",
    5: "reason_poss_team_absent",
    6: "reason_direction_unresolved",
}

_SHARD_SCHEMA_VERSION = "das-native-parity-4"
_EMITTED_SHARD_COLUMNS = [
    "grain",  # "team" (per scored frame) | "player" (per player per frame) | "match" (per-match scalars)
    "provider",
    "game_id",
    "period_id",
    "frame_id",
    "player_id",
    # --- team/player rows: reference (accessible-space) vs native numpy, both in reference quadrature ---
    "abs_das",
    "rel_das",
    "abs_as",
    "rel_as",
    "finite_ref",
    "finite_native",
    "quad_shift_das",  # native periodic - native reference (the ADR-108 shift)
    "numba_minus_numpy_das",  # native numba - native numpy (reference quad); NaN when numba absent
    "numba_minus_numpy_as",  # the same for AS (team + player rows): all four numba cells are measured
    # The frame's Reason code, so the reduce can grade / finite-mask over reason==OK rows only (a non-OK
    # frame yields native NaN vs a fictional reference value). Present on team/player rows; OK on match
    # rows. The per-class NaN-degrade accounting is the match rows' reason_counts (D-BALLNAN, ...).
    "reason",
    # --- match rows: per-match scalars (NaN on team/player rows) ---
    "n_scored_frames",
    "ms_frame_ref",
    "ms_frame_numpy",
    "ms_frame_numba",
    "ms_frame_periodic",
    "n_dkey_frames",  # frames whose (game, frame_id) recurs in another period (the D-KEY figure)
    "n_dir_compared",  # frames compared against the library's OWN direction inference
    "n_dir_disagree",  # ... of which the team DAS disagrees
    *(_REASON_COLS[k] for k in sorted(_REASON_COLS)),
]


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056)."""
    return declare_inputs(
        driver="validate_das_native_parity",
        params={
            "das_params": dataclasses.asdict(DAS_PARAMS),
            "reference_quadrature": _REFERENCE_PARAMS.quadrature,
            "reference_env_pins": {"accessible-space": "2.0.15", "pandas": "<3"},
            "thread_sweep": list(_THREAD_SWEEP),
            "schema": _SHARD_SCHEMA_VERSION,
            "golden_tolerances": {"numpy": _GOLDEN_TOL_NUMPY, "numba": _GOLDEN_TOL_NUMBA},
        },
        extractors=["silly_kicks.tracking._das_pack", "silly_kicks.tracking._das_engine"],
        models=["silly_kicks.tracking._das_params"],
    )


# --------------------------------------------------------------------------------------------------
# Legs.
# --------------------------------------------------------------------------------------------------


def _direction_column(frames: pd.DataFrame, *, direction_col: str | None) -> pd.DataFrame:
    """Return ``frames`` carrying :data:`_DIR_COL` -- the per-frame +1/-1/NaN direction all legs share.

    Production: build the ``GoalMap`` from the FULL frames (ADR-055, ``resolve_defended_goals``) and
    derive the sign from the possession team's attacked goal. The test injects golden frames that
    already carry a ``dir`` column and passes ``direction_col="dir"`` (those frames have no keeper rows,
    so the GoalMap path is exercised by ``test_das_pack`` / ``test_das_divergences`` D-DIR instead).
    """
    out = frames.copy()
    if direction_col is not None:
        out[_DIR_COL] = pd.to_numeric(out[direction_col], errors="coerce").astype("float64")
        return out

    gm = resolve_defended_goals(frames)
    poss_by_frame = frames.groupby(list(_FRAME_KEYS), observed=True)["team_in_possession"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    )
    dir_by_key: dict[tuple, float] = {}
    for key, poss in poss_by_frame.items():
        k = cast(tuple, key)  # a 3-key groupby always yields a 3-tuple index
        if poss is None or pd.isna(poss):
            dir_by_key[k] = np.nan
            continue
        attacked = gm.attacked_goal(k[0], k[1], poss, allow_guess=True)
        dir_by_key[k] = np.nan if attacked is None else (1.0 if attacked >= _X_OFFSET else -1.0)
    keys = list(zip(*(out[c] for c in _FRAME_KEYS), strict=True))
    out[_DIR_COL] = [dir_by_key.get(tuple(k), np.nan) for k in keys]
    return out


def _native_player_key(v) -> int | str:
    """The native leg's player join key: the reference leg's own rule (``_id_key``) over ``canonical_id``.

    Integral ids (SkillCorner, GS) key as ints, as before; string ids (IDSSE ``DFL-OBJ-*``) key as the
    string. Player rows never carry an NA id.
    """
    return _id_key(str(canonical_id(v)))


def _run_native(
    frames: pd.DataFrame, params, *, engine: Literal["auto", "numpy", "numba"]
) -> tuple[pd.DataFrame, pd.DataFrame, PackedFrames]:
    """Pack ``frames`` (direction from :data:`_DIR_COL`), run ``compute_das``, return tidy team/player
    frames sorted by key plus the ``PackedFrames`` (the reduce reads its per-frame reason + defender
    geometry to tag divergences) -- the driver's own copy of the test helper (scripts must not import
    tests)."""
    from silly_kicks.tracking._das_engine import compute_das
    from silly_kicks.tracking._das_pack import pack_frames

    packed = pack_frames(frames, attacking_direction_col=_DIR_COL)
    res = compute_das(packed, params, engine=engine)
    keys = packed.keys
    team = (
        pd.DataFrame(
            {
                "game_id": keys["game_id"].to_numpy(),
                "period_id": keys["period_id"].to_numpy(),
                "frame_id": keys["frame_id"].to_numpy(),
                "team_as": res.team_as,
                "team_das": res.team_das,
            }
        )
        .sort_values(list(_FRAME_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    counts = np.diff(packed.offsets)
    rep = keys.loc[keys.index.repeat(counts)].reset_index(drop=True)
    pids = frames["player_id"].to_numpy()[packed.p_input_pos]
    player = (
        pd.DataFrame(
            {
                "game_id": rep["game_id"].to_numpy(),
                "period_id": rep["period_id"].to_numpy(),
                "frame_id": rep["frame_id"].to_numpy(),
                "player_id": [_native_player_key(v) for v in pids],
                "player_as": res.player_as,
                "player_das": res.player_das,
            }
        )
        .sort_values([*_FRAME_KEYS, "player_id"], kind="stable")
        .reset_index(drop=True)
    )
    return team, player, packed


def _frame_flags(packed: PackedFrames) -> pd.DataFrame:
    """One row per scored frame: ``(game_id, period_id, frame_id, reason)`` for the reduce.

    Keys carry the native dtypes (from ``packed.keys``) so the left-merge onto the parity rows aligns.
    The ``reason`` lets the reduce grade / finite-mask over the scoreable (``reason == OK``) rows only.
    """
    keys = packed.keys.reset_index(drop=True)
    return pd.DataFrame(
        {
            "game_id": keys["game_id"].to_numpy(),
            "period_id": keys["period_id"].to_numpy(),
            "frame_id": keys["frame_id"].to_numpy(),
            "reason": packed.reason.astype(np.uint8),
        }
    )


# The reference DAS leg runs ``accessible-space`` 2.0.15 in a PINNED PANDAS-2 SUBPROCESS: the library is
# silently broken under the driver's pandas 3 (Copy-on-Write makes its internal ``PLAYER_POS`` read-only,
# so its offside step raises a ``ValueError`` it catches and skips -- see scripts/_das_reference_leg.py
# and ADR-107/108). The recipe (``_REFERENCE_COMMON`` / ``_reference_lib_frames``) and the fail-loud env
# guard live in that sk-free module; the gate that pins the recipe to the golden generator's ``_COMMON``
# is tests/scripts/test_das_reference_leg.py.
_REFERENCE_MODULE = Path(__file__).with_name("_das_reference_leg.py")
_PROVISION_HINT = (
    "python3.12 -m venv ~/das-parity/py2ref && "
    "~/das-parity/py2ref/bin/pip install 'accessible-space==2.0.15' 'pandas<3'"
)
_PROBE_CODE = (
    "import importlib.metadata as m, pandas, numpy, platform, sys;"
    "sys.stdout.write('\\n'.join([pandas.__version__, numpy.__version__, "
    "m.version('accessible-space'), platform.python_version()]) + '\\n')"
)


def _n_dkey_frames(frames: pd.DataFrame) -> int:
    """Distinct (game, period, frame) keys whose (game, frame_id) recurs in ANOTHER period of the same game:
    the frames the old accessible-space keying (frame_id alone) conflated -- the D-KEY production figure."""
    keys = frames[list(_FRAME_KEYS)].drop_duplicates()
    per = keys.groupby(["game_id", "frame_id"])["period_id"].nunique()
    collide = per[per > 1].index
    return int(keys.set_index(["game_id", "frame_id"]).index.isin(collide).sum())


def _resolve_reference_python(arg: str | None) -> str:
    """The pandas-2 + accessible-space==2.0.15 interpreter for the reference leg (a DOCUMENTED
    PREREQUISITE; never auto-provisioned inside a clean-tree-gated run). Fail loud if absent."""
    p = arg or os.environ.get("SK_DAS_REFERENCE_PYTHON")
    if not p or not Path(p).is_file():
        raise SystemExit(
            "reference python not found; pass --reference-python or set SK_DAS_REFERENCE_PYTHON to a "
            f"pandas<3 + accessible-space==2.0.15 interpreter. Provision it with:\n    {_PROVISION_HINT}"
        )
    return p


def _probe_reference_env(reference_python: str) -> dict[str, str]:
    """Invoke the reference interpreter ONCE before the corpus pass; return its version strings (stamped
    into the artifact) and raise via ``_check_reference_env`` on pandas>=3 or a wrong accessible-space."""
    out = subprocess.run(  # noqa: S603 -- resolved documented-prerequisite interpreter + fixed probe code
        [reference_python, "-c", _PROBE_CODE], capture_output=True, text=True, check=True
    )
    pv, nv, av, pyv = [s for s in out.stdout.splitlines() if s][:4]
    _check_reference_env(pv, av)
    return {"pandas": pv, "numpy": nv, "accessible_space": av, "python": pyv}


def _reference_leg_subprocess(
    frames: pd.DataFrame, *, reference_python: str, repeat: int = 1, infer_direction: bool = False
) -> dict[str, np.ndarray]:
    """Marshal one match's scored frames to a temp parquet OUTSIDE the repo tree (an in-tree write would
    trip the clean-tree guard, ADR-037), run the reference module under ``reference_python``, and read
    team/player AS+DAS back. Raise on a non-zero exit or missing output -- never a silently-empty
    reference leg that would read as a vacuous parity pass."""
    d = Path(tempfile.mkdtemp(prefix="das_ref_"))
    try:
        in_pq = d / "in.parquet"
        frames.to_parquet(in_pq)
        subprocess.run(  # noqa: S603 -- resolved documented-prerequisite interpreter + our own module path
            [
                reference_python,
                str(_REFERENCE_MODULE),
                str(in_pq),
                str(d),
                str(repeat),
                "1" if infer_direction else "0",
            ],
            check=True,
        )
        team = pd.read_parquet(d / "team.parquet")
        player = pd.read_parquet(d / "player.parquet")
        return {
            "team_keys": team[list(_FRAME_KEYS)].to_numpy(),
            "team_as": team["as"].to_numpy(dtype=float),
            "team_das": team["das"].to_numpy(dtype=float),
            "player_keys": player[[*_FRAME_KEYS, "player_id"]].to_numpy(),
            "player_as": player["as"].to_numpy(dtype=float),
            "player_das": player["das"].to_numpy(dtype=float),
            # the reference library's in-process compute time (a 0-d array keeps the annotation true)
            "compute_s": np.asarray(
                json.loads((d / "timing.json").read_text(encoding="utf-8"))["compute_s"], dtype=float
            ),
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _scored_frames(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The production scoring set: the frames each action LINKS to (spec 7.2).

    Real corpus actions carry ``time_seconds`` (not ``frame_id``), so the link is time-based
    (``link_actions_to_frames``, ADR-004) and the scored set is the union of linked ``(period_id,
    frame_id)`` frames -- NOT every frame in the match (a full match is ~1e6 frames; scoring all of
    them is both wrong per spec and ruinously expensive). When ``actions`` already carries a
    ``frame_id`` (the golden/test fixture, pre-linked) it is used directly.
    """
    if actions is None:
        return frames.reset_index(drop=True)
    if "frame_id" in getattr(actions, "columns", ()):
        linked_ids = {int(v) for v in pd.to_numeric(actions["frame_id"], errors="coerce").dropna()}
        return (
            frames[frames["frame_id"].isin(linked_ids)].reset_index(drop=True)
            if linked_ids
            else frames.reset_index(drop=True)
        )

    from silly_kicks.tracking import link_actions_to_frames

    pointers, _report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    linked = pointers.merge(actions[["action_id", "period_id"]], on="action_id", how="left")
    linked = linked.loc[linked["frame_id"].notna(), ["period_id", "frame_id"]].drop_duplicates()
    if linked.empty:
        return frames.iloc[0:0].reset_index(drop=True)
    linked["period_id"] = linked["period_id"].astype(frames["period_id"].dtype)
    linked["frame_id"] = linked["frame_id"].astype(frames["frame_id"].dtype)
    return frames.merge(linked, on=["period_id", "frame_id"], how="inner").reset_index(drop=True)


def _prepare_possession(frames: pd.DataFrame) -> pd.DataFrame:
    """Derive ``team_in_possession`` (+ ``ball_carrier_player_id``) if the frames lack it -- the DAS
    caller prerequisite every consumer runs (ADR-004). Raw pining tracking carries velocity + goalkeeper
    flags but NOT possession, so the driver derives it here; the golden/test fixture already has it and
    is passed through unchanged."""
    if "team_in_possession" in frames.columns:
        return frames
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    return derive_team_in_possession(frames, infer_ball_carrier(frames))


def _abs_rel(ref: np.ndarray, native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise absolute and relative abs-diff (relative to |ref|, NaN when ref==0)."""
    absd = np.abs(ref - native)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(ref) > 0, absd / np.abs(ref), np.nan)
    return absd, rel


_LegT = TypeVar("_LegT")


def _time_leg(fn: Callable[[], _LegT], *, repeat: int = 1) -> tuple[_LegT, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)`` (best wall time over ``repeat`` runs)."""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    for _ in range(max(0, repeat - 1)):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _measure_match(item, *, reference_leg, inferred_leg=None, direction_col: str | None = None) -> pd.DataFrame:
    """One match -> a long shard: one ``team`` row per scored frame, one ``player`` row per player per
    scored frame, one ``match`` row of per-match scalars (timings, reason counts). EMPTY (columns
    present) when the match scores nothing -- "ran, produced nothing", never a crash that loses the pass.
    """
    provider, actions, frames = item[0], item[2], item[3]
    if frames is None or len(frames) == 0:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    scored = _scored_frames(actions, frames)
    if scored.empty:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    scored = _prepare_possession(scored)
    scored = _direction_column(scored, direction_col=direction_col)
    n_scored = int(scored[list(_FRAME_KEYS)].drop_duplicates().shape[0])
    # The match row's game_id is the frames' game (int), matching the team/player rows -- so the shard's
    # game_id column stays one dtype (a mixed int/str object column is unwritable to parquet) and the
    # reduce's distinct-game count is meaningful. match_id (item[1]) may be a different string token.
    match_game_id = scored["game_id"].iloc[0]

    # Legs, timed. Reference and native numpy are compared for parity; periodic gives the quad shift.
    (ref, t_ref) = _time_leg(lambda: reference_leg(scored))
    ((np_team, np_player, packed), t_np) = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numpy"))
    if _numba_available():
        ((nb_team, nb_player, _r), t_nb) = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numba"))
    else:
        nb_team = nb_player = None
        t_nb = np.nan
    ((per_team, _pp, _r2), t_per) = _time_leg(lambda: _run_native(scored, DAS_PARAMS, engine="numpy"))

    flags = _frame_flags(packed)  # per-frame reason + D-OFF, merged onto each parity row for the reduce
    rows: list[dict] = []
    rows += _team_rows(provider, ref, np_team, nb_team, per_team, flags)
    rows += _player_rows(provider, ref, np_player, nb_player, flags)

    reason_counts = pd.Series(packed.reason).map(_REASON_COLS).value_counts().to_dict()
    match_row = {
        "grain": "match",
        "provider": provider,
        "game_id": match_game_id,
        "n_scored_frames": n_scored,
        "ms_frame_ref": 1e3 * t_ref / n_scored if n_scored else np.nan,
        "ms_frame_numpy": 1e3 * t_np / n_scored if n_scored else np.nan,
        "ms_frame_numba": (1e3 * t_nb / n_scored if (n_scored and np.isfinite(t_nb)) else np.nan),
        "ms_frame_periodic": 1e3 * t_per / n_scored if n_scored else np.nan,
    }
    for col in _REASON_COLS.values():
        match_row[col] = int(reason_counts.get(col, 0))
    # Match rows carry no per-frame reason; set OK so the column stays a clean uint8 (a mixed uint8/NaN
    # object column is unwritable to parquet). The reduce reads reason on team/player rows only (it
    # filters by grain first).
    n_dkey = _n_dkey_frames(scored)
    n_cmp = n_dis = 0
    if inferred_leg is not None:
        inf = inferred_leg(scored)
        a = _join_on_keys(
            ref["team_keys"],
            {"r": ref["team_das"]},
            pd.DataFrame(
                {
                    "game_id": inf["team_keys"][:, 0],
                    "period_id": inf["team_keys"][:, 1],
                    "frame_id": inf["team_keys"][:, 2],
                    "i": inf["team_das"],
                }
            ),
            list(_FRAME_KEYS),
        )
        both = np.isfinite(a["r"].to_numpy(float)) & np.isfinite(a["i"].to_numpy(float))
        r, i = a["r"].to_numpy(float)[both], a["i"].to_numpy(float)[both]
        n_cmp, n_dis = int(both.sum()), int((np.abs(i - r) > 1e-9 * np.maximum(1.0, np.abs(r))).sum())
    match_row.update(
        {
            "n_dkey_frames": n_dkey,
            "n_dir_compared": n_cmp if inferred_leg is not None else np.nan,
            "n_dir_disagree": n_dis if inferred_leg is not None else np.nan,
        }
    )
    match_row["reason"] = int(Reason.OK)
    rows.append(match_row)

    shard = pd.DataFrame(rows).reindex(columns=_EMITTED_SHARD_COLUMNS)
    shard["reason"] = shard["reason"].astype("uint8")
    return shard


def _join_on_keys(ref_keys, ref_vals: dict, native: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Align the reference arrays (keyed by ``ref_keys``) to a native tidy frame on the shared keys.

    The reference keys may be an object array (production ``_reference_leg_subprocess``) while native keys are
    typed; cast each key column to the native dtype so the merge aligns (an int-vs-object key silently
    matches NOTHING and the parity would read as a vacuous zero-row join)."""
    ref_df = pd.DataFrame(ref_keys, columns=key_cols)
    for c in key_cols:
        ref_df[c] = ref_df[c].astype(native[c].dtype)
    for name, arr in ref_vals.items():
        ref_df[name] = arr
    return ref_df.merge(native, on=key_cols, how="inner", suffixes=("_ref", "_nat"))


def _team_rows(provider, ref, np_team, nb_team, per_team, flags) -> list[dict]:
    merged = _join_on_keys(
        ref["team_keys"],
        {"ref_as": ref["team_as"], "ref_das": ref["team_das"]},
        np_team,
        list(_FRAME_KEYS),
    )
    if nb_team is not None:
        merged = merged.merge(
            nb_team.rename(columns={"team_das": "nb_das", "team_as": "nb_as"})[[*_FRAME_KEYS, "nb_das", "nb_as"]],
            on=list(_FRAME_KEYS),
            how="left",
        )
    merged = merged.merge(
        per_team.rename(columns={"team_das": "per_das"})[[*_FRAME_KEYS, "per_das"]], on=list(_FRAME_KEYS), how="left"
    )
    merged = merged.merge(flags, on=list(_FRAME_KEYS), how="left")
    abs_das, rel_das = _abs_rel(merged["ref_das"].to_numpy(), merged["team_das"].to_numpy())
    abs_as, rel_as = _abs_rel(merged["ref_as"].to_numpy(), merged["team_as"].to_numpy())
    nb = merged["nb_das"].to_numpy() if "nb_das" in merged.columns else np.full(len(merged), np.nan)
    nb_as = merged["nb_as"].to_numpy() if "nb_as" in merged.columns else np.full(len(merged), np.nan)
    rows = []
    for i in range(len(merged)):
        rows.append(
            {
                "grain": "team",
                "provider": provider,
                "game_id": merged["game_id"].iloc[i],
                "period_id": merged["period_id"].iloc[i],
                "frame_id": merged["frame_id"].iloc[i],
                "abs_das": abs_das[i],
                "rel_das": rel_das[i],
                "abs_as": abs_as[i],
                "rel_as": rel_as[i],
                "finite_ref": bool(np.isfinite(merged["ref_das"].iloc[i])),
                "finite_native": bool(np.isfinite(merged["team_das"].iloc[i])),
                "quad_shift_das": float(merged["per_das"].iloc[i] - merged["ref_das"].iloc[i]),
                "numba_minus_numpy_das": float(nb[i] - merged["team_das"].iloc[i]) if np.isfinite(nb[i]) else np.nan,
                "numba_minus_numpy_as": (
                    float(nb_as[i] - merged["team_as"].iloc[i]) if np.isfinite(nb_as[i]) else np.nan
                ),
                "reason": int(merged["reason"].iloc[i]),
            }
        )
    return rows


def _player_rows(provider, ref, np_player, nb_player, flags) -> list[dict]:
    merged = _join_on_keys(
        ref["player_keys"],
        {"ref_as": ref["player_as"], "ref_das": ref["player_das"]},
        np_player,
        [*_FRAME_KEYS, "player_id"],
    )
    if nb_player is not None:
        nb_cols = nb_player.rename(columns={"player_das": "nb_das", "player_as": "nb_as"})
        merged = merged.merge(
            nb_cols[[*_FRAME_KEYS, "player_id", "nb_das", "nb_as"]], on=[*_FRAME_KEYS, "player_id"], how="left"
        )
    nb_d = merged["nb_das"].to_numpy(float) if "nb_das" in merged.columns else np.full(len(merged), np.nan)
    nb_a = merged["nb_as"].to_numpy(float) if "nb_as" in merged.columns else np.full(len(merged), np.nan)
    merged = merged.merge(flags, on=list(_FRAME_KEYS), how="left")
    abs_das, rel_das = _abs_rel(merged["ref_das"].to_numpy(), merged["player_das"].to_numpy())
    abs_as, rel_as = _abs_rel(merged["ref_as"].to_numpy(), merged["player_as"].to_numpy())
    rows = []
    for i in range(len(merged)):
        rows.append(
            {
                "grain": "player",
                "provider": provider,
                "game_id": merged["game_id"].iloc[i],
                "period_id": merged["period_id"].iloc[i],
                "frame_id": merged["frame_id"].iloc[i],
                "player_id": merged["player_id"].iloc[i],
                "abs_das": abs_das[i],
                "rel_das": rel_das[i],
                "abs_as": abs_as[i],
                "rel_as": rel_as[i],
                "finite_ref": bool(np.isfinite(merged["ref_das"].iloc[i])),
                "finite_native": bool(np.isfinite(merged["player_das"].iloc[i])),
                "numba_minus_numpy_das": (
                    float(nb_d[i] - merged["player_das"].iloc[i]) if np.isfinite(nb_d[i]) else np.nan
                ),
                "numba_minus_numpy_as": (
                    float(nb_a[i] - merged["player_as"].iloc[i]) if np.isfinite(nb_a[i]) else np.nan
                ),
                "reason": int(merged["reason"].iloc[i]),
            }
        )
    return rows


# --------------------------------------------------------------------------------------------------
# Reduce (corpus statistics over ALL shards, per provider).
# --------------------------------------------------------------------------------------------------


def _pct(values, q: float) -> float:
    """Percentile over finite values (accepts a Series or array); NaN on an empty/all-NaN input."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _grade_grain(df: pd.DataFrame) -> dict:
    """max / p99 / p50 of absolute + relative abs-diff for DAS and AS on a team- or player-grain slice."""
    return {
        out: {
            "abs": {
                "max": _pct(df[f"abs_{out}"], 100),
                "p99": _pct(df[f"abs_{out}"], 99),
                "p50": _pct(df[f"abs_{out}"], 50),
            },
            "rel": {
                "max": _pct(df[f"rel_{out}"], 100),
                "p99": _pct(df[f"rel_{out}"], 99),
                "p50": _pct(df[f"rel_{out}"], 50),
            },
        }
        for out in ("das", "as")
    }


def _n_outside_golden_bound(abs_d, rel_d, tol: float, extra=None) -> int:
    """Rows violating ``|native - ref| <= tol + tol * |ref|`` (np.allclose with rtol = atol = tol).

    ``|ref|`` is recovered from the shard pair (``rel = abs / |ref|``; NaN when ``ref == 0``, where the
    bound is ``atol`` alone). ``extra`` adds ``|numba - numpy|``, so ``abs + extra`` bounds
    ``|numba - ref|`` by the triangle inequality (a conservative count). Non-finite rows are
    finite-mask cases, counted by ``finite_mask_mismatches``.
    """
    a = np.asarray(abs_d, dtype=float)
    r = np.asarray(rel_d, dtype=float)
    dist = a + np.abs(np.asarray(extra, dtype=float)) if extra is not None else a
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_mag = np.where(np.isfinite(r) & (r > 0), a / r, 0.0)
    return int((np.isfinite(dist) & (dist > tol + tol * ref_mag)).sum())


def _clean_ok(g: pd.DataFrame) -> pd.DataFrame:
    """The scoreable rows of a team-/player-grain slice: ``reason == Reason.OK``.

    A non-OK frame yields native NaN vs a fictional reference value (a finite-mask mismatch), so the
    headline grade and the finite-mask check run over the OK rows only; the per-class NaN-degrade
    accounting is the match rows' ``reason_counts``. Cast through the NULLABLE ``Int64`` first: on a
    grain-mixed ``concat`` the ``reason`` column arrives object-typed with NaN, and a plain object
    ``.fillna`` downcast is deprecated in pandas 3.
    """
    if g.empty:
        return g
    reason = g["reason"].astype("Int64").fillna(int(Reason.OK)).astype("int64")
    return g[reason == int(Reason.OK)]


def reduce_parity(shards: list[pd.DataFrame]) -> dict:
    """Corpus parity/perf statistics per provider (spec 7.2). Empty shards -> ``{}``.

    The headline ``team``/``player`` grades and the finite-mask mismatch run over the scoreable
    (``reason == OK``) rows only (the mismatch must then read 0); the NaN-degrade classes (D-BALLNAN,
    D-POSSABSENT, ...) are accounted for by ``reason_counts``. There is no divergence-exclusion block: the
    reference leg respects offside (it runs faithfully in the pandas-2 subprocess) and keys frames
    collision-free, so the former D-OFF / D-KEY classes do not arise.
    """
    if not shards:
        return {}
    combined = pd.concat(shards, ignore_index=True)
    if combined.empty:
        return {}
    out: dict = {}
    for provider, sub in combined.groupby("provider"):
        team = sub[sub["grain"] == "team"].copy()
        player = sub[sub["grain"] == "player"].copy()
        match = sub[sub["grain"] == "match"]
        team_ok = _clean_ok(team)
        player_ok = _clean_ok(player)

        def _mask_mismatch(g: pd.DataFrame) -> int:
            return int((g["finite_ref"].astype("boolean") != g["finite_native"].astype("boolean")).sum())

        quad = team["quad_shift_das"].abs()
        nbmax = team["numba_minus_numpy_das"].abs()
        out[str(provider)] = {
            "team": _grade_grain(team_ok) if not team_ok.empty else {},
            "player": _grade_grain(player_ok) if not player_ok.empty else {},
            "finite_mask_mismatches": {"team": _mask_mismatch(team_ok), "player": _mask_mismatch(player_ok)},
            "quadrature_shift_das": {"median": _pct(quad, 50), "p90": _pct(quad, 90), "max": _pct(quad, 100)},
            "numba_vs_numpy_das_max_abs": _pct(nbmax, 100),
            "numba_vs_numpy_max_abs": {
                grain: {o: _pct(g[f"numba_minus_numpy_{o}"].abs(), 100) for o in ("das", "as")}
                for grain, g in (("team", team_ok), ("player", player_ok))
            },
            "n_outside_golden_bound": {
                eng: {
                    grain: {
                        o: _n_outside_golden_bound(
                            g[f"abs_{o}"],
                            g[f"rel_{o}"],
                            tol,
                            extra=(g[f"numba_minus_numpy_{o}"] if eng == "numba" else None),
                        )
                        for o in ("das", "as")
                    }
                    for grain, g in (("team", team_ok), ("player", player_ok))
                }
                for eng, tol in (("numpy", _GOLDEN_TOL_NUMPY), ("numba", _GOLDEN_TOL_NUMBA))
            },
            # CCC-PLAN-23: how many FINITE numba comparisons each cell actually made -- a NaN diff (empty or
            # misaligned numba merge) is "not compared", so a vacuous cell reads 0 here, never "clean".
            "numba_compared": {
                grain: {o: int(np.isfinite(g[f"numba_minus_numpy_{o}"].to_numpy(float)).sum()) for o in ("das", "as")}
                for grain, g in (("team", team_ok), ("player", player_ok))
            },
            "finite_counts": {
                grain: {
                    "ref": int(g["finite_ref"].astype(bool).sum()),
                    "native": int(g["finite_native"].astype(bool).sum()),
                    "rows": len(g),
                }
                for grain, g in (("team", team), ("player", player))
            },
            "d_key_frames": int(match["n_dkey_frames"].fillna(0).sum()),
            "direction": {
                "n_compared": int(match["n_dir_compared"].fillna(0).sum()),
                "n_disagree": int(match["n_dir_disagree"].fillna(0).sum()),
            },
            "timings_ms_per_frame": {
                leg: _pct(match[f"ms_frame_{leg}"], 50) for leg in ("ref", "numpy", "numba", "periodic")
            },
            "reason_counts": {col: int(match[col].fillna(0).sum()) for col in _REASON_COLS.values()},
            "n_scored_frames": int(match["n_scored_frames"].fillna(0).sum()),
            "n_matches_scored": int(match["game_id"].nunique()),
        }
    return out


def _map_token(direction_col: str | None, commit: str) -> dict:
    """The map's shard-generation token inputs. The RUN COMMIT is one of them (combined-cycle spec 2):
    shards are attributable to the commit that built them even if a worker dies before its manifest."""
    return {
        "metric": "das_native_parity",
        "schema": _SHARD_SCHEMA_VERSION,
        "direction": direction_col or "goal_map",
        "commit": commit,
    }


def _map_generation(commit: str, direction_col: str | None = None) -> str:
    """The shard-generation directory name the map writes for ``commit`` (and the reduce expects)."""
    from scripts._driver import _token

    return _token(_map_token(direction_col, commit), None)


def run_corpus(
    refs,
    load,
    dest: Path,
    *,
    prov: dict,
    shard_root: Path | None = None,
    direction_col: str | None = None,
    reference_leg,
    reference_env: dict | None = None,
    shards_only: bool = False,
    worker_tag: str = "serial",
    inferred_leg=None,
) -> dict:
    """The map+reduce+write, factored out of ``main`` so the reduce PATH is testable offline.

    ``load(ref) -> (provider, match_id, actions, frames)``; ``reference_leg`` and ``direction_col`` are
    injected by the test (golden reference outputs + the golden ``dir`` column), so the full corpus
    pass runs with no ``accessible-space`` and no network. Production passes the pandas-2 subprocess leg
    (``_reference_leg_subprocess``) and the ``GoalMap`` direction. ``reference_env`` is the reference
    interpreter's version strings (from ``_probe_reference_env``); it rides each worker's manifest and is
    stamped into the reduced artifact for provenance.

    ``shards_only`` runs the MAP only: it writes the per-match shards plus THIS worker's
    ``manifest_<worker_tag>.json`` (``res.manifest()`` + ``reference_env``; otherwise in-memory only),
    then returns WITHOUT reducing. That is what lets N processes each take a ``--match-ids-json`` subset
    against one resumable ``shard_root`` and have the launcher call :func:`reduce_parity_artifact` exactly
    ONCE over the full population afterward -- N concurrent subset-scoped ``metrics.json`` writes would
    each race an artifact that is NOT the full-corpus one (DPL-PLAN-03). Every worker MUST pass a UNIQUE
    ``worker_tag`` or the manifests collide. The serial path writes its manifest too and reduces through
    the same single-sourced :func:`reduce_parity_artifact`.
    """
    from scripts._driver import for_each

    def _work(item):
        return _measure_match(item, reference_leg=reference_leg, inferred_leg=inferred_leg, direction_col=direction_col)

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        shard_root=shard_root if shard_root is not None else dest / "shards",
        token_inputs=_map_token(direction_col, prov["commit"]),
        label="match",
    )
    # Persist this worker's manifest so the reduce can aggregate exclusions it cannot see from shards
    # alone (SB360 structural exclusion, `.excluded.json` markers). res.manifest() is in-memory only.
    (res.shard_dir / f"manifest_{worker_tag}.json").write_text(
        json.dumps(
            {
                **res.manifest(),
                "reference_env": reference_env or {},
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
            },
            default=str,
        ),
        encoding="utf-8",
    )
    if shards_only:
        return {"shards_only": True, "shard_dir": str(res.shard_dir)}
    root = shard_root if shard_root is not None else dest / "shards"
    return reduce_parity_artifact(refs, root, dest, prov=prov, direction_col=direction_col)


def reduce_parity_artifact(refs, shard_root: Path, dest: Path, *, prov: dict, direction_col: str | None = None) -> dict:
    """Reduce the shards under ``shard_root`` into the single combined ``metrics.json`` (spec 7.2).

    Single-sources the reduce for both the serial ``run_corpus`` and the parallel launcher, which calls
    this ONCE after every ``shards_only`` worker finishes. Parity VALUES come from the shard parquet;
    the population's exclusion counters come from ``_partition.aggregate_manifests`` (summing the
    per-worker ``manifest_*.json``), never re-derived from shards -- shards cannot see a match excluded
    with no output row (DPL-PLAN-10). ``listed_per_provider`` comes from the FULL ``refs`` (as serial
    ``_population`` does), so a subset worker never under-reports the corpus. The emitted schema is
    byte-identical to the pre-split serial artifact.
    """
    from scripts._partition import aggregate_manifests

    shard_root = Path(shard_root)
    shard_files = sorted(shard_root.rglob("*.parquet"))
    if not shard_files:
        raise SystemExit(f"no parity shards under {shard_root} -- run the map (shards) first")
    gen_dirs = {p.parent for p in shard_files}
    if len(gen_dirs) != 1:
        raise SystemExit(
            f"expected ONE shard generation under {shard_root}, found {sorted(str(g) for g in gen_dirs)}; "
            "use a fresh --shard-root per corpus run"
        )
    gen_dir = next(iter(gen_dirs))
    from scripts._driver import exclusion_path, shard_path

    expected = _map_generation(prov["commit"], direction_col)
    if gen_dir.name != expected:
        # The token covers the commit AND every other map input (B r3 CCC-PLAN-26): name them all.
        raise SystemExit(
            f"the shard generation {gen_dir.name} does not match this reduce's token {expected} "
            f"(commit {prov['commit']}, direction_col {direction_col!r}); "
            "reduce at the build commit with the build's flags"
        )
    missing = [
        r.key for r in refs if not shard_path(gen_dir, r.key).is_file() and not exclusion_path(gen_dir, r.key).is_file()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(refs)} listed matches have no shard (first: {missing[:3]}); "
            "re-run that worker (it resumes)"
        )
    # reference_env (the pandas-2 reference interpreter's versions) is map-time provenance carried on the
    # per-worker manifests, so a --reduce-only invocation needs no reference interpreter (README hazard 4).
    # Collect the DISTINCT non-empty reference_env across workers. A mixed-reference-interpreter corpus
    # (workers run against different pandas/accessible-space versions) yields non-comparable parity
    # numbers, so refuse rather than silently stamp one (DRR-IMPL-01). The launcher passes one
    # --reference-python to all workers, so this fires only on operator error; the reduce is cheap to redo.
    seen: dict[str, dict] = {}
    for mp in sorted(gen_dir.glob("manifest_*.json")):
        env = json.loads(mp.read_text(encoding="utf-8")).get("reference_env")
        if env:
            seen[json.dumps(env, sort_keys=True)] = env
    if len(seen) > 1:
        raise SystemExit(
            "reference_env disagreement across workers -- a mixed-reference-interpreter corpus parity "
            f"artifact is invalid: {sorted(seen)}. Re-run the map with one --reference-python."
        )
    reference_env: dict = next(iter(seen.values())) if seen else {}
    shards = [pd.read_parquet(s) for s in shard_files]
    parity = reduce_parity(shards)
    scored_providers = {p: parity[p]["n_matches_scored"] for p in parity}
    # Sum the per-worker manifests, then rebuild the exact `res.manifest()` shape so the artifact's
    # top-level fields are unchanged from the serial version (Hyrum: metrics.json is a published seam).
    agg = aggregate_manifests(gen_dir, defaults=("n_attempted", "n_failed", "n_counters_unrecorded", "n_excluded"))
    # Defence in depth (CCC-PLAN-21): the commit-keyed generation already refuses a worker built at another
    # commit, so this fires only for a foreign manifest inside THIS generation (test_a_planted_foreign_manifest...).
    foreign = sorted(set(agg["commits_seen"]) - {prov["commit"]})
    if foreign:
        raise SystemExit(f"worker manifest(s) from another commit {foreign}")
    manifest = {
        "generation": gen_dir.name,
        "n_attempted": agg["n_attempted"],
        "n_failed": agg["n_failed"],
        "n_counters_unrecorded": agg["n_counters_unrecorded"],
        "n_excluded": agg["n_excluded"],
    }
    out = {
        "providers": parity,
        "population": _population(refs, scored_providers, manifest, accounted=len(refs)),
        **manifest,
        "run_commit": prov["commit"],
        "run_tree_dirty": bool(prov["dirty"] or agg["run_tree_dirty"]),
        "commit_consistent": agg["commit_consistent"],
        "commits_seen": agg["commits_seen"],
        "run_tree_state": prov.get("tree_state"),
        "run_platform": prov.get("platform"),
        "run_machine": prov.get("machine"),
        "reference_env": reference_env or {},
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _population(refs, scored_providers: dict, manifest: dict, *, accounted: int) -> dict:
    """Matches listed / scored / excluded per provider (spec 7.2). SB360 is structurally unscoreable
    (velocity-less freeze-frames, ADR-063): recorded, not silently dropped."""
    listed: dict[str, int] = {}
    for ref in refs:
        listed[ref.provider] = listed.get(ref.provider, 0) + 1
    return {
        "listed_per_provider": listed,
        # every listed key has a shard or an exclusion marker (else the reduce refused before this point)
        "accounted": accounted,
        "scored_per_provider": scored_providers,
        "excluded": {
            "attempted": manifest.get("n_attempted"),
            "failed": manifest.get("n_failed"),
            "excluded": manifest.get("n_excluded"),
            "note": (
                "SB360 freeze-frames are velocity-less (ADR-063) -> structurally unscoreable; excluded and recorded."
            ),
        },
    }


_PATH_TIMING_MODULE = Path(__file__).resolve().parent / "_das_path_timing.py"
#: The per-run memory ceiling (GiB) for the add_das / das_xfns path legs. The OLD path's das_xfns needed
#: 113.4 GiB on an IDSSE match and was OOM-killed inside a 110G cap on a GS match (combined-cycle Phase B);
#: a run that hits the ceiling is recorded as not fitting, never allowed to take the box down.
_PATH_MEMORY_LIMIT_GIB = 100.0
_COORD_COLS = ("x", "y", "z", "vx", "vy", "speed", "x_smoothed", "y_smoothed")
_FOREIGN_CPU_MAX = 0.05  # contention gate: other processes' CPU over the whole benchmark


def _sweep_counts() -> list[int]:
    """The `_THREAD_SWEEP` counts this box can run (numba refuses more threads than CPUs)."""
    return [k for k in _THREAD_SWEEP if k <= (os.cpu_count() or 1)]


def _benchmark_match_ids(sample_path: Path) -> dict[str, list[str]]:
    """The benchmark population IS the sample file whose SHA-256 the artifact records (list-matches shape)."""
    return _load_match_ids(json.loads(Path(sample_path).read_text(encoding="utf-8")))


def _old_path_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """The frames as the pre-F1b (4.127.0) path stored them: float64 coords, no category ids."""
    out = frames.copy()
    for c in _COORD_COLS:
        if c in out.columns:
            out[c] = out[c].astype("float64")
    for c in ("team_id", "player_id"):
        if c in out.columns and isinstance(out[c].dtype, pd.CategoricalDtype):
            integer = pd.api.types.is_integer_dtype(out[c].cat.categories.dtype)
            out[c] = out[c].astype(object).astype("Int64") if integer else out[c].astype(object)
    return out


def _path_subprocess(
    frames,
    actions,
    *,
    python: str,
    repeat: int,
    expect_native: bool,
    memory_limit_gib: float = _PATH_MEMORY_LIMIT_GIB,
) -> dict:
    """Time add_das / das_xfns in a pandas-2 interpreter; refuse unless it ran the expected engine.

    The timing module enforces ``memory_limit_gib`` on itself and, above it, returns what finished plus an
    ``over_memory`` record (a result, not a failure).
    """
    d = Path(tempfile.mkdtemp(prefix="das_path_"))
    try:
        frames.to_parquet(d / "frames.parquet")
        actions.to_parquet(d / "actions.parquet")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run(  # noqa: S603 -- resolved prerequisite interpreter + our own module path
            [python, str(_PATH_TIMING_MODULE), str(d), str(d), str(repeat), str(float(memory_limit_gib))],
            check=True,
            env=env,
        )
        timing = json.loads((d / "timing.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if bool(timing.get("native")) is not expect_native:
        which = "new" if expect_native else "old"
        raise SystemExit(
            f"the {which} path interpreter ran native={timing.get('native')} (silly-kicks {timing.get('silly_kicks')})"
        )
    return timing


def _keeper_counterfactual(scored: pd.DataFrame, shift_m: float = 1.0) -> tuple[pd.DataFrame, bool]:
    """The gkdv-shaped pair leg: every defending keeper moved +shift_m in x (kinematics only)."""
    from silly_kicks.id_compat import ids_differ

    cf = scored.copy()
    tip = cf["team_in_possession"]
    moved = (
        cf["is_goalkeeper"].fillna(False).astype(bool).to_numpy()
        & ~cf["is_ball"].astype(bool).to_numpy()
        & tip.notna().to_numpy()
        & ids_differ(cf["team_id"], tip).to_numpy()
    )
    if moved.any():
        cf.loc[moved, "x"] = (cf.loc[moved, "x"].astype("float64") + shift_m).astype(cf["x"].dtype)
    return cf, bool(moved.any())


def _bench_match(
    item,
    *,
    reference_python: str,
    old_path_python: str,
    new_path_python: str,
    repeat: int,
    memory_limit_gib: float = _PATH_MEMORY_LIMIT_GIB,
) -> dict:
    """One match's spec 4.2 legs, best-of-``repeat`` after a warm-up. No match id is recorded.

    A path leg that hits ``memory_limit_gib`` records what finished plus ``<new|old>_path_over_memory``
    (phase, limit, peak); its missing time drops that match from that speedup only.
    """
    from silly_kicks.tracking._das import get_individual_das, individual_das_paired
    from silly_kicks.tracking._das_engine import compute_das
    from silly_kicks.tracking._das_pack import pack_frames

    provider, _match_id, actions, frames = item
    scored = _direction_column(_prepare_possession(_scored_frames(actions, frames)), direction_col=None)
    n = int(scored[list(_FRAME_KEYS)].drop_duplicates().shape[0])
    row: dict = {"provider": provider, "n_scored_frames": n}
    if n == 0:
        return row
    ref = _reference_leg_subprocess(scored, reference_python=reference_python, repeat=repeat)
    row["ms_frame_ref"] = 1e3 * ref["compute_s"] / n
    _, t = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numpy"), repeat=repeat)
    row["ms_frame_numpy"] = 1e3 * t / n
    _run_native(scored, _REFERENCE_PARAMS, engine="numba")  # JIT warm-up, untimed
    _, t = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numba"), repeat=repeat)
    row["ms_frame_numba_serial"] = 1e3 * t / n
    packed = pack_frames(scored, attacking_direction_col=_DIR_COL)
    _, t = _time_leg(lambda: compute_das(packed, DAS_PARAMS, engine="numpy"), repeat=repeat)
    row["ms_frame_numpy_periodic"] = 1e3 * t / n
    threads = {}
    for k in _sweep_counts():
        compute_das(packed, DAS_PARAMS, engine="numba", n_threads=k)  # prange compiles separately
        _, t = _time_leg(lambda k=k: compute_das(packed, DAS_PARAMS, engine="numba", n_threads=k), repeat=repeat)
        threads[str(k)] = 1e3 * t / n
    row["numba_threads_ms_frame"] = threads
    # add_das refuses frames without team_in_possession and raw pining frames lack it: derive it (the caller
    # prerequisite, untimed) once, so both path legs time the same input.
    old_frames = _old_path_frames(_prepare_possession(frames))
    for which, python, native in (("new", new_path_python, True), ("old", old_path_python, False)):
        t = _path_subprocess(
            old_frames, actions, python=python, repeat=repeat, expect_native=native, memory_limit_gib=memory_limit_gib
        )
        for leg in ("add_das", "das_xfns"):
            if f"{leg}_s" in t:
                row[f"{leg}_{which}_s"] = t[f"{leg}_s"]
        if t.get("over_memory"):
            row[f"{which}_path_over_memory"] = {k: t.get(k) for k in ("phase", "limit_gib", "peak_gib")}
        row[f"pandas_{which}"] = t["pandas"]
    cf, any_moved = _keeper_counterfactual(scored)
    if any_moved:
        _, row["paired_s"] = _time_leg(
            lambda: individual_das_paired(scored, cf, attacking_direction_col=_DIR_COL), repeat=repeat
        )
        _, row["independent_s"] = _time_leg(
            lambda: (
                get_individual_das(scored, attacking_direction_col=_DIR_COL),
                get_individual_das(cf, attacking_direction_col=_DIR_COL),
            ),
            repeat=repeat,
        )
    return row


def summarize_benchmark(rows: list[dict]) -> dict:
    """The spec 4.2 figures (pure). ms/frame legs: ratio of medians; per-match legs: median ratio.

    A path-leg speedup uses the matches where BOTH paths finished that call; ``speedup_n`` says how many,
    and ``<old|new>_path_over_memory`` counts per provider the matches whose path leg hit the ceiling.
    """
    ok = [r for r in rows if r.get("n_scored_frames")]

    def med(key):
        return float(np.median([r[key] for r in ok]))

    def speedup(leg: str) -> tuple[float | None, int]:
        both = [r[f"{leg}_old_s"] / r[f"{leg}_new_s"] for r in ok if f"{leg}_old_s" in r and f"{leg}_new_s" in r]
        return (float(np.median(both)) if both else None), len(both)

    def over_memory(which: str) -> dict[str, int]:
        hits: dict[str, int] = {}
        for r in ok:
            if f"{which}_path_over_memory" in r:
                hits[str(r["provider"])] = hits.get(str(r["provider"]), 0) + 1
        return hits

    ref, npy, nb = med("ms_frame_ref"), med("ms_frame_numpy"), med("ms_frame_numba_serial")
    counts = sorted({k for r in ok for k in r["numba_threads_ms_frame"]}, key=int)  # those the box could run
    tk = {k: float(np.median([r["numba_threads_ms_frame"][k] for r in ok])) for k in counts}
    paired = [r["paired_s"] / r["independent_s"] for r in ok if "paired_s" in r]
    (add_sp, add_n), (xfn_sp, xfn_n) = speedup("add_das"), speedup("das_xfns")
    return {
        "ref_over_numba_serial": ref / nb,
        "ref_over_numpy": ref / npy,
        "prange_efficiency": {k: tk["1"] / (int(k) * v) for k, v in tk.items()},
        "add_das_speedup": add_sp,
        "das_xfns_speedup": xfn_sp,
        "speedup_n": {"add_das": add_n, "das_xfns": xfn_n},
        "old_path_over_memory": over_memory("old"),
        "new_path_over_memory": over_memory("new"),
        "paired_over_independent": float(np.median(paired)) if paired else None,
        "ms_frame_median": {"ref": ref, "numpy": npy, "numba_serial": nb},
        "seconds_per_frame": {"numba_serial": tk["1"] / 1e3, "numpy": med("ms_frame_numpy_periodic") / 1e3},
        "n_matches": len(ok),
    }


def _foreign_cpu_fraction(*, busy0: float, busy1: float, own0: float, own1: float, elapsed: float, ncpu: int) -> float:
    """Other processes' share of the machine over the run: (machine busy - own) / (elapsed * ncpu)."""
    return max(0.0, (busy1 - busy0) - (own1 - own0)) / (elapsed * ncpu)


def _cpu_snapshot() -> tuple[float, float] | None:
    """(machine busy cpu-seconds from /proc/stat, own+children cpu-seconds); None off Linux."""
    if sys.platform == "win32":
        return None
    import resource

    fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
    vals = [int(v) for v in fields]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    busy = (sum(vals) - idle) / os.sysconf("SC_CLK_TCK")
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return busy, me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def _peak_rss_bytes() -> dict | None:
    if sys.platform == "win32":
        return None
    import resource

    return {
        "self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }


def _loadavg() -> float | None:
    return None if sys.platform == "win32" else os.getloadavg()[0]


def run_benchmark(
    refs,
    load,
    dest: Path,
    *,
    prov: dict,
    reference_python: str,
    old_path_python: str,
    new_path_python: str,
    repeat: int,
    sample_sha256: str,
    memory_limit_gib: float = _PATH_MEMORY_LIMIT_GIB,
) -> dict:
    """The benchmark artifact (spec 12 D1): run ALONE after every other process. Provider-only rows."""
    snap0, t0, load0 = _cpu_snapshot(), time.perf_counter(), _loadavg()
    rows = [
        _bench_match(
            load(ref),
            reference_python=reference_python,
            old_path_python=old_path_python,
            new_path_python=new_path_python,
            repeat=repeat,
            memory_limit_gib=memory_limit_gib,
        )
        for ref in refs
    ]
    snap1, elapsed = _cpu_snapshot(), time.perf_counter() - t0
    foreign = (
        _foreign_cpu_fraction(
            busy0=snap0[0], busy1=snap1[0], own0=snap0[1], own1=snap1[1], elapsed=elapsed, ncpu=os.cpu_count() or 1
        )
        if snap0 and snap1
        else None
    )
    out = {
        "summary": summarize_benchmark(rows),
        "path_memory_limit_gib": float(memory_limit_gib),
        "per_match": rows,
        "sample_sha256": sample_sha256,
        "repeat": repeat,
        "thread_sweep": _sweep_counts(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "contention": {
            "foreign_cpu_fraction": foreign,
            "max": _FOREIGN_CPU_MAX,
            "loadavg_1m": {"before": load0, "after": _loadavg()},
        },
        "reference_env": _probe_reference_env(reference_python),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "run_platform": prov.get("platform"),
        "run_machine": prov.get("machine"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "performance.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------------


def _load_match_ids(spec: list[dict]) -> dict[str, list[str]]:
    """Group a ``--list-matches``-shaped list (``[{"provider", "match_id"}, ...]``) into the
    ``pining_source(match_ids=)`` mapping.

    Selection then flows through ``_wanted_for_provider`` -- the SAME rule ``for_each`` resumes on -- so
    an owner can split ``--list-matches`` into N subsets and run N processes against ONE resumable shard
    root (each does its own matches; resume-before-load skips the rest), turning the ~14 CPU-hour
    sequential reference leg into a parallel pass.
    """
    out: dict[str, list[str]] = {}
    for entry in spec:
        out.setdefault(str(entry["provider"]), []).append(str(entry["match_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--shard-root", default=None, help="shard root (default <out>/shards)")
    ap.add_argument("--providers", nargs="*", default=None, help="providers to walk (default: all velocity-bearing)")
    ap.add_argument("--token", default=None, help="pining token (else resolved from the environment)")
    ap.add_argument(
        "--reference-python",
        default=None,
        help="pandas<3 + accessible-space==2.0.15 interpreter for the reference leg (else "
        "$SK_DAS_REFERENCE_PYTHON); a documented prerequisite -- see the module docstring",
    )
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help="restrict the run to the matches in this JSON file (the --list-matches shape: "
        "[{provider, match_id}, ...]); split it across processes to parallelise the reference leg",
    )
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    ap.add_argument(
        "--shards-only",
        action="store_true",
        help="MAP only: write shards + this worker's manifest, skip the reduce (parallel worker). "
        "Pair with a UNIQUE --worker-tag and a shared --shard-root; the launcher reduces once at the end.",
    )
    ap.add_argument(
        "--reduce-only",
        action="store_true",
        help="REDUCE only: assemble metrics.json from the shards already under --shard-root (or "
        "<out>/shards), then exit. Run after all --shards-only workers finish.",
    )
    ap.add_argument(
        "--worker-tag",
        default="serial",
        help="unique tag for this worker's manifest_<tag>.json under --shards-only (default 'serial').",
    )
    ap.add_argument(
        "--benchmark", action="store_true", help="write performance.json over --benchmark-sample-json (run ALONE)"
    )
    ap.add_argument("--benchmark-sample-json", default=None, help="the fixed sample ([{provider, match_id}, ...])")
    ap.add_argument(
        "--old-path-python",
        default=None,
        help="silly-kicks 4.127.0 + accessible-space 2.0.15 + pandas<3 (else $SK_DAS_OLDPATH_PYTHON)",
    )
    ap.add_argument(
        "--new-path-python", default=None, help="this checkout installed with pandas<3 (else $SK_DAS_NEWPATH_PYTHON)"
    )
    ap.add_argument("--repeat", type=int, default=3, help="best-of-N timing repeats (benchmark only)")
    ap.add_argument(
        "--path-memory-limit-gib",
        type=float,
        default=_PATH_MEMORY_LIMIT_GIB,
        help="memory ceiling per add_das/das_xfns path run (benchmark only); a run above it is recorded as "
        "not fitting (default %(default)s)",
    )
    ap.add_argument(
        "--print-generation",
        action="store_true",
        help="print the shard generation a map at THIS commit writes (the launcher's --das-generation) and exit",
    )
    args = ap.parse_args()

    if args.print_generation:
        from scripts._provenance import git_provenance as _git_provenance

        print(_map_generation(_git_provenance()["commit"]))
        return

    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # Clean-tree guard FIRST, before any corpus work (ADR-037). --list-matches writes no artifact.
    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "platform": "n/a", "machine": "n/a"}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    providers: list[str] = list(args.providers) if args.providers else list(_DEFAULT_PROVIDERS)

    if args.list_matches:
        refs, _ = pining_source(providers, token=args.token)
        print(json.dumps([{"provider": r.provider, "match_id": r.match_id} for r in refs], indent=2))
        return

    dest = Path(args.out)
    cache_dir = resolve_cache_dir(args.cache_dir)
    match_ids = (
        _load_match_ids(json.loads(Path(args.match_ids_json).read_text("utf-8"))) if args.match_ids_json else None
    )
    if args.benchmark:
        if args.match_ids_json:
            raise SystemExit(
                "--benchmark takes its population from --benchmark-sample-json only; drop --match-ids-json"
            )
        if not args.benchmark_sample_json:
            raise SystemExit("--benchmark needs --benchmark-sample-json (a fixed, seeded sample)")
        match_ids = _benchmark_match_ids(Path(args.benchmark_sample_json))
    from scripts._partition import providers_for_slice

    refs, base_load = pining_source(
        providers_for_slice(providers, match_ids),
        token=args.token,
        match_ids=match_ids,
        max_per_provider=args.max_matches,
        cache_dir=cache_dir,
    )
    if match_ids is not None and not refs:
        raise SystemExit(f"--match-ids-json {args.match_ids_json} selected no matches from providers {providers}.")

    def _load(ref):
        lm = base_load(ref)
        return (lm.provider, lm.match_id, lm.actions, lm.frames)

    if args.benchmark:
        if not args.benchmark_sample_json:
            raise SystemExit("--benchmark needs --benchmark-sample-json (a fixed, seeded sample)")
        old_python = args.old_path_python or os.environ.get("SK_DAS_OLDPATH_PYTHON")
        new_python = args.new_path_python or os.environ.get("SK_DAS_NEWPATH_PYTHON")
        if not old_python or not new_python:
            raise SystemExit("--benchmark needs both pandas-2 path interpreters (old 4.127.0, new = this checkout)")
        out = run_benchmark(
            refs,
            _load,
            dest,
            prov=prov,
            reference_python=_resolve_reference_python(args.reference_python),
            old_path_python=old_python,
            new_path_python=new_python,
            repeat=args.repeat,
            sample_sha256=hashlib.sha256(Path(args.benchmark_sample_json).read_bytes()).hexdigest(),
            memory_limit_gib=args.path_memory_limit_gib,
        )
        print(json.dumps(out["summary"], indent=2, default=str))
        return

    # Reduce-only: assemble the combined artifact from shards already on disk (after the workers).
    # Needs NO reference interpreter -- reference_env rides the manifests (README hazard 4).
    if args.reduce_only:
        root = Path(args.shard_root) if args.shard_root else dest / "shards"
        out = reduce_parity_artifact(refs, root, dest, prov=prov, direction_col=None)
        print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))
        return

    # Map (shards-only or full): the pandas-2 subprocess reference leg + GoalMap direction (ADR-055).
    # Resolve + probe the reference interpreter BEFORE any corpus work (fail-loud on a bad/absent env).
    reference_python = _resolve_reference_python(args.reference_python)
    ref_env = _probe_reference_env(reference_python)
    out = run_corpus(
        refs,
        _load,
        dest,
        prov=prov,
        shard_root=Path(args.shard_root) if args.shard_root else None,
        reference_leg=functools.partial(_reference_leg_subprocess, reference_python=reference_python),
        # das-native 7.2: the library's OWN direction inference vs the GoalMap direction (team DAS)
        inferred_leg=functools.partial(
            _reference_leg_subprocess, reference_python=reference_python, infer_direction=True
        ),
        reference_env=ref_env,
        shards_only=args.shards_only,
        worker_tag=args.worker_tag,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))


if __name__ == "__main__":
    main()
