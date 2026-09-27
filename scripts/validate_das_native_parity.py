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
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs

# Corpus seam (owner-ratified reuse; ADR-052 D14). ``pining_source`` lists refs (resume-before-load
# over ``list_match_refs``) and loads one full match per item; exposed at module scope so a test can
# inject a fake corpus through it (tests/scripts/_fake_corpus.install_fake_corpus patches this name).
from scripts._loader_pining import pining_source, resolve_cache_dir
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._das_params import DAS_PARAMS
from silly_kicks.tracking._gk_resolve import resolve_defended_goals

_FRAME_KEYS = ("game_id", "period_id", "frame_id")
_X_OFFSET = 52.5
_Y_OFFSET = 34.0

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

_SHARD_SCHEMA_VERSION = "das-native-parity-1"
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
    # --- match rows: per-match scalars (NaN on team/player rows) ---
    "n_scored_frames",
    "ms_frame_ref",
    "ms_frame_numpy",
    "ms_frame_numba",
    "ms_frame_periodic",
    *(_REASON_COLS[k] for k in sorted(_REASON_COLS)),
]


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056)."""
    return declare_inputs(
        driver="validate_das_native_parity",
        params={
            "das_params": dataclasses.asdict(DAS_PARAMS),
            "reference_quadrature": _REFERENCE_PARAMS.quadrature,
            "thread_sweep": list(_THREAD_SWEEP),
            "schema": _SHARD_SCHEMA_VERSION,
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


def _run_native(
    frames: pd.DataFrame, params, *, engine: Literal["auto", "numpy", "numba"]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Pack ``frames`` (direction from :data:`_DIR_COL`), run ``compute_das``, return tidy team/player
    frames sorted by key -- the driver's own copy of the test helper (scripts must not import tests)."""
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
                "player_id": [int(canonical_id(v)) for v in pids],  # type: ignore[arg-type]  # player rows: never NA
                "player_as": res.player_as,
                "player_das": res.player_das,
            }
        )
        .sort_values([*_FRAME_KEYS, "player_id"], kind="stable")
        .reset_index(drop=True)
    )
    return team, player, packed.reason


# The ``accessible-space`` call kwargs -- byte-for-byte the golden generator's ``_COMMON``
# (``tests/tracking/_fixtures/das_golden/_generate.py``). ``attacking_direction_col`` is :data:`_DIR_COL`
# (the shared per-frame direction) rather than the generator's ``"dir"``, so all four legs pin the SAME
# direction. ``test_das_native_parity_driver`` gate-checks this against the generator to catch drift.
_REFERENCE_COMMON: dict[str, Any] = dict(
    frame_col="frame_id",
    player_col="player_id",
    team_col="team_id",
    x_col="x",
    y_col="y",
    vx_col="vx",
    vy_col="vy",
    team_in_possession_col="team_in_possession",
    ball_player_id="ball",
    period_col="period_id",
    attacking_direction_col=_DIR_COL,
    infer_attacking_direction=False,
    use_progress_bar=False,
)


def _reference_lib_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Library-input copy, byte-for-byte the golden generator's ``_lib_frames``: centre coords to the
    ``accessible-space`` origin, float64, ball ``player_id`` -> ``"ball"`` / ball team ``None``, and
    string ``"p<id>"`` / ``"t<id>"`` ids (the recipe the frozen golden reference was produced with).

    accessible-space 2.0.15 is numpy-era and does ``arr[:, np.newaxis]`` on the team array; a
    pyarrow-backed column (real pining frames use the arrow dtype backend) raises
    ``IndexError: too many indices``, so every arrow column is coerced to numpy first. The native
    engine consumes arrow frames fine (``np.asarray`` at the pack boundary); this is a reference-leg
    input requirement only.
    """
    out = frames.copy()
    out["x"] = out["x"].astype("float64") - _X_OFFSET
    out["y"] = out["y"].astype("float64") - _Y_OFFSET
    out["vx"] = out["vx"].astype("float64")
    out["vy"] = out["vy"].astype("float64")
    is_ball = out["is_ball"].to_numpy(dtype=bool)
    pid = out["player_id"].astype("object")
    pid[is_ball] = "ball"
    pid[~is_ball] = ["p" + str(int(v)) for v in out.loc[~is_ball, "player_id"]]
    out["player_id"] = pid
    tid = out["team_id"].astype("object")
    tid[is_ball] = None
    tid[~is_ball] = ["t" + str(int(v)) for v in out.loc[~is_ball, "team_id"]]
    out["team_id"] = tid
    out["team_in_possession"] = ["t" + str(int(v)) if pd.notna(v) else None for v in out["team_in_possession"]]
    # accessible-space 2.0.15 is numpy-era and does ``team_array[:, np.newaxis]``; under pandas 3 a
    # string-list assignment defaults to the pyarrow-backed ``str`` StringDtype, whose ArrowStringArray
    # raises ``IndexError: too many indices``. Force the id / possession columns the library arrays to
    # plain ``object`` (numpy-backed) -- the golden generator escapes this only because it runs on pandas 2.
    for c in ("player_id", "team_id", "team_in_possession"):
        out[c] = out[c].astype(object)
    return out


def _reference_leg(frames: pd.DataFrame) -> dict[str, np.ndarray]:
    """The ``accessible-space`` 2.0.15 leg (LAZILY imported; the ``das-reference`` dev extra).

    Prepares the library input and reads the ``ReturnValueDAS`` arrays EXACTLY as the golden generator's
    ``_run_team_and_player`` / ``_emit_team_player`` do -- ``team.acc_space`` / ``team.das`` and
    ``ind.player_acc_space`` / ``ind.player_das`` align to the POSSESSION-FILTERED rows in input order,
    team values deduped per frame. Returns team/player AS/DAS keyed by ``(game, period, frame[, player])``.
    The driver's test REPLACES this whole function with the frozen golden reference outputs, so the corpus
    reduce path runs with no library and no network.
    """
    import accessible_space as asp  # pyright: ignore[reportMissingImports]  # lazy; das-reference dev extra only, absent in CI

    lib = _reference_lib_frames(frames).reset_index(drop=True)
    team = asp.get_dangerous_accessible_space(lib.copy(), **_REFERENCE_COMMON)
    ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **_REFERENCE_COMMON)
    as_t = np.asarray(team.acc_space, dtype=float)
    das_t = np.asarray(team.das, dtype=float)
    as_p = np.asarray(ind.player_acc_space, dtype=float)
    das_p = np.asarray(ind.player_das, dtype=float)

    kept = lib["team_in_possession"].notna().to_numpy()
    per_frame: dict[tuple, tuple[float, float]] = {}
    player_rows: list[tuple] = []
    ti = 0
    for i in range(len(lib)):
        if not kept[i]:
            continue
        r = lib.iloc[i]
        key = (r["game_id"], r["period_id"], r["frame_id"])
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        if not bool(r["is_ball"]):
            player_rows.append((*key, int(str(r["player_id"])[1:]), as_p[ti], das_p[ti]))
        ti += 1

    team_keys = sorted(per_frame)
    pid_sorted = sorted(player_rows, key=lambda t: (t[0], t[1], t[2], t[3]))
    return {
        "team_keys": np.array(team_keys, dtype=object),
        "team_as": np.array([per_frame[k][0] for k in team_keys], dtype=float),
        "team_das": np.array([per_frame[k][1] for k in team_keys], dtype=float),
        "player_keys": np.array([r[:4] for r in pid_sorted], dtype=object),
        "player_as": np.array([r[4] for r in pid_sorted], dtype=float),
        "player_das": np.array([r[5] for r in pid_sorted], dtype=float),
    }


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


def _measure_match(item, *, reference_leg=_reference_leg, direction_col: str | None = None) -> pd.DataFrame:
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
    ((np_team, np_player, reason), t_np) = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numpy"))
    if _numba_available():
        ((nb_team, _nb_player, _r), t_nb) = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numba"))
    else:
        nb_team, t_nb = None, np.nan
    ((per_team, _pp, _r2), t_per) = _time_leg(lambda: _run_native(scored, DAS_PARAMS, engine="numpy"))

    rows: list[dict] = []
    rows += _team_rows(provider, ref, np_team, nb_team, per_team)
    rows += _player_rows(provider, ref, np_player)

    reason_counts = pd.Series(reason).map(_REASON_COLS).value_counts().to_dict()
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
    rows.append(match_row)

    return pd.DataFrame(rows).reindex(columns=_EMITTED_SHARD_COLUMNS)


def _join_on_keys(ref_keys, ref_vals: dict, native: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Align the reference arrays (keyed by ``ref_keys``) to a native tidy frame on the shared keys.

    The reference keys may be an object array (production ``_reference_leg``) while native keys are
    typed; cast each key column to the native dtype so the merge aligns (an int-vs-object key silently
    matches NOTHING and the parity would read as a vacuous zero-row join)."""
    ref_df = pd.DataFrame(ref_keys, columns=key_cols)
    for c in key_cols:
        ref_df[c] = ref_df[c].astype(native[c].dtype)
    for name, arr in ref_vals.items():
        ref_df[name] = arr
    return ref_df.merge(native, on=key_cols, how="inner", suffixes=("_ref", "_nat"))


def _team_rows(provider, ref, np_team, nb_team, per_team) -> list[dict]:
    merged = _join_on_keys(
        ref["team_keys"],
        {"ref_as": ref["team_as"], "ref_das": ref["team_das"]},
        np_team,
        list(_FRAME_KEYS),
    )
    if nb_team is not None:
        merged = merged.merge(
            nb_team.rename(columns={"team_das": "nb_das"})[[*_FRAME_KEYS, "nb_das"]], on=list(_FRAME_KEYS), how="left"
        )
    merged = merged.merge(
        per_team.rename(columns={"team_das": "per_das"})[[*_FRAME_KEYS, "per_das"]], on=list(_FRAME_KEYS), how="left"
    )
    abs_das, rel_das = _abs_rel(merged["ref_das"].to_numpy(), merged["team_das"].to_numpy())
    abs_as, rel_as = _abs_rel(merged["ref_as"].to_numpy(), merged["team_as"].to_numpy())
    nb = merged["nb_das"].to_numpy() if "nb_das" in merged.columns else np.full(len(merged), np.nan)
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
            }
        )
    return rows


def _player_rows(provider, ref, np_player) -> list[dict]:
    merged = _join_on_keys(
        ref["player_keys"],
        {"ref_as": ref["player_as"], "ref_das": ref["player_das"]},
        np_player,
        [*_FRAME_KEYS, "player_id"],
    )
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


def reduce_parity(shards: list[pd.DataFrame]) -> dict:
    """Corpus parity/perf statistics per provider (spec 7.2). Empty shards -> ``{}``."""
    if not shards:
        return {}
    combined = pd.concat(shards, ignore_index=True)
    if combined.empty:
        return {}
    out: dict = {}
    for provider, sub in combined.groupby("provider"):
        team = sub[sub["grain"] == "team"]
        player = sub[sub["grain"] == "player"]
        match = sub[sub["grain"] == "match"]

        def _mask_mismatch(g: pd.DataFrame) -> int:
            return int((g["finite_ref"].astype("boolean") != g["finite_native"].astype("boolean")).sum())

        quad = team["quad_shift_das"].abs()
        nbmax = team["numba_minus_numpy_das"].abs()
        out[str(provider)] = {
            "team": _grade_grain(team) if not team.empty else {},
            "player": _grade_grain(player) if not player.empty else {},
            "finite_mask_mismatches": {"team": _mask_mismatch(team), "player": _mask_mismatch(player)},
            "quadrature_shift_das": {"median": _pct(quad, 50), "p90": _pct(quad, 90), "max": _pct(quad, 100)},
            "numba_vs_numpy_das_max_abs": _pct(nbmax, 100),
            "timings_ms_per_frame": {
                leg: _pct(match[f"ms_frame_{leg}"], 50) for leg in ("ref", "numpy", "numba", "periodic")
            },
            "reason_counts": {col: int(match[col].fillna(0).sum()) for col in _REASON_COLS.values()},
            "n_scored_frames": int(match["n_scored_frames"].fillna(0).sum()),
            "n_matches_scored": int(match["game_id"].nunique()),
        }
    return out


def run_corpus(
    refs,
    load,
    dest: Path,
    *,
    prov: dict,
    shard_root: Path | None = None,
    direction_col: str | None = None,
    reference_leg=_reference_leg,
) -> dict:
    """The map+reduce+write, factored out of ``main`` so the reduce PATH is testable offline.

    ``load(ref) -> (provider, match_id, actions, frames)``; ``reference_leg`` and ``direction_col`` are
    injected by the test (golden reference outputs + the golden ``dir`` column), so the full corpus
    pass runs with no ``accessible-space`` and no network. Production passes the defaults (the lazy
    ``accessible-space`` leg and the ``GoalMap`` direction).
    """
    from scripts._driver import for_each

    def _work(item):
        return _measure_match(item, reference_leg=reference_leg, direction_col=direction_col)

    res = for_each(
        refs,
        key=lambda ref: ref.key,
        load=load,
        work=_work,
        shard_root=shard_root if shard_root is not None else dest / "shards",
        token_inputs={
            "metric": "das_native_parity",
            "schema": _SHARD_SCHEMA_VERSION,
            "direction": direction_col or "goal_map",
        },
        label="match",
    )
    shards = [pd.read_parquet(s) for s in sorted(res.shard_dir.glob("*.parquet"))]
    parity = reduce_parity(shards)
    scored_providers = {p: parity[p]["n_matches_scored"] for p in parity}
    manifest = res.manifest()
    out = {
        "providers": parity,
        "population": _population(refs, scored_providers, manifest),
        **manifest,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "run_platform": prov.get("platform"),
        "run_machine": prov.get("machine"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _population(refs, scored_providers: dict, manifest: dict) -> dict:
    """Matches listed / scored / excluded per provider (spec 7.2). SB360 is structurally unscoreable
    (velocity-less freeze-frames, ADR-063): recorded, not silently dropped."""
    listed: dict[str, int] = {}
    for ref in refs:
        listed[ref.provider] = listed.get(ref.provider, 0) + 1
    return {
        "listed_per_provider": listed,
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


# --------------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--shard-root", default=None, help="shard root (default <out>/shards)")
    ap.add_argument("--providers", nargs="*", default=None, help="providers to walk (default: all velocity-bearing)")
    ap.add_argument("--token", default=None, help="pining token (else resolved from the environment)")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    args = ap.parse_args()

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
    refs, base_load = pining_source(providers, token=args.token, max_per_provider=args.max_matches, cache_dir=cache_dir)

    def _load(ref):
        lm = base_load(ref)
        return (lm.provider, lm.match_id, lm.actions, lm.frames)

    # Production defaults: the lazy accessible-space reference leg + GoalMap direction (ADR-055).
    out = run_corpus(
        refs,
        _load,
        dest,
        prov=prov,
        shard_root=Path(args.shard_root) if args.shard_root else None,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))


if __name__ == "__main__":
    main()
