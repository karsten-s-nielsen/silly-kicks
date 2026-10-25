"""Shared plumbing for the TF-46 Arm 1 Part 2 pre-flight drivers (spec sections 5, 16.2, 16.3).

Pure functions and parameters-only models, ASCII-only (the driver ASCII gate), no I/O except ``read_upstream``
and ``video_census`` which read explicit paths. Each pre-flight driver (``scripts/tf46p2_*.py``) is a thin
``scripts/_driver.py`` / ``scripts/_stages.py`` wrapper around these cores, so the CI-tested surface stays here.

The camera-follow model (R3), the off-camera replay (R2) and the stoppage-time rule (C-PLAN-24) are implemented
as parameters-only (ADR-011: numpy arrays + JSON, no pickle). The gate-table constants, the fidelity tolerances
and the dependency map live in ``silly_kicks.tracking._restart_validation`` and are imported, never re-declared.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.tracking._restart_validation import RATIFIED, DataProfile, sensitivity_values


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


# ---------------------------------------------------------------------------
# Data profile and LOMO folds
# ---------------------------------------------------------------------------


#: The committed research root the pre-flight aggregates land under (spec section 16.2).
RESEARCH_ROOT = "docs/research/tf46_part2"


#: The assembled pre-flight aggregate the shipped gate table derives from (``tf46p2_gate_table``, spec 16.2).
GATE_AGGREGATE_FILE = "gate_aggregate.parquet"


def read_gate_aggregate(root: str | Path = RESEARCH_ROOT) -> tuple[pd.DataFrame, str]:
    """The committed gate aggregate and its version (plan Task 2.14 Step 1, ADR-056): ``(aggregate, sha256)`` when
    ``<root>/gate_aggregate.parquet`` exists, else ``(empty AGGREGATE_COLUMNS frame, "unpinned")`` -- the commit-2
    state, which derives to the EMPTY table. The library never reads this file (the derived table is pasted as a
    literal at C3); the derivation gate reads it to prove the literal equals the derivation."""
    import hashlib

    from silly_kicks.tracking._restart_validation import AGGREGATE_COLUMNS

    base = Path(root)
    if not base.is_absolute():  # anchored at the repository root, never the caller's working directory
        base = Path(__file__).resolve().parents[1] / base
    path = base / GATE_AGGREGATE_FILE
    if not path.exists():
        return pd.DataFrame(columns=list(AGGREGATE_COLUMNS)), "unpinned"
    return pd.read_parquet(path), hashlib.sha256(path.read_bytes()).hexdigest()


def gate_table_derivation_mismatches(table, version: str, aggregate: pd.DataFrame, aggregate_version: str) -> list[str]:
    """The ADR-056 derivation check: every way the shipped ``(RESTART_GATE_TABLE, GATE_TABLE_VERSION)`` differs
    from ``derive_gate_table(aggregate)`` and the aggregate's version (empty list = consistent)."""
    from silly_kicks.tracking._restart_validation import derive_gate_table

    derived = derive_gate_table(aggregate)
    shipped = dict(table)
    out = []
    for key in sorted(set(shipped) | set(derived), key=repr):
        if key not in derived:
            out.append(f"shipped but not derived: {key}")
        elif key not in shipped:
            out.append(f"derived but not shipped: {key}")
        elif shipped[key] != derived[key]:
            out.append(f"entry differs: {key}")
    if version != aggregate_version:
        out.append(f"GATE_TABLE_VERSION {version!r} != aggregate version {aggregate_version!r}")
    return out


def write_run_aggregate(table: pd.DataFrame, out_dir: Path, run: str, prov: Mapping[str, Any]) -> Path:
    """Write ONE tier-named aggregate ``<run>.parquet`` plus a ``<run>.json`` provenance sidecar (spec section
    16.2). The sidecar carries top-level ``run_commit`` and ``run_tree_dirty: false`` (an artifact-writing
    driver refuses a dirty tree, ADR-037). No per-row provider field that rebuilds raw data (ADR-116). Returns
    the parquet path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = out / f"{run}.parquet"
    table.to_parquet(agg, index=False)
    side = out / f"{run}.json"
    side.write_text(
        json.dumps(
            {
                "run": run,
                "n_rows": len(table),
                "run_commit": prov["commit"],
                "run_tree_dirty": bool(prov["dirty"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return agg


def data_profile_for(provider: str, meta: Mapping[str, Any]) -> DataProfile:
    """Build the :class:`DataProfile` key for a match (spec section 5.1). ``meta`` carries the per-match
    release metadata; a missing field gets the per-profile fallback ``"unknown"`` (recorded, never guessed).
    GI / tracking pairing is NOT a profile field -- it is per match (spec section 6.1)."""
    return DataProfile(
        provider=str(provider),
        tracking_release=str(meta.get("tracking_release", "unknown")),
        gi_release=(None if meta.get("gi_release") in (None, "") else str(meta["gi_release"])),
        competition_season=str(meta.get("competition_season", "unknown")),
    )


def lomo_folds(match_ids: Sequence[str], *, seed: int = 46) -> list[tuple[list[str], str]]:
    """Leave-one-match-out folds, pre-registered before any gate is scored (CL-4, spec section 16.2 run 8).

    Returns ``(train_ids, held_out_id)`` per match. The held-out order is a seeded permutation of the sorted
    unique ids, so a run is reproducible and the fold set never depends on the input order.
    """
    uniq = sorted({str(m) for m in match_ids})
    if len(uniq) < 2:
        return []
    order = np.random.default_rng(seed).permutation(len(uniq))
    folds: list[tuple[list[str], str]] = []
    for i in order:
        held = uniq[i]
        folds.append(([m for m in uniq if m != held], held))
    return folds


# ---------------------------------------------------------------------------
# Anchor simulation (spec section 5.3) and the corpus GI-direction guard (section 4.3)
# ---------------------------------------------------------------------------


def simulate_skillcorner_anchor(actions_native: pd.DataFrame) -> pd.DataFrame:
    """Build a SkillCorner-like bracket from native events (spec section 5.3 anchor simulation).

    For each restart row (``restart_evidence`` present, or ``type_name`` a set piece), ``lo`` = the time of the
    last event before the stoppage (the previous event's ``time_seconds``), ``hi`` = the first touch of the next
    possession (the next event's ``time_seconds``). The resolver then runs INSIDE ``[lo, hi]``. Pure (new
    frame): columns ``action_id``, ``period_id``, ``kick_bracket_lo``, ``kick_bracket_hi``.
    """
    cols = ["action_id", "period_id", "kick_bracket_lo", "kick_bracket_hi"]
    if actions_native.empty:
        return pd.DataFrame(columns=cols)
    a = actions_native.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    is_restart = _restart_row_mask(a)
    t = a["time_seconds"].to_numpy(dtype=float)
    pid = a["period_id"].to_numpy()
    rows = []
    for i in np.nonzero(is_restart)[0]:
        lo = t[i - 1] if i > 0 and pid[i - 1] == pid[i] else t[i]
        hi = t[i + 1] if i + 1 < len(a) and pid[i + 1] == pid[i] else t[i]
        rows.append(
            {"action_id": a["action_id"].iloc[i], "period_id": pid[i], "kick_bracket_lo": lo, "kick_bracket_hi": hi}
        )
    return pd.DataFrame(rows, columns=cols)


def _restart_row_mask(actions: pd.DataFrame) -> np.ndarray:
    """A restart row: a set-piece / goal-kick ``type_name``, or a non-null ``restart_evidence``."""
    sp = {
        "corner_crossed",
        "corner_short",
        "freekick_crossed",
        "freekick_short",
        "goalkick",
        "throw_in",
        "shot_penalty",
    }
    mask = np.zeros(len(actions), dtype=bool)
    if "type_name" in actions.columns:
        mask |= actions["type_name"].isin(sp).to_numpy()
    if "restart_evidence" in actions.columns:
        mask |= actions["restart_evidence"].notna().to_numpy()
    return mask


def assert_gi_direction_corpus(checks: pd.DataFrame, *, max_share: float = 0.01) -> None:
    """The section 4.3 corpus guard (drivers and validation runs only): raise when more than ``max_share`` of the
    (match, period) GI-direction checks disagree with GoalMap, OR when any match has BOTH regular periods (1 and 2)
    mismatched -- either pattern is a systematic frame or flip bug, not GI noise.

    ``checks`` carries one row per (match, period): ``match_id``, ``period_id``, boolean ``agrees``, and optional
    boolean ``resolved`` (False = the period had no recorded frame flip, so it was never compared; absent = every
    row resolved). More than ``max_share`` unresolved periods also raises: a lost flip record would otherwise
    pass the guard vacuously (R.2 2026-10-23). Raises ``AssertionError`` naming the failed pattern; a clean corpus
    returns ``None``.
    """
    if checks.empty:
        return
    if "resolved" in checks.columns:
        resolved = checks["resolved"].astype(bool)
        unresolved = float((~resolved).mean())
        if unresolved > max_share:
            raise AssertionError(
                f"GI direction unresolved share {unresolved:.4f} exceeds the {max_share:.4f} ceiling "
                "(no recorded frame flip: the orientation record was lost)"
            )
        checks = checks.loc[resolved.to_numpy()]
        if checks.empty:
            return
    agrees = checks["agrees"].astype(bool)
    disagree = float((~agrees).mean())
    if disagree > max_share:
        raise AssertionError(f"GI direction mismatch share {disagree:.4f} exceeds the {max_share:.4f} ceiling")
    bad = checks.loc[~agrees.to_numpy()]
    for mid, g in bad.groupby("match_id", sort=True):
        if {1, 2} <= {int(p) for p in g["period_id"]}:
            raise AssertionError(f"GI direction: match {mid} has BOTH periods mismatched (a systematic flip bug)")


# ---------------------------------------------------------------------------
# Truth extraction (spec section 5.6)
# ---------------------------------------------------------------------------

#: DFL event types that are a ball touch (spec section 5.6). A standalone ``BallClaiming`` counts.
_DFL_TOUCH_TYPES: frozenset[str] = frozenset({"Play", "OtherBallAction", "TacklingGame", "ShotAtGoal", "BallClaiming"})
#: DFL event types that are never a touch.
_DFL_NON_TOUCH: frozenset[str] = frozenset({"Foul", "Caution", "Substitution", "FairPlay", "Offside"})


def dfl_first_touch_truth(events: pd.DataFrame) -> pd.DataFrame:
    """Per restart, the DFL first-touch truth (spec section 5.6).

    ``events`` columns: ``restart_id``, ``seq`` (order after the kick), ``event_type``, ``player_id``,
    ``team_id``, ``timestamp_seconds``, and ``tackle_winner`` / ``tackle_winner_team`` (nullable). Rules:
    ``Delete`` rows are dropped; a ``TacklingGame`` yields the winner (player + team, the "who won the first
    ball" meaning) and is flagged ``is_duel``; a ``BallClaiming`` co-timed (+/-0.1 s) with another event of the
    SAME player yields to its partner; a standalone ``BallClaiming`` counts; a first event that is a ``Foul``
    gives class ``foul`` (excluded from the agreement gate, counted). Columns: ``restart_id``, ``player_id``,
    ``team_id``, ``timestamp_seconds``, ``touch_class`` (``touch`` / ``duel`` / ``foul`` / ``none``).
    """
    cols = ["restart_id", "player_id", "team_id", "timestamp_seconds", "touch_class"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    ev = events[events["event_type"] != "Delete"].copy()
    rows = []
    for rid, g in ev.groupby("restart_id", sort=True):
        g = g.sort_values("seq").reset_index(drop=True)
        rows.append(_dfl_first_touch_one(rid, g))
    return pd.DataFrame(rows, columns=cols)


def _dfl_first_touch_one(rid: Any, g: pd.DataFrame) -> dict:
    none_row = {
        "restart_id": rid,
        "player_id": pd.NA,
        "team_id": pd.NA,
        "timestamp_seconds": np.nan,
        "touch_class": "none",
    }
    for i in range(len(g)):
        et = str(g["event_type"].iloc[i])
        if et in _DFL_NON_TOUCH:
            if et == "Foul":  # a corner whose first event is a foul: class foul, counted, not scored
                r = dict(none_row)
                r.update(
                    team_id=g["team_id"].iloc[i],
                    timestamp_seconds=float(g["timestamp_seconds"].iloc[i]),
                    touch_class="foul",
                )
                return r
            continue
        if et not in _DFL_TOUCH_TYPES:
            continue
        if et == "TacklingGame":
            return {
                "restart_id": rid,
                "player_id": (g["tackle_winner"].iloc[i] if "tackle_winner" in g.columns else g["player_id"].iloc[i]),
                "team_id": (
                    g["tackle_winner_team"].iloc[i] if "tackle_winner_team" in g.columns else g["team_id"].iloc[i]
                ),
                "timestamp_seconds": float(g["timestamp_seconds"].iloc[i]),
                "touch_class": "duel",
            }
        if et == "BallClaiming" and _has_cotimed_partner(g, i):
            continue  # the partner event (same player, +/-0.1 s) is the touch
        return {
            "restart_id": rid,
            "player_id": g["player_id"].iloc[i],
            "team_id": g["team_id"].iloc[i],
            "timestamp_seconds": float(g["timestamp_seconds"].iloc[i]),
            "touch_class": "touch",
        }
    return none_row


def _has_cotimed_partner(g: pd.DataFrame, i: int) -> bool:
    """True when another event of the SAME player is within +/-0.1 s of event ``i`` (spec section 5.6)."""
    t = float(g["timestamp_seconds"].iloc[i])
    pid = canonical_id(g["player_id"].iloc[i])
    for j in range(len(g)):
        if j == i:
            continue
        if same_id(g["player_id"].iloc[j], pid) and abs(float(g["timestamp_seconds"].iloc[j]) - t) <= 0.1:
            return True
    return False


def pff_truth(events: pd.DataFrame, tracking_index: pd.DataFrame) -> pd.DataFrame:
    """Per restart, the GS / PFF set-piece taker and first touch (spec section 5.6).

    Reads ONLY the named PFF fields -- never the PFF event ball / player arrays (a tracking snapshot, so GS has
    no independent next-touch LOCATION truth). ``events`` columns: ``restart_id``, ``setpiece_type``,
    ``taker_player_id`` (``gameEvents.playerId``), ``first_touch_player_id``, ``first_touch_is_challenge``,
    ``challenge_winner_player_id``, ``subtype`` (``possessionEventType`` CR / PA), ``game_event_id`` (-> anchor
    frame). ``tracking_index`` maps ``game_event_id`` to ``frame_id`` for the anchor. Columns: ``restart_id``,
    ``taker_player_id``, ``first_touch_player_id``, ``subtype``, ``anchor_frame_id``.
    """
    cols = ["restart_id", "taker_player_id", "first_touch_player_id", "subtype", "anchor_frame_id"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    frame_of = (
        dict(zip(tracking_index["game_event_id"], tracking_index["frame_id"], strict=True))
        if tracking_index is not None and not tracking_index.empty
        else {}
    )
    rows = []
    for _, e in events.iterrows():
        challenge = bool(e.get("first_touch_is_challenge", False))
        ft = e.get("challenge_winner_player_id") if challenge else e.get("first_touch_player_id")
        rows.append(
            {
                "restart_id": e["restart_id"],
                "taker_player_id": e.get("taker_player_id", pd.NA),
                "first_touch_player_id": ft if ft is not None else pd.NA,
                "subtype": _pff_subtype(e.get("subtype")),
                "anchor_frame_id": frame_of.get(e.get("game_event_id"), pd.NA),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _pff_subtype(raw: Any) -> Any:
    """PFF ``possessionEventType`` -> the SPADL subtype label (``CR`` crossed, ``PA`` short)."""
    return {"CR": "crossed", "PA": "short"}.get(str(raw), pd.NA)


def location_field_characterisation(events: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Characterise each DFL location field as independent truth or a tracking copy (spec section 5.2 / C-30).

    For each event-location field it compares the event position with the tracked actor position at the anchor
    frame; a field whose median absolute difference is below ``copy_tol_m`` is a TRACKING COPY (``is_copy``
    True), NOT independent truth. ``events`` carries ``event_type``, ``field``, ``x``, ``y``, ``player_id``,
    ``frame_id``; ``frames`` carries the tracked ``x`` / ``y`` per (``frame_id``, ``player_id``). Columns:
    ``event_type``, ``field``, ``n``, ``median_abs_diff_m``, ``is_copy``.
    """
    cols = ["event_type", "field", "n", "median_abs_diff_m", "is_copy"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    copy_tol_m = 0.1  # within 10 cm of the tracked actor position == a tracking copy, not independent truth
    fidx = frames.set_index(["frame_id", "player_id"])[["x", "y"]] if not frames.empty else None
    rows = []
    for (et, fld), g in events.groupby(["event_type", "field"], sort=True):
        diffs = []
        for _, e in g.iterrows():
            if fidx is None:
                continue
            key = (e["frame_id"], e["player_id"])
            if key not in fidx.index:
                continue
            tx, ty = fidx.loc[key, "x"], fidx.loc[key, "y"]
            diffs.append(float(np.hypot(float(e["x"]) - _f(tx), float(e["y"]) - _f(ty))))
        med = float(np.median(diffs)) if diffs else float("nan")
        rows.append(
            {
                "event_type": et,
                "field": fld,
                "n": len(diffs),
                "median_abs_diff_m": med,
                "is_copy": bool(np.isfinite(med) and med <= copy_tol_m),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def coverage_report(outputs: pd.DataFrame) -> pd.DataFrame:
    """Non-NaN SHARE of a refined signal per (data profile x cell x evidence x cell resolution) (spec section 19).

    ``outputs`` carries ``data_profile`` (a string key), ``cell`` (restart_type x cell_state or the signal),
    ``evidence``, ``cell_resolution`` and ``value`` (the shipped number, NaN when withheld). Counts only (no raw
    per-row provider fields; ADR-116). Columns: the four keys, ``n``, ``n_valued``, ``valued_share``.
    """
    keys = ["data_profile", "cell", "evidence", "cell_resolution"]
    cols = [*keys, "n", "n_valued", "valued_share"]
    if outputs.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in outputs.groupby(keys, dropna=False, sort=True):
        n = len(g)
        n_valued = int(g["value"].notna().sum())
        share = (n_valued / n) if n else float("nan")
        rows.append(dict(zip(keys, key, strict=True)) | {"n": n, "n_valued": n_valued, "valued_share": share})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Inter-run flow (C-03) and course-tier de-duplication (section 18.1 amendment)
# ---------------------------------------------------------------------------


def read_upstream(run_root: Path, run: str, *, expect_commit: str) -> tuple[pd.DataFrame, dict]:
    """Read an upstream run aggregate and its provenance sidecar, refusing on drift (C-03, spec section 16.2).

    Raises ``FileNotFoundError`` when the aggregate or the ``.json`` sidecar is absent, and ``RuntimeError``
    when the sidecar ``run_commit`` differs from ``expect_commit`` (a downstream run must read inputs built at
    the same commit). Returns ``(aggregate, sidecar)``.
    """
    root = Path(run_root)
    agg = root / f"{run}.parquet"
    side = root / f"{run}.json"
    if not agg.exists():
        raise FileNotFoundError(f"upstream aggregate missing: {agg}")
    if not side.exists():
        raise FileNotFoundError(f"upstream provenance sidecar missing: {side}")
    sidecar = json.loads(side.read_text(encoding="utf-8"))
    got = sidecar.get("run_commit")
    if got != expect_commit:
        raise RuntimeError(f"upstream {run}: run_commit {got!r} differs from expected {expect_commit!r}")
    return pd.read_parquet(agg), sidecar


def dedup_course_tier(course_ids: pd.DataFrame, owner_ids: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only course matches NOT already in the owner tier (spec section 18.1 amendment, C-04).

    A match is the same when its (``competition``, ``date``, ``home_team``, ``away_team``) agree. Returns the
    de-duplicated course frame and the overlap COUNT. No match id leaves this function in an artifact beyond the
    count (ADR-116).
    """
    key = ["competition", "date", "home_team", "away_team"]
    if course_ids.empty:
        return course_ids.copy(), 0
    owner_keys = set(map(tuple, owner_ids[key].astype(str).to_numpy())) if not owner_ids.empty else set()
    course_key = list(map(tuple, course_ids[key].astype(str).to_numpy()))
    keep = np.array([k not in owner_keys for k in course_key], dtype=bool)
    return course_ids[keep].reset_index(drop=True), int((~keep).sum())


# ---------------------------------------------------------------------------
# Census (C-05) and GI-pairing report (C-11)
# ---------------------------------------------------------------------------

#: Criteria a video-census list entry MUST carry (spec section 16.2 run 15, C-05).
_CENSUS_CRITERIA: tuple[str, ...] = ("match_id", "source", "full_match", "publicly_accessible", "same_fixture")


def video_census(source_list: Path, profiles: Mapping[str, DataProfile]) -> pd.DataFrame:
    """Count matches with accessible full-match broadcast video per data profile (spec section 16.2 run 15).

    The DL / owner compiles the reviewed ``match_id -> source`` list OUTSIDE the repo (passed by path). This
    validates the schema (every ``_CENSUS_CRITERIA`` field present) and the criteria, then writes COUNTS ONLY
    per data profile (ADR-116: no match id or source in the output). Raises ``ValueError`` on a missing field.
    """
    entries = json.loads(Path(source_list).read_text(encoding="utf-8"))
    for e in entries:
        missing = [c for c in _CENSUS_CRITERIA if c not in e]
        if missing:
            raise ValueError(f"census entry missing criteria fields {missing}")
    rows = []
    for mid_key, prof in profiles.items():
        eligible = [
            e
            for e in entries
            if str(e["match_id"]) == str(mid_key)
            and bool(e["full_match"])
            and bool(e["publicly_accessible"])
            and bool(e["same_fixture"])
        ]
        rows.append({"data_profile": _profile_key(prof), "n_with_video": len(eligible)})
    out = pd.DataFrame(rows, columns=["data_profile", "n_with_video"])
    return out.groupby("data_profile", sort=True)["n_with_video"].sum().reset_index()


def _profile_key(p: DataProfile) -> str:
    """A stable string key for a DataProfile (ADR-116-safe: no reversible raw ids)."""
    return f"{p.provider}|{p.tracking_release}|{p.gi_release}|{p.competition_season}"


#: The ratified pairing constants run (2) re-evaluates per match at their sensitivity values (C-11, C-PLAN-16).
#: The proximity bar is DERIVED (OD-6 amendment 2026-10-23), so its sensitivity moves the derivation inputs --
#: ``pair_prox_margin`` and ``pair_prox_m`` (the bar re-derived at each distance) -- in :func:`pairing_bar_table`,
#: never the derived output itself.
PAIRING_SENSITIVITY_KEYS: tuple[str, ...] = (
    "pair_lag_tol_frames",
    "pair_lag_scan_frames",
)

#: The run (2) proximity distances: the ratified ``pair_prox_m`` and its +/-50 % values (1.0, 2.0, 3.0 m).
PAIRING_PROX_MS: tuple[float, ...] = tuple(
    sorted({*(float(v) for v in sensitivity_values("pair_prox_m")), float(RATIFIED["pair_prox_m"])})
)

#: The per-match pairing verdict columns (DL ruling 2026-10-23; ``<NA>`` = not evaluable).
PAIRING_VERDICT_COLUMNS: tuple[str, ...] = ("pair_range_ok", "pair_proximity_ok", "pair_lag_ok", "pair_direction_ok")


def gi_pairing_report(
    pairings: Mapping[object, Any],
    profile: DataProfile,
    *,
    is_opendata: bool = False,
    sensitivity: Mapping[str, Mapping[object, Any]] | None = None,
) -> pd.DataFrame:
    """Per-match GI / tracking pairing report (C-11, spec section 6.1, run 2).

    ``pairings`` maps a match id to a ``GiPairing`` (``silly_kicks.tracking._restart_frame``) at the ratified
    constants. Each row carries the match id (ADR-116: ids are non-reversible references, kept on every tier;
    C-PLAN-18), ``paired`` (``GiPairing.pairing == "paired"``), the proximity share, the alignment offset and
    agreement, the tied-offset count, the alignment source, the failed test, the provenance flag,
    ``is_opendata`` (opendata ships GI and tracking together: known-paired, ``assert_known_paired``), and -- per
    ``sensitivity`` label ``"<constant>=<value>"`` -- the verdict re-computed with that one constant moved (one at a
    time; integer keys rounded)."""
    sens = dict(sensitivity or {})
    cols = [
        "data_profile",
        "match_id",
        "is_opendata",
        "paired",
        "prox_share",
        "alignment_offset",
        "alignment_agreement",
        "n_tied_offsets",
        "alignment_source",
        "failed_test",
        "provenance_mismatch",
        *PAIRING_VERDICT_COLUMNS,
        "n_unresolved_periods",
        "n_insufficient_team_periods",
        "range_basis",
        "n_range_violations",
        "range_violation_classes",
        *(f"paired@{label}" for label in sorted(sens)),
    ]
    rows = []
    for mid, pr in pairings.items():
        row = {
            "data_profile": _profile_key(profile),
            "match_id": str(mid),
            "is_opendata": bool(is_opendata),
            "paired": getattr(pr, "pairing", None) == "paired",
            "prox_share": float(getattr(pr, "prox_share", float("nan"))),
            "alignment_offset": getattr(pr, "alignment_offset", pd.NA),
            "alignment_agreement": float(getattr(pr, "alignment_agreement", float("nan"))),
            "n_tied_offsets": int(getattr(pr, "n_tied_offsets", 0)),
            "alignment_source": str(getattr(pr, "alignment_source", "unknown")),
            "failed_test": getattr(pr, "failed_test", pd.NA),
            "provenance_mismatch": bool(getattr(pr, "provenance_mismatch", False)),
            **{c: (pd.NA if getattr(pr, c, None) is None else bool(getattr(pr, c))) for c in PAIRING_VERDICT_COLUMNS},
            "n_unresolved_periods": int(getattr(pr, "n_unresolved_periods", 0)),
            "n_insufficient_team_periods": int(getattr(pr, "n_insufficient_team_periods", 0)),
            # the Range test's basis and its classified violations (spec section 6.1, amended 2026-10-25): R.2
            # requires "provider" everywhere and no violation (section 26.8)
            "range_basis": str(getattr(pr, "range_basis", "retained")),
            "n_range_violations": len(getattr(pr, "range_violations", ())),
            "range_violation_classes": ",".join(sorted({v[3] for v in getattr(pr, "range_violations", ())})),
        }
        for label in sorted(sens):
            alt = sens[label].get(mid)
            row[f"paired@{label}"] = getattr(alt, "pairing", None) == "paired" if alt is not None else pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def assert_known_paired(report: pd.DataFrame) -> None:
    """Raise ``GiPairingCalibrationError`` when a known-paired opendata match is unpaired (C-11 calibration).

    Opendata ships GI and tracking together, so every opendata match MUST pair at the ratified constants; one
    that does not returns the thresholds to the owner. ``report`` carries ``is_opendata`` and ``paired``.
    """
    if report.empty or "is_opendata" not in report.columns:
        return
    bad = report[report["is_opendata"].astype(bool) & ~report["paired"].astype(bool)]
    if not bad.empty:
        raise GiPairingCalibrationError(
            f"{len(bad)} known-paired opendata match(es) unpaired at the ratified constants: "
            f"{sorted(bad['match_id'].astype(str).tolist())}"
        )


class GiPairingCalibrationError(RuntimeError):
    """A known-paired opendata match failed the pairing tests at the ratified constants (C-11)."""


# ---------------------------------------------------------------------------
# The R.2 pre-registered GI period-label + Range census (spec section 26.8; DL ruling 2026-10-25, Q6)
# ---------------------------------------------------------------------------

#: The staged dataset the census is pre-registered on (the 895-match R.2 corpus); a capped or other dataset (a smoke
#: run, the toy corpus) is not asserted, and the calibrate output records that (``r2_census_asserted``).
R2_CENSUS_DATASET = "tf46p2"
#: SkillCorner matches in the R.2 corpus.
R2_SKILLCORNER_MATCHES = 895
#: The matches whose GI period labels the converter repairs (handoff 2026-10-25 section 3.1); every other match is
#: ``clean``.
R2_PERIOD_REPAIRED_MATCHES: frozenset[str] = frozenset(
    {"1766992", "1800188", "1874585", "1904770", "1949363", "2008992", "2037362"}
)
#: The repaired events per frame-clock type over the corpus.
R2_PERIOD_RELABELLED: Mapping[str, int] = {"player_possession": 7, "passing_option": 48, "off_ball_run": 0}
#: The on_ball_engagement rows withheld from tackle enrichment for a period conflict, per match.
R2_OBE_PERIOD_CONFLICTS: Mapping[str, int] = {"2048846": 1, "2052202": 1}


class R2CensusDeviationError(RuntimeError):
    """The R.2 corpus deviates from the pre-registered GI period-label / Range census (spec section 26.8): the run
    stops and returns to the owner; no in-run fix."""


def r2_census_applies(dataset: object, max_matches: object) -> bool:
    """True on the full pre-registered R.2 dataset only (``dataset == R2_CENSUS_DATASET`` and no ``max_matches``
    cap)."""
    return str(dataset) == R2_CENSUS_DATASET and max_matches is None


def assert_r2_period_census(period_repair: pd.DataFrame, pairing: pd.DataFrame) -> None:
    """Raise ``R2CensusDeviationError`` naming every deviation from the spec section 26.8 pre-registration:
    ``period_repair`` = the per-match ``period_repair`` profile records, ``pairing`` = the :func:`gi_pairing_report`
    rows. Expected: 895 SkillCorner matches; ``repaired`` in exactly :data:`R2_PERIOD_REPAIRED_MATCHES` (all
    ``boundary_block``) and ``clean`` elsewhere; the relabel totals :data:`R2_PERIOD_RELABELLED`; the OBE conflicts
    :data:`R2_OBE_PERIOD_CONFLICTS`; no ``period_start_mismatch``; ``range_basis == "provider"`` and no Range
    violation everywhere; every match paired."""
    dev: list[str] = []
    pr = period_repair if not period_repair.empty else pd.DataFrame(columns=["match_id", "status"])
    ids = pr["match_id"].astype(str)
    if len(pr) != R2_SKILLCORNER_MATCHES:
        dev.append(f"{len(pr)} SkillCorner census records, expected {R2_SKILLCORNER_MATCHES}")
    status = dict(zip(ids, pr["status"].astype(str), strict=True))
    repaired = {m for m, s in status.items() if s == "repaired"}
    if repaired != set(R2_PERIOD_REPAIRED_MATCHES):
        dev.append(f"repaired matches {sorted(repaired)}, expected {sorted(R2_PERIOD_REPAIRED_MATCHES)}")
    other = sorted({s for m, s in status.items() if m not in R2_PERIOD_REPAIRED_MATCHES and s != "clean"})
    if other:
        dev.append(f"non-clean statuses outside the repaired set: {other}")
    if len(pr):
        totals = {t: int(pr[f"relabelled_{t}"].astype(int).sum()) for t in R2_PERIOD_RELABELLED}
        if totals != dict(R2_PERIOD_RELABELLED):
            dev.append(f"relabelled totals {totals}, expected {dict(R2_PERIOD_RELABELLED)}")
        shapes = sorted({str(s) for m, s in zip(ids, pr["shape"], strict=True) if m in repaired})
        if repaired and shapes != ["boundary_block"]:
            dev.append(f"repair shapes {shapes}, expected ['boundary_block']")
        obe = {m: int(n) for m, n in zip(ids, pr["obe_period_conflict"].astype(int), strict=True) if int(n)}
        if obe != dict(R2_OBE_PERIOD_CONFLICTS):
            dev.append(f"OBE period conflicts {obe}, expected {dict(R2_OBE_PERIOD_CONFLICTS)}")
        mism = sorted(m for m, v in zip(ids, pr["period_start_mismatch"].astype(bool), strict=True) if v)
        if mism:
            dev.append(f"match_periods start mismatches in {mism}, expected none")
    pa = pairing if not pairing.empty else pd.DataFrame(columns=["match_id", "paired", "range_basis"])
    if len(pa) != R2_SKILLCORNER_MATCHES:
        dev.append(f"{len(pa)} pairing rows, expected {R2_SKILLCORNER_MATCHES}")
    if len(pa):
        not_provider = sorted(pa.loc[pa["range_basis"].astype(str) != "provider", "match_id"].astype(str))
        if not_provider:
            dev.append(f"range_basis != 'provider' in {len(not_provider)} match(es): {not_provider[:10]}")
        violating = sorted(pa.loc[pa["n_range_violations"].astype(int) > 0, "match_id"].astype(str))
        if violating:
            dev.append(f"Range violations in {violating[:10]}")
        unpaired = sorted(pa.loc[~pa["paired"].astype(bool), "match_id"].astype(str))
        if unpaired:
            dev.append(f"{len(unpaired)} unpaired match(es): {unpaired[:10]}")
    if dev:
        raise R2CensusDeviationError(
            "R.2 deviates from the pre-registered GI period-label / Range census (spec section 26.8): " + "; ".join(dev)
        )


def derive_pairing_bar(negative_shares, *, quantile: float, margin: float) -> float:
    """The proximity bar rule (OD-6 amendment 2026-10-23, spec sections 6.1 / 16.1): the ``quantile`` (numpy
    ``linear`` method) of the finite negative-control shares plus ``margin``; NaN when there is no negative."""
    neg = pd.to_numeric(pd.Series(list(negative_shares), dtype="object"), errors="coerce").to_numpy("float64")
    neg = neg[np.isfinite(neg)]
    return float(np.quantile(neg, quantile, method="linear") + margin) if len(neg) else float("nan")


#: ``pairing_bar_table`` columns.
PAIRING_BAR_COLUMNS: tuple[str, ...] = (
    "data_profile",
    "variant",
    "prox_m",
    "margin",
    "n_negatives",
    "neg_quantile",
    "bar",
    "n_matches",
    "n_evaluable_matches",
    "n_proximity_pass",
    "n_paired",
    "n_known_paired",
    "n_known_paired_pass",
    "min_known_paired_margin",
)


def _verdicts_ok(pairing: pd.DataFrame, cols: Sequence[str]) -> dict[str, bool]:
    """match id -> every named verdict column is True (``<NA>`` / False -> False)."""
    if pairing.empty:
        return {}
    ok = np.ones(len(pairing), dtype=bool)
    for c in cols:
        ok &= pairing[c].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return dict(zip(pairing["match_id"].astype(str), ok, strict=True))


def pairing_bar_table(prox: pd.DataFrame, pairing: pd.DataFrame) -> pd.DataFrame:
    """Run (2) bar derivation per data profile (OD-6 amendment 2026-10-23). For the base variant and each
    sensitivity variant -- ``pair_prox_margin`` at its +/-50 % values, ``pair_prox_m`` at its +/-50 % values with the
    bar RE-DERIVED from the negatives at that distance -- one row: the bar from the time-shift negatives of the
    profile, the matches whose own share (shift 0) clears it, the matches that also pass the recorded range / lag /
    direction verdicts (on SkillCorner neither reads the distance: the lag source is ``sc_possession``), and the
    known-paired pass count with its minimum margin.

    ``prox``: ``data_profile``, ``match_id``, ``is_opendata``, ``prox_m``, ``shift_s``, ``prox_share`` (the
    ``pairing_proximity_profile`` records). ``pairing``: the :func:`gi_pairing_report` rows."""
    if prox.empty:
        return pd.DataFrame(columns=list(PAIRING_BAR_COLUMNS))
    q = float(RATIFIED["pair_neg_quantile"])
    base_m, base_margin = float(RATIFIED["pair_prox_m"]), float(RATIFIED["pair_prox_margin"])
    variants = [("base", base_m, base_margin)]
    variants += [(f"pair_prox_margin={v}", base_m, float(v)) for v in sensitivity_values("pair_prox_margin")]
    variants += [(f"pair_prox_m={v}", float(v), base_margin) for v in sensitivity_values("pair_prox_m")]
    other_ok = _verdicts_ok(pairing, ("pair_range_ok", "pair_lag_ok", "pair_direction_ok"))
    rows = []
    for pk, g in prox.groupby("data_profile", sort=True):
        for label, m, margin in variants:
            at_m = g[np.isclose(g["prox_m"].astype(float).to_numpy(), m)]
            shift = at_m["shift_s"].astype(float).to_numpy()
            share_all = at_m["prox_share"].astype(float).to_numpy()
            neg = share_all[(shift != 0.0) & np.isfinite(share_all)]
            bar = derive_pairing_bar(neg, quantile=q, margin=margin)
            pos = at_m[shift == 0.0]
            share = pos["prox_share"].astype(float).to_numpy()
            passes = (np.isfinite(share) & (share >= bar)) if np.isfinite(bar) else np.zeros(len(pos), dtype=bool)
            others = np.array([other_ok.get(str(mid), False) for mid in pos["match_id"]], dtype=bool)
            known = pos["is_opendata"].astype(bool).to_numpy()
            kshare = share[known]
            min_margin = float(np.nanmin(kshare - bar)) if known.any() and np.isfinite(bar) else float("nan")
            rows.append(
                {
                    "data_profile": pk,
                    "variant": label,
                    "prox_m": m,
                    "margin": margin,
                    "n_negatives": len(neg),
                    "neg_quantile": bar - margin if np.isfinite(bar) else float("nan"),
                    "bar": bar,
                    "n_matches": len(pos),
                    "n_evaluable_matches": int(np.isfinite(share).sum()),
                    "n_proximity_pass": int(passes.sum()),
                    "n_paired": int((passes & others).sum()),
                    "n_known_paired": int(known.sum()),
                    "n_known_paired_pass": int(passes[known].sum()),
                    "min_known_paired_margin": min_margin,
                }
            )
    return pd.DataFrame(rows, columns=list(PAIRING_BAR_COLUMNS))


def assert_pairing_bar_holds(bar_table: pd.DataFrame) -> None:
    """The per-profile guard (OD-6 amendment 2026-10-23): raise ``GiPairingBarError`` when the base-variant bar of
    a data profile (its own time-shift negatives: quantile + margin) exceeds the adopted ``pair_prox_share``, or
    when a profile with an evaluable match share has no negative control. Either returns the bar to the owner
    with that profile. A profile without any evaluable GI possession start has nothing to pair (every match fails
    proximity as not evaluable) and nothing to guard."""
    if bar_table.empty:
        return
    adopted = float(RATIFIED["pair_prox_share"])
    base = bar_table[bar_table["variant"] == "base"]
    bars = base["bar"].astype(float).to_numpy()
    quants = base["neg_quantile"].astype(float).to_numpy()
    n_neg = base["n_negatives"].astype(int).to_numpy()
    n_eval = base["n_evaluable_matches"].astype(int).to_numpy()
    bad = [
        f"{pk}: bar={b:.4f} (quantile {qv:.4f}, n={n})"
        for pk, b, qv, n, ev in zip(base["data_profile"], bars, quants, n_neg, n_eval, strict=True)
        if ev > 0 and (not np.isfinite(b) or b > adopted + 1e-12)
    ]
    if bad:
        raise GiPairingBarError(f"derived pairing bar exceeds the adopted pair_prox_share={adopted} on: {bad}")


def pairing_lag_table(pairing: pd.DataFrame) -> pd.DataFrame:
    """Run (2) alignment-offset distribution per data profile (second R.2 stop ruling 2026-10-23): one row per
    (``data_profile``, ``alignment_offset``) with its match count ``n``; unevaluable offsets are left out."""
    cols = ["data_profile", "alignment_offset", "n"]
    if pairing.empty or "alignment_offset" not in pairing.columns:
        return pd.DataFrame(columns=cols)
    off = pd.to_numeric(pairing["alignment_offset"], errors="coerce")
    ok = off.notna().to_numpy()
    sub = pd.DataFrame(
        {"data_profile": pairing["data_profile"].to_numpy()[ok], "alignment_offset": off[ok].astype(int)}
    )
    tab = sub.groupby(["data_profile", "alignment_offset"], sort=True).size().reset_index(name="n")
    return tab[cols]


def assert_pairing_lag_holds(pairing: pd.DataFrame) -> None:
    """The lag guard (second R.2 stop ruling 2026-10-23): raise ``GiPairingLagError`` when the median alignment
    offset of a data profile lies outside ``[-pair_lag_tol_frames, +pair_lag_tol_frames]`` -- a profile-wide
    desynchronisation returns the lag rule to the owner (the test is never re-centred on a measured offset)."""
    tab = pairing_lag_table(pairing)
    if tab.empty:
        return
    tol = int(RATIFIED["pair_lag_tol_frames"])
    bad = []
    for pk, g in tab.groupby("data_profile", sort=True):
        offsets = np.repeat(g["alignment_offset"].to_numpy("int64"), g["n"].to_numpy("int64"))
        med = float(np.median(offsets))
        if abs(med) > tol:
            bad.append(f"{pk}: median offset {med:+.1f} frames (n={len(offsets)})")
    if bad:
        raise GiPairingLagError(f"median alignment offset outside +/-{tol} frames on: {bad}")


class GiPairingLagError(RuntimeError):
    """The median GI / tracking alignment offset of a data profile lies outside the lag tolerance."""


class GiPairingBarError(RuntimeError):
    """The negative controls of a data profile derive a proximity bar above the adopted one (OD-6 amendment)."""


# ---------------------------------------------------------------------------
# Stoppage time s0 and the event phase (spec section 5.3, C-PLAN-24)
# ---------------------------------------------------------------------------

#: The three s0 sources, in resolution order (spec section 5.3, C-PLAN-24 ruling).
S0_SOURCE_VALUES: tuple[str, ...] = ("gi_bracket_lo", "ball_state_dead_run", "last_live_event")

#: Seconds the restart phase extends past the kick (spec section 5.3).
_RESTART_TAIL_S = 6.0


def stoppage_time(restarts: pd.DataFrame, frames: pd.DataFrame | None, events: pd.DataFrame) -> pd.DataFrame:
    """Per restart, the stoppage time ``s0`` and its source (spec section 5.3, C-PLAN-24 ruling).

    Resolution order: (i) a SkillCorner derived row -> ``kick_bracket_lo`` (``"gi_bracket_lo"``); (ii) a frame
    stream with a ``ball_state`` dead run ENDING at the kick -> its first ``"dead"`` frame
    (``"ball_state_dead_run"``); (iii) otherwise the time of the last live-play event before the restart
    (``"last_live_event"``). ``restarts`` carries ``restart_id``, ``period_id``, ``t_kick``,
    ``kick_bracket_lo`` and ``is_skillcorner_derived``. Columns: ``restart_id``, ``s0``, ``s0_source``.
    """
    cols = ["restart_id", "s0", "s0_source"]
    if restarts.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in restarts.iterrows():
        s0, src = _one_stoppage(r, frames, events)
        rows.append({"restart_id": r["restart_id"], "s0": s0, "s0_source": src})
    return pd.DataFrame(rows, columns=cols)


def _one_stoppage(r: pd.Series, frames: pd.DataFrame | None, events: pd.DataFrame) -> tuple[float, str]:
    if bool(r.get("is_skillcorner_derived", False)) and np.isfinite(float(r.get("kick_bracket_lo", np.nan))):
        return float(r["kick_bracket_lo"]), "gi_bracket_lo"
    t_kick = float(r["t_kick"])
    pid = r["period_id"]
    if frames is not None and not frames.empty and "ball_state" in frames.columns:
        dead = _dead_run_start_at_kick(frames, pid, t_kick)
        if dead is not None:
            return dead, "ball_state_dead_run"
    if events is not None and not events.empty:
        live = events[(events["period_id"] == pid) & (events["time_seconds"] < t_kick)]
        if "is_live" in live.columns:
            live = live[live["is_live"].astype(bool)]
        if not live.empty:
            return float(live["time_seconds"].max()), "last_live_event"
    return t_kick, "last_live_event"


def _dead_run_start_at_kick(frames: pd.DataFrame, period_id: Any, t_kick: float, *, tol_s: float = 0.2) -> float | None:
    """The first ``"dead"`` frame of the dead run that ENDS at the kick (spec section 5.3). An earlier dead run
    that ends before the kick is not used."""
    f = frames[(frames["period_id"] == period_id)].sort_values("time_seconds")
    if f.empty:
        return None
    t = f["time_seconds"].to_numpy(dtype=float)
    dead = (f["ball_state"].astype(str) == "dead").to_numpy()
    # the dead frames at or just before the kick
    before = t <= t_kick + tol_s
    if not (dead & before).any():
        return None
    last_idx = np.nonzero(dead & before)[0][-1]
    # the run must end at the kick: the next frame after last_idx is at/after the kick (or the kick is in the run)
    if last_idx + 1 < len(t) and t[last_idx + 1] < t_kick - tol_s:
        return None  # the dead run ended before the kick -> an earlier stoppage
    start = last_idx
    while start > 0 and dead[start - 1]:
        start -= 1
    return float(t[start])


def event_phase(frames_index: pd.DataFrame, restarts: pd.DataFrame) -> pd.Series:
    """Label each frame ``"restart"`` or ``"open"`` (spec section 5.3; one definition for fit and scoring).

    A frame is ``"restart"`` when it lies in ``[s0, t_kick + 6 s]`` of any restart in its period
    (``kick_bracket_hi`` in place of ``t_kick`` when ``t_kick`` is withheld). ``frames_index`` carries
    ``period_id`` and ``time_seconds``; ``restarts`` carries ``period_id``, ``s0``, ``t_kick`` and
    ``kick_bracket_hi``. Returned Series is index-aligned to ``frames_index``.
    """
    phase = pd.Series(["open"] * len(frames_index), index=frames_index.index, dtype="object")
    if frames_index.empty or restarts.empty:
        return phase
    ft = frames_index["time_seconds"].to_numpy(dtype=float)
    fp = frames_index["period_id"].to_numpy()
    is_restart = np.zeros(len(frames_index), dtype=bool)
    for _, r in restarts.iterrows():
        s0 = float(r["s0"])
        tk = float(r["t_kick"]) if np.isfinite(float(r.get("t_kick", np.nan))) else float(r["kick_bracket_hi"])
        hi = tk + _RESTART_TAIL_S
        is_restart |= (fp == r["period_id"]) & (ft >= s0) & (ft <= hi)
    phase[is_restart] = "restart"
    return phase


# ---------------------------------------------------------------------------
# Regions and the section 16.3 fidelity statistics
# ---------------------------------------------------------------------------

#: The four fidelity regions (spec section 16.3), in precedence order.
REGIONS: tuple[str, ...] = ("corner_flag", "touchline", "goalkick_box", "rest")
_PITCH_L, _PITCH_W = 105.0, 68.0
_PEN_HALF_W = 20.16  # penalty-area half width from centre (metres); the "goal-kick box" proxy


def region_labels(
    x: np.ndarray, y: np.ndarray, *, pitch_length: float = _PITCH_L, pitch_width: float = _PITCH_W
) -> np.ndarray:
    """Classify each (x, y) into one of :data:`REGIONS` (spec section 16.3), precedence corner_flag > touchline
    > goalkick_box > rest. Corner-flag zone: within 10 m of a corner; touchline band: within 3 m of a side
    line; goal-kick box: inside either penalty area. NaN positions -> ``"rest"``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.array(["rest"] * len(x), dtype=object)
    cx = np.minimum(x, pitch_length - x)  # distance to the nearer goal line
    cy = np.minimum(y, pitch_width - y)  # distance to the nearer side line
    corner = np.hypot(cx, cy) <= 10.0
    touch = (cy <= 3.0) & ~corner
    box = (cx <= 16.5) & (np.abs(y - pitch_width / 2) <= _PEN_HALF_W) & ~corner & ~touch
    out[box] = "goalkick_box"
    out[touch] = "touchline"
    out[corner] = "corner_flag"
    return out


def _detected(frames: pd.DataFrame) -> np.ndarray:
    """The carried source detection flag (spec section 17.7): a boolean ``detected`` column when present, else
    ``visibility == "detected"`` / truthy. An extrapolated or filled position never counts as detected."""
    if "detected" in frames.columns:
        return frames["detected"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    vis = frames["visibility"]
    if vis.dtype == bool or vis.dtype == "boolean":
        return vis.astype("boolean").fillna(False).to_numpy(dtype=bool)
    return (vis.astype("string") == "detected").fillna(False).to_numpy(dtype=bool)


def _filled(frames: pd.DataFrame) -> np.ndarray:
    """Ball frames that are filled or extrapolated, not detected (spec section 16.3 fill share)."""
    if "fill" in frames.columns:
        return frames["fill"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    if "visibility" in frames.columns and not (
        frames["visibility"].dtype == bool or frames["visibility"].dtype == "boolean"
    ):
        return frames["visibility"].astype("string").isin(["extrapolated", "filled"]).fillna(False).to_numpy(dtype=bool)
    return ~_detected(frames)


def _gap_lengths_s(times: np.ndarray, detected: np.ndarray) -> list[float]:
    """Lengths (seconds) of maximal runs of UNDETECTED frames in a time-sorted ball stream."""
    gaps: list[float] = []
    i = 0
    n = len(detected)
    while i < n:
        if detected[i]:
            i += 1
            continue
        j = i
        while j < n and not detected[j]:
            j += 1
        # the gap spans from the last detected frame to the next detected frame
        lo = times[i - 1] if i > 0 else times[i]
        hi = times[j] if j < n else times[j - 1]
        gaps.append(float(hi - lo))
        i = j
    return gaps


@dataclass(frozen=True)
class FidelityStat:
    """One measured statistic: ``region``, ``statistic``, a scalar ``value``, the ``n`` and an optional
    distribution ``samples`` (a list, for the KS / p90 statistics)."""

    region: str
    statistic: str
    value: float
    n: int
    samples: tuple[float, ...] | None = None


def measure_profile_stats(
    frames: pd.DataFrame, actions: pd.DataFrame, contacts: pd.DataFrame | None, *, regions: Sequence[str] = REGIONS
) -> pd.DataFrame:
    """Every section 16.3 statistic per region (C-02, C-PLAN-15).

    ``frames`` carries ball and player rows (``is_ball``, ``x``, ``y``, ``visibility`` / ``detected``,
    ``z_source``, ``time_seconds``, ``period_id``, ``player_id``); ``actions`` the restart kicks (``t_kick`` /
    ``K``); ``contacts`` the TF-79 contacts (nearest-player-at-contact). Returns one row per
    (region, statistic) with ``value``, ``n`` and, for the distribution statistics, a ``samples`` list. Ball
    detection, gaps and fill read the carried detection flag (C-PLAN-15): extrapolated positions never count.
    """
    ball = frames[frames["is_ball"].astype(bool)].sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    players = frames[~frames["is_ball"].astype(bool)]
    out: list[FidelityStat] = []
    # frame rate (global): the median positive dt of the ball stream
    dt = np.diff(ball["time_seconds"].to_numpy(dtype=float))
    dt = dt[dt > 0]
    out.append(FidelityStat("rest", "frame_rate", float(1.0 / np.median(dt)) if len(dt) else float("nan"), len(dt)))
    # z_source mix (global): share estimated
    if "z_source" in ball.columns and len(ball):
        est = float((ball["z_source"].astype("string") == "estimated").mean())
        out.append(FidelityStat("rest", "z_source_mix", est, len(ball)))
    breg = region_labels(ball["x"].to_numpy(), ball["y"].to_numpy()) if len(ball) else np.array([], dtype=object)
    bdet = _detected(ball) if len(ball) else np.array([], dtype=bool)
    bfill = _filled(ball) if len(ball) else np.array([], dtype=bool)
    bt = ball["time_seconds"].to_numpy(dtype=float) if len(ball) else np.array([])
    for reg in regions:
        m = breg == reg
        n = int(m.sum())
        det_rate = float(bdet[m].mean()) if n else float("nan")
        out.append(FidelityStat(reg, "ball_detection_rate", det_rate, n))
        out.append(FidelityStat(reg, "fill_share", float(bfill[m].mean()) if n else float("nan"), n))
        gaps = _gap_lengths_s(bt[m], bdet[m]) if n else []
        p90 = float(np.percentile(gaps, 90)) if gaps else float("nan")
        out.append(FidelityStat(reg, "gap_distribution", p90, len(gaps), samples=tuple(gaps)))
        sd = _rest_window_noise_sd(ball[m]) if n else float("nan")
        out.append(FidelityStat(reg, "noise_sd", sd, n))
        vis = _player_visibility_near_ball(players, ball[m]) if n else float("nan")
        out.append(FidelityStat(reg, "player_visibility_near_ball", vis, n))
    # kick-window miss rate (global)
    out.append(FidelityStat("rest", "kick_window_miss_rate", _kick_window_miss(ball, actions, bdet, bt), len(actions)))
    # nearest player visible at contact (global)
    if contacts is not None and not contacts.empty and "nearest_player_detected" in contacts.columns:
        share = float(contacts["nearest_player_detected"].astype("boolean").fillna(False).mean())
        out.append(FidelityStat("rest", "nearest_player_visible_at_contact", share, len(contacts)))
    # identity-swap proxy (global, per player-hour)
    out.append(FidelityStat("rest", "identity_swap_rate", *_identity_swap_rate(players)))
    return pd.DataFrame([vars(s) for s in out])


def _rest_window_noise_sd(ball_region: pd.DataFrame) -> float:
    """Position SD over stationary windows (>= 0.5 s at rest) of the region's ball stream (spec section 16.3)."""
    if len(ball_region) < 3:
        return float("nan")
    t = ball_region["time_seconds"].to_numpy(dtype=float)
    x = ball_region["x"].to_numpy(dtype=float)
    y = ball_region["y"].to_numpy(dtype=float)
    speed = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-6)
    at_rest = np.concatenate([[False], speed < 0.5])  # a slow ball row
    if at_rest.sum() < 3:
        return float("nan")
    return float(np.sqrt(np.nanvar(x[at_rest]) + np.nanvar(y[at_rest])))


def _player_visibility_near_ball(players: pd.DataFrame, ball_region: pd.DataFrame, *, radius_m: float = 10.0) -> float:
    """Share of player-frames detected within ``radius_m`` of the ball, over the region's ball frames."""
    if players.empty or ball_region.empty:
        return float("nan")
    ball_xy = ball_region.set_index(["period_id", "time_seconds"])[["x", "y"]]
    det = _detected(players)
    near = 0
    seen = 0
    pt = list(zip(players["period_id"], players["time_seconds"], players["x"], players["y"], det, strict=True))
    for pid, ts, px, py, d in pt:
        key = (pid, ts)
        if key not in ball_xy.index:
            continue
        bx, by = ball_xy.loc[key, "x"], ball_xy.loc[key, "y"]
        if np.hypot(float(px) - _f(bx), float(py) - _f(by)) <= radius_m:
            seen += 1
            near += int(bool(d))
    return (near / seen) if seen else float("nan")


def _kick_window_miss(
    ball: pd.DataFrame, actions: pd.DataFrame, bdet: np.ndarray, bt: np.ndarray, *, window_s: float = 0.3
) -> float:
    """Share of restart kicks with NO detected ball frame within +/- ``window_s`` of K (spec section 16.3)."""
    if actions is None or actions.empty or len(ball) == 0:
        return float("nan")
    kcol = "t_kick" if "t_kick" in actions.columns else "K" if "K" in actions.columns else None
    if kcol is None:
        return float("nan")
    bp = ball["period_id"].to_numpy()
    kicks = actions.dropna(subset=[kcol])
    if kicks.empty:
        return float("nan")
    misses = 0
    for _, k in kicks.iterrows():
        kt = float(k[kcol])
        near = (bp == k["period_id"]) & (np.abs(bt - kt) <= window_s) & bdet
        misses += int(not near.any())
    return misses / len(kicks)


def _identity_swap_rate(players: pd.DataFrame) -> tuple[float, int]:
    """Proxy identity-swap rate per player-hour: same-id jumps implying > 12 m/s between consecutive frames."""
    if players.empty:
        return float("nan"), 0
    jumps = 0
    hours = 0.0
    for _pid, g in players.groupby("player_id", sort=False):
        g = g.sort_values(["period_id", "time_seconds"])
        t = g["time_seconds"].to_numpy(dtype=float)
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dt = np.diff(t)
        speed = np.hypot(np.diff(x), np.diff(y)) / np.maximum(dt, 1e-6)
        jumps += int((speed > 12.0).sum())
        hours += float(np.clip(dt, 0, 5).sum()) / 3600.0
    return (jumps / hours if hours > 0 else float("nan")), round(hours * 3600)


def redetection_error_stats(frames: pd.DataFrame) -> pd.DataFrame:
    """Off-camera position error at re-detection, binned by time off camera (spec section 16.3, R2).

    Per player spell of ``visibility == False`` frames followed by a detected frame: the distance (m) between
    the last extrapolated position and the first detected position, binned (< 1 s, 1-3 s, 3-10 s, > 10 s).
    Spells that end at a period end are excluded and counted. Columns: ``bin``, ``n``, ``p90_m``,
    ``n_excluded_period_end``, ``samples``.
    """
    bins = ("<1s", "1-3s", "3-10s", ">10s")
    per_bin: dict[str, list[float]] = {b: [] for b in bins}
    excluded = 0
    if not frames.empty:
        players = frames[~frames.get("is_ball", pd.Series(False, index=frames.index)).astype(bool)]
        for _pid, g in players.groupby("player_id", sort=False):
            exc, samples = _spell_jumps(g)
            excluded += exc
            for length_s, jump in samples:
                per_bin[_offcam_bin(length_s)].append(jump)
    rows = []
    for b in bins:
        s = per_bin[b]
        rows.append(
            {
                "bin": b,
                "n": len(s),
                "p90_m": float(np.percentile(s, 90)) if s else float("nan"),
                "n_excluded_period_end": excluded if b == bins[0] else 0,
                "samples": tuple(s),
            }
        )
    return pd.DataFrame(rows, columns=["bin", "n", "p90_m", "n_excluded_period_end", "samples"])


def _offcam_bin(length_s: float) -> str:
    if length_s < 1.0:
        return "<1s"
    if length_s < 3.0:
        return "1-3s"
    if length_s <= 10.0:
        return "3-10s"
    return ">10s"


def _spell_jumps(g: pd.DataFrame) -> tuple[int, list[tuple[float, float]]]:
    """Per off-camera spell of one player: (excluded-at-period-end count, [(length_s, jump_m), ...])."""
    g = g.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    det = _detected(g)
    t = g["time_seconds"].to_numpy(dtype=float)
    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    period = g["period_id"].to_numpy()
    excluded = 0
    out: list[tuple[float, float]] = []
    i = 0
    n = len(g)
    while i < n:
        if det[i]:
            i += 1
            continue
        j = i
        while j < n and not det[j]:
            j += 1
        if j >= n or period[j] != period[i]:
            excluded += 1  # the spell ends at a period end (no re-detection in the same period)
            i = j
            continue
        length_s = float(t[j] - t[i - 1]) if i > 0 else float(t[j] - t[i])
        jump = float(np.hypot(x[j] - x[j - 1], y[j] - y[j - 1]))
        out.append((length_s, jump))
        i = j
    return excluded, out


def fidelity_report(degraded_stats: pd.DataFrame, target_stats: pd.DataFrame) -> pd.DataFrame:
    """Score every section 16.3 statistic per region against its tolerance (C-02, C-46).

    Joins the degraded and target measurements by (region, statistic) and applies the rule from
    :data:`FIDELITY_TOLERANCES` with the margin rule: margin = max(fixed value, 2 x match-cluster bootstrap SE
    of the target), capped at 2 x the fixed value; a statistic whose 2 x SE exceeds the cap FAILS
    CHARACTERISATION. Columns: ``region``, ``statistic``, ``degraded``, ``target``, ``passed``,
    ``failed_characterisation``.
    """
    cols = ["region", "statistic", "degraded", "target", "passed", "failed_characterisation"]
    if degraded_stats.empty or target_stats.empty:
        return pd.DataFrame(columns=cols)
    tgt = target_stats.set_index(["region", "statistic"])
    rows = []
    for _, d in degraded_stats.iterrows():
        key = (d["region"], d["statistic"])
        if key not in tgt.index:
            continue
        t = cast(pd.Series, tgt.loc[key])
        passed, failed_char = _fidelity_pass(d["statistic"], d, t)
        rows.append(
            {
                "region": d["region"],
                "statistic": d["statistic"],
                "degraded": float(d["value"]),
                "target": float(t["value"]),
                "passed": passed,
                "failed_characterisation": failed_char,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _fidelity_pass(stat: str, d: pd.Series, t: pd.Series) -> tuple[bool, bool]:
    """Apply one statistic's rule; returns (passed, failed_characterisation)."""
    dv, tv = float(d["value"]), float(t["value"])
    if not np.isfinite(dv) and not np.isfinite(tv):
        return True, False  # a region with no data either side: nothing to mismatch (vacuously passed)
    if stat in ("gap_distribution", "off_camera_position_error"):
        ks = _ks_distance(d.get("samples"), t.get("samples"))
        p90_ok = _within_rel(dv, tv, 0.20)
        return (np.isfinite(ks) and ks <= 0.10 and p90_ok), False
    if stat == "noise_sd":
        ratio = dv / tv if tv else float("nan")
        return (np.isfinite(ratio) and 0.85 <= ratio <= 1.15), False
    if stat == "frame_rate":
        return (np.isfinite(dv) and np.isfinite(tv) and abs(dv - tv) < 1e-6), False
    if stat == "z_source_mix":
        return (abs(dv - tv) < 1e-6), False
    # the absolute-difference statistics (detection rate, fill, visibility, kick-window miss, nearest at contact)
    tol = 0.05 if stat == "fill_share" else 0.03
    return (np.isfinite(dv) and np.isfinite(tv) and abs(dv - tv) <= tol), False


def _within_rel(a: float, b: float, rel: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return not np.isfinite(a) and not np.isfinite(b)
    return abs(a - b) / abs(b) <= rel


def _ks_distance(a: Any, b: Any) -> float:
    """Two-sample Kolmogorov-Smirnov distance between two sample tuples (NaN if either is empty)."""
    a = np.asarray(a if a is not None else [], dtype=float)
    b = np.asarray(b if b is not None else [], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    from scipy.stats import ks_2samp

    result: Any = ks_2samp(a, b)  # scipy's result type is untyped in the stubs
    return float(result.statistic)


# ---------------------------------------------------------------------------
# Degradation (spec section 5.3) and penalty semantics (B-05)
# ---------------------------------------------------------------------------


def degrade(frames: pd.DataFrame, profile_stats: Mapping[str, Any], *, seed: int) -> pd.DataFrame:
    """Degrade a full-observation (IDSSE / GS) tracking frame to a target SkillCorner profile (spec section 5.3).

    Reproduces the profile's frame rate, ball-detection gap-length distribution (per region), ball-position
    noise and fill behaviour. ``profile_stats`` carries the target ``frame_rate`` and, per region, the target
    ``gap_lengths`` (a sample) and ``noise_sd``. Seeded. The camera-follow visibility model and the off-camera
    position replay are applied by :func:`apply_camera_follow` / :func:`replay_offcamera_error` (they need the
    fitted models); this function reproduces the ball-level statistics and the gap structure. Pure (new frame).
    """
    rng = np.random.default_rng(seed)
    out = frames.copy()
    rate = float(profile_stats.get("frame_rate", 0.0))
    if rate > 0 and "time_seconds" in out.columns:
        out = _resample_to_rate(out, rate)
    out = _inject_ball_gaps(out, profile_stats, rng)
    out = _inject_ball_noise(out, profile_stats, rng)
    return out.reset_index(drop=True)


def _resample_to_rate(frames: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Keep the frames nearest a uniform grid at ``rate`` Hz, per period (spec section 16.3 frame-rate stat)."""
    keep_ids: list[Any] = []
    for (_pid), g in frames.groupby("period_id", sort=True):
        times = np.sort(g["time_seconds"].unique())
        if len(times) < 2:
            keep_ids.extend(g["frame_id"].unique().tolist())
            continue
        grid = np.arange(times.min(), times.max() + 1e-9, 1.0 / rate)
        chosen = {float(times[np.abs(times - gt).argmin()]) for gt in grid}
        keep_ids.extend(g[g["time_seconds"].isin(chosen)]["frame_id"].unique().tolist())
    return frames[frames["frame_id"].isin(set(keep_ids))].copy()


def _inject_ball_gaps(frames: pd.DataFrame, profile_stats: Mapping[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Flip ball-frame detection off in runs whose lengths match the target gap distribution, per region."""
    out = frames.copy()
    if "detected" not in out.columns:
        out["detected"] = ~_filled(out) if "visibility" in out.columns else True
    ball_mask = out["is_ball"].astype(bool).to_numpy()
    ball = out[ball_mask].sort_values(["period_id", "time_seconds"])
    target_gaps = {reg: list(profile_stats.get("gap_lengths", {}).get(reg, [])) for reg in REGIONS}
    det = ball["detected"].to_numpy(dtype=bool).copy()
    reg = region_labels(ball["x"].to_numpy(), ball["y"].to_numpy())
    t = ball["time_seconds"].to_numpy(dtype=float)
    for region, gaps in target_gaps.items():
        if not gaps:
            continue
        idx = np.nonzero(reg == region)[0]
        for g_len in gaps:
            if len(idx) == 0:
                break
            start_pos = int(rng.integers(0, len(idx)))
            start = idx[start_pos]
            end_t = t[start] + float(g_len)
            run = idx[(t[idx] >= t[start]) & (t[idx] <= end_t)]
            det[run] = False
    ball = ball.assign(detected=det)
    out.loc[ball.index, "detected"] = ball["detected"].to_numpy()
    return out


def _inject_ball_noise(
    frames: pd.DataFrame, profile_stats: Mapping[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    """Add Gaussian position noise to ball rows to match the target rest-window SD, per region."""
    out = frames.copy()
    ball_idx = out.index[out["is_ball"].astype(bool)]
    if len(ball_idx) == 0:
        return out
    reg = region_labels(out.loc[ball_idx, "x"].to_numpy(), out.loc[ball_idx, "y"].to_numpy())
    sd_by_region = profile_stats.get("noise_sd", {})
    for region in REGIONS:
        sd = float(sd_by_region.get(region, 0.0)) if isinstance(sd_by_region, Mapping) else 0.0
        if sd <= 0:
            continue
        sel = ball_idx[reg == region]
        out.loc[sel, "x"] = out.loc[sel, "x"].to_numpy() + rng.normal(0.0, sd / np.sqrt(2), size=len(sel))
        out.loc[sel, "y"] = out.loc[sel, "y"].to_numpy() + rng.normal(0.0, sd / np.sqrt(2), size=len(sel))
    return out


def simulate_skillcorner_penalty_semantics(native_events: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Build the SkillCorner goal-time semantics for native penalties (B-05, spec section 12).

    SkillCorner has NO event goal time: a goal is read from the SCORE written at the post-goal kick-off. This
    maps each native penalty to (i) ``goal_evidence`` = ``"score_at_kickoff"`` when the native event is a goal
    (the score increments at the next kick-off), (ii) post-goal ball handling (the ball returns to the centre
    spot), and (iii) a post-save rebound flag. Columns: ``restart_id``, ``outcome``, ``goal_evidence``,
    ``has_post_goal_kickoff``, ``has_rebound``. Run (12) scores the ``"attribution"`` cell on this.
    """
    cols = ["restart_id", "outcome", "goal_evidence", "has_post_goal_kickoff", "has_rebound"]
    if native_events.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, e in native_events.iterrows():
        outcome = str(e.get("outcome", "unknown"))
        is_goal = outcome == "goal"
        rows.append(
            {
                "restart_id": e["restart_id"],
                "outcome": outcome,
                "goal_evidence": "score_at_kickoff" if is_goal else "none",
                "has_post_goal_kickoff": bool(is_goal),
                "has_rebound": bool(e.get("rebound", False)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Camera-follow model (R3) and footprint chain (C-PLAN-22/-24)
# ---------------------------------------------------------------------------

#: The six seam kinds a stitched footprint chain can carry (spec section 5.3).
SEAM_KINDS: tuple[str, ...] = ("open-setup", "open-core", "setup-setup", "setup-core", "core-open", "open-open")

_BALL_BINS_X, _BALL_BINS_Y = 6, 4
_CORNER_COLS = ("cam_tl", "cam_tr", "cam_br", "cam_bl")


def _ball_bin(x: float, y: float, *, pitch_length: float = _PITCH_L, pitch_width: float = _PITCH_W) -> int:
    """The ball-position bin index in a 6 x 4 grid over the pitch (spec section 5.3). NaN -> bin 0."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return 0
    ix = int(np.clip(x / pitch_length * _BALL_BINS_X, 0, _BALL_BINS_X - 1))
    iy = int(np.clip(y / pitch_width * _BALL_BINS_Y, 0, _BALL_BINS_Y - 1))
    return iy * _BALL_BINS_X + ix


def _footprint_corners(row: pd.Series) -> np.ndarray:
    """The (4, 2) footprint corners of a footprint row (NaN when the span is a null-footprint replay)."""
    return np.array([[float(row[f"{c}_x"]), float(row[f"{c}_y"])] for c in _CORNER_COLS], dtype=float)


def _seq_corners(seq: np.ndarray) -> np.ndarray:
    """A stored sequence ``(T, 9)`` -> its ball-relative corners ``(T, 4, 2)`` (the first 8 columns)."""
    return np.asarray(seq, dtype=float)[:, :8].reshape(len(seq), 4, 2)


def _seq_krel(seq: np.ndarray) -> np.ndarray:
    """A stored sequence ``(T, 9)`` -> its per-frame kick-relative time (column 8)."""
    return np.asarray(seq, dtype=float)[:, 8]


@dataclass
class CameraFollowModel:
    """Parameters-only camera-follow model (ADR-011; spec section 5.3, R3).

    ``sequences`` maps ``(ball_bin, phase)`` to the EMPIRICAL list of footprint sequences relative to the ball,
    one per calibration event window; a sequence is an ``(T, 4, 2)`` array (NaN rows are null-footprint replay
    spans). ``pooling`` records, per cell, the neighbouring bins pooled into it (a cell with fewer than
    ``min_events`` sequences pools within its phase, never across). ``calibration_matches`` are the ids the fit
    was restricted to (never the held-out matches). ``seg_s`` is the open-play segment length.
    """

    sequences: dict[tuple[int, str], list[np.ndarray]] = field(default_factory=dict)
    pooling: dict[tuple[int, str], list[int]] = field(default_factory=dict)
    calibration_matches: tuple[str, ...] = ()
    seg_s: float = float(RATIFIED["cam_seg_s"])
    min_events: int = 30

    def cell(self, ball_bin: int, phase: str) -> list[np.ndarray]:
        """The sequences for ``(ball_bin, phase)`` after pooling (empty when the cell has none)."""
        return self.sequences.get((ball_bin, phase), [])

    def to_dict(self) -> dict:
        return {
            "sequences": {f"{b}|{p}": [s.tolist() for s in seqs] for (b, p), seqs in self.sequences.items()},
            "pooling": {f"{b}|{p}": v for (b, p), v in self.pooling.items()},
            "calibration_matches": list(self.calibration_matches),
            "seg_s": self.seg_s,
            "min_events": self.min_events,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CameraFollowModel:
        def _key(k: str) -> tuple[int, str]:
            b, p = k.split("|", 1)
            return int(b), p

        seqs = {_key(k): [np.asarray(s, dtype=float) for s in v] for k, v in d["sequences"].items()}
        pool = {_key(k): list(v) for k, v in d.get("pooling", {}).items()}
        return cls(
            seqs,
            pool,
            tuple(d.get("calibration_matches", ())),
            float(d.get("seg_s", RATIFIED["cam_seg_s"])),
            int(d.get("min_events", 30)),
        )


def fit_camera_follow(
    footprints: pd.DataFrame,
    ball: pd.DataFrame,
    phase: pd.Series,
    *,
    calibration_matches: Sequence[str],
    min_events: int = 30,
    seg_s: float = float(RATIFIED["cam_seg_s"]),
) -> CameraFollowModel:
    """Fit the camera-follow model on the CALIBRATION matches only (spec section 5.3, R3; C-46 fit/score split).

    ``footprints`` carries ``game_id`` / ``period_id`` / ``frame_id`` / ``time_seconds`` and the eight
    ``cam_*_x`` / ``cam_*_y`` corner columns; ``ball`` the per-frame ball ``x`` / ``y`` (index-aligned to
    ``footprints``); ``phase`` the per-frame ``"restart"`` / ``"open"`` label (index-aligned). Each window
    (an open-play ``seg_s`` segment, or a restart span) stores ONE footprint-relative-to-ball sequence under its
    first frame's ``(ball_bin, phase)``. Cells with fewer than ``min_events`` sequences pool neighbouring bins
    within the same phase (never across phases); the pooling is recorded.
    """
    cal = {str(m) for m in calibration_matches}
    fp = footprints.reset_index(drop=True)
    bl = ball.reset_index(drop=True)
    ph = phase.reset_index(drop=True)
    keep = fp["game_id"].astype(str).isin(cal).to_numpy()
    model = CameraFollowModel(seg_s=seg_s, min_events=min_events, calibration_matches=tuple(sorted(cal)))
    cells: dict[tuple[int, str], list[np.ndarray]] = {}
    for (_gid, _pid), gi in fp[keep].groupby(["game_id", "period_id"], sort=True):
        gi = gi.sort_values("time_seconds")
        for window in _fit_windows(gi, ph.to_numpy(), seg_s):
            first = window[0]
            bx, by = _f(bl.loc[first, "x"]), _f(bl.loc[first, "y"])
            b = _ball_bin(bx, by)
            p = str(ph.iloc[first])
            t0 = _f(fp.loc[first, "time_seconds"])
            rows = []
            for i in window:
                rel = _footprint_corners(cast(pd.Series, fp.loc[i])) - np.array(
                    [_f(bl.loc[i, "x"]), _f(bl.loc[i, "y"])]
                )
                ti = _f(fp.loc[i, "time_seconds"])
                # kick-relative time for a restart window (the driver supplies `kick_relative_s`); else from the
                # window start, so an open-play draw keeps its own internal timing.
                krel = (
                    _f(fp.loc[i, "kick_relative_s"])
                    if "kick_relative_s" in fp.columns and pd.notna(fp.loc[i, "kick_relative_s"])
                    else ti - t0
                )
                rows.append(np.concatenate([rel.reshape(8), [krel]]))
            cells.setdefault((b, p), []).append(np.stack(rows))
    model.sequences = _pool_cells(cells, min_events)
    model.pooling = _pooling_record(cells, min_events)
    return model


def _fit_windows(gi: pd.DataFrame, phase_all: np.ndarray, seg_s: float) -> list[list[int]]:
    """Split one (game, period) footprint block into windows: contiguous same-phase runs, open-play runs
    further chopped into ``seg_s`` segments. Returns lists of row indices (into the original frame)."""
    idx = gi.index.to_numpy()
    times = gi["time_seconds"].to_numpy(dtype=float)
    ph = phase_all[idx]
    windows: list[list[int]] = []
    i = 0
    n = len(idx)
    while i < n:
        j = i
        while j < n and ph[j] == ph[i]:
            j += 1
        run = list(range(i, j))
        if ph[i] == "open":
            seg_start = i
            while seg_start < j:
                t0 = times[seg_start]
                seg = [k for k in range(seg_start, j) if times[k] - t0 < seg_s]
                if not seg:
                    seg = [seg_start]
                windows.append([idx[k] for k in seg])
                seg_start = seg[-1] + 1
        else:
            windows.append([idx[k] for k in run])
        i = j
    return windows


def _bin_neighbours(b: int) -> list[int]:
    """The 4-neighbour bins of ball bin ``b`` in the 6 x 4 grid (within bounds)."""
    ix, iy = b % _BALL_BINS_X, b // _BALL_BINS_X
    out = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < _BALL_BINS_X and 0 <= ny < _BALL_BINS_Y:
            out.append(ny * _BALL_BINS_X + nx)
    return out


def _pool_cells(
    cells: Mapping[tuple[int, str], list[np.ndarray]], min_events: int
) -> dict[tuple[int, str], list[np.ndarray]]:
    """A cell with fewer than ``min_events`` sequences absorbs its within-phase neighbours' sequences."""
    out: dict[tuple[int, str], list[np.ndarray]] = {}
    for (b, p), seqs in cells.items():
        pooled = list(seqs)
        if len(seqs) < min_events:
            for nb in _bin_neighbours(b):
                pooled.extend(cells.get((nb, p), []))
        out[(b, p)] = pooled
    return out


def _pooling_record(
    cells: Mapping[tuple[int, str], list[np.ndarray]], min_events: int
) -> dict[tuple[int, str], list[int]]:
    """Per cell, the neighbour bins pooled in (empty when the cell did not pool)."""
    rec: dict[tuple[int, str], list[int]] = {}
    for (b, p), seqs in cells.items():
        rec[(b, p)] = [nb for nb in _bin_neighbours(b) if (nb, p) in cells] if len(seqs) < min_events else []
    return rec


@dataclass
class FootprintChain:
    """The per-frame footprint of a stitched chain plus its diagnostics (spec section 5.3).

    ``frame_id`` / ``footprint`` give the absolute footprint (``(N, 4, 2)``, NaN rows = null-footprint replay
    spans). ``seams`` maps a frame_id to its seam kind (one of :data:`SEAM_KINDS`). ``holds`` lists
    ``(frame_id, duration_s)`` edge holds. ``s0_source`` records each restart's stoppage source.
    """

    frame_id: np.ndarray
    footprint: np.ndarray
    seams: dict[Any, str] = field(default_factory=dict)
    holds: list[tuple[Any, float]] = field(default_factory=list)
    s0_source: dict[Any, str] = field(default_factory=dict)


def _choose_sequence(
    cells: list[np.ndarray], prev_last_rel: np.ndarray | None, stitch_k: int, rng: np.random.Generator
) -> np.ndarray | None:
    """Pick one sequence by the k-nearest-first-footprint seeded rule (spec section 5.3 stitching).

    From ``cells`` take the ``stitch_k`` whose first-frame ball-relative footprint is nearest (mean corner
    distance, m) to ``prev_last_rel``, then choose one at random. ``prev_last_rel is None`` (chain start) or
    ``stitch_k >= len(cells)`` (the non-vacuity control) picks uniformly at random among all.
    """
    if not cells:
        return None
    if prev_last_rel is None or stitch_k >= len(cells):
        return cells[int(rng.integers(len(cells)))]
    firsts = [_seq_corners(s)[0] for s in cells]
    dists = np.array(
        [float(np.nanmean(np.hypot(f[:, 0] - prev_last_rel[:, 0], f[:, 1] - prev_last_rel[:, 1]))) for f in firsts]
    )
    order = np.argsort(np.where(np.isfinite(dists), dists, np.inf))
    cand = [cells[i] for i in order[: max(1, stitch_k)]]
    return cand[int(rng.integers(len(cand)))]


def build_footprint_chain(
    model: CameraFollowModel,
    timeline: pd.DataFrame,
    restarts: pd.DataFrame,
    *,
    seed: int,
    seg_s: float = float(RATIFIED["cam_seg_s"]),
    stitch_k: int = int(RATIFIED["cam_stitch_k"]),
) -> FootprintChain:
    """Stitch a per-frame footprint chain for one match (spec section 5.3, C-PLAN-22 / -24; DL ruling 1).

    Open play is fixed ``seg_s`` segments; a restart window ``[s0, t_kick + 6 s]`` is a CORE
    ``[max(s0, t_kick - seg_s), t_kick + 6 s]`` (one restart sequence aligned on ``t_kick``, ``kick_bracket_hi``
    when withheld) plus SETUP segments filling ``[s0, core start)``. Each segment and the core join the previous
    by the ``stitch_k``-nearest seeded rule. Every seam is tagged with one of :data:`SEAM_KINDS`; an edge hold
    (a sequence shorter than its span) is recorded with its duration; ``s0_source`` is carried per restart.
    """
    rng = np.random.default_rng(seed)
    tl = timeline.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    frame_ids: list[Any] = []
    foot: list[np.ndarray] = []
    seams: dict[Any, str] = {}
    holds: list[tuple[Any, float]] = []
    s0_source: dict[Any, str] = {}
    for pid, g in tl.groupby("period_id", sort=True):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        rs = restarts[restarts["period_id"] == pid].copy() if not restarts.empty else restarts.iloc[0:0]
        spans = _period_spans(g, rs, seg_s)
        state = {"prev_last_rel": None, "prev_phase": None, "first": True}
        for span in spans:
            if span["kind_phase"] == "restart":
                s0_source[span["restart_id"]] = str(span.get("s0_source", "last_live_event"))
            _emit_segment(span, model, g, rng, stitch_k, frame_ids, foot, seams, holds, state)
    return FootprintChain(np.array(frame_ids, dtype=object), np.array(foot, dtype=float), seams, holds, s0_source)


def _period_spans(g: pd.DataFrame, restarts: pd.DataFrame, seg_s: float) -> list[dict]:
    """Ordered open / setup / core spans for one period (spec section 5.3). Each span: ``frames`` (row indices
    into ``g``), ``kind_phase`` (``open`` / ``restart``), ``seg_kind`` (``open`` / ``setup`` / ``core``), and,
    on a restart, ``t_kick`` / ``s0`` / ``restart_id`` / ``s0_source``."""
    t = g["time_seconds"].to_numpy(dtype=float)
    windows = []
    for _, r in restarts.iterrows():
        s0 = float(r["s0"])
        tk = float(r["t_kick"]) if np.isfinite(float(r.get("t_kick", np.nan))) else float(r["kick_bracket_hi"])
        windows.append((s0, tk + _RESTART_TAIL_S, tk, r.get("restart_id"), str(r.get("s0_source", "last_live_event"))))
    windows.sort()
    spans: list[dict] = []
    cursor = 0.0 if len(t) == 0 else t[0] - 1e-9
    for s0, hi, tk, rid, src in windows:
        if s0 > cursor:  # open play before the restart
            spans.extend(_open_segments(g, t, cursor, s0, seg_s))
        core_start = max(s0, tk - seg_s)
        if core_start > s0:  # setup segments fill [s0, core_start)
            spans.extend(_setup_segments(g, t, s0, core_start, seg_s, tk, rid, src))
        core_idx = [i for i in range(len(t)) if core_start <= t[i] <= hi]
        if core_idx:
            spans.append(
                {
                    "frames": core_idx,
                    "kind_phase": "restart",
                    "seg_kind": "core",
                    "t_kick": tk,
                    "s0": s0,
                    "restart_id": rid,
                    "s0_source": src,
                    "has_setup": core_start > s0,
                }
            )
        cursor = hi
    if cursor < (t[-1] if len(t) else cursor):
        spans.extend(_open_segments(g, t, cursor, t[-1] + 1e-9, seg_s))
    return spans


def _open_segments(g: pd.DataFrame, t: np.ndarray, lo: float, hi: float, seg_s: float) -> list[dict]:
    """Chop ``[lo, hi)`` open play into ``seg_s`` segments (spec section 5.3)."""
    out = []
    start = lo
    while start < hi:
        idx = [i for i in range(len(t)) if start <= t[i] < min(start + seg_s, hi)]
        if idx:
            out.append({"frames": idx, "kind_phase": "open", "seg_kind": "open"})
        start += seg_s
    return out


def _setup_segments(
    g: pd.DataFrame, t: np.ndarray, lo: float, hi: float, seg_s: float, tk: float, rid: Any, src: str
) -> list[dict]:
    """Chop the dead-ball setup ``[lo, hi)`` into ``seg_s`` restart-phase segments (spec section 5.3)."""
    out = []
    start = lo
    while start < hi:
        idx = [i for i in range(len(t)) if start <= t[i] < min(start + seg_s, hi)]
        if idx:
            out.append(
                {
                    "frames": idx,
                    "kind_phase": "restart",
                    "seg_kind": "setup",
                    "t_kick": tk,
                    "restart_id": rid,
                    "s0_source": src,
                }
            )
        start += seg_s
    return out


def _emit_segment(span, model, g, rng, stitch_k, frame_ids, foot, seams, holds, state) -> None:
    """Draw a sequence for one span and append its per-frame absolute footprint, recording the seam and hold."""
    idx = span["frames"]
    if not idx:
        return
    first = idx[0]
    bx0, by0 = float(g.loc[first, "x"]), float(g.loc[first, "y"])
    phase = span["kind_phase"]
    cells = model.cell(_ball_bin(bx0, by0), phase)
    seq = _choose_sequence(cells, state["prev_last_rel"], stitch_k, rng)
    seam = _seam_kind(span, state)
    if seam is not None:
        seams[g.loc[first, "frame_id"]] = seam
    assigned_last_rel = _assign_span(span, g, seq, frame_ids, foot, holds)
    if assigned_last_rel is not None:
        state["prev_last_rel"] = assigned_last_rel
    state["prev_phase"] = span["seg_kind"]
    state["first"] = False


def _seam_kind(span: dict, state: dict) -> str | None:
    """The seam kind at the start of ``span`` given the previous segment kind (spec section 5.3)."""
    if state["first"]:
        return None
    prev = state["prev_phase"]
    seg = span["seg_kind"]
    if seg == "open":
        return "core-open" if prev == "core" else "open-open"
    if seg == "setup":
        return "open-setup" if prev == "open" else "setup-setup"
    if seg == "core":
        return "setup-core" if prev == "setup" else "open-core"
    return "open-open"


def _assign_span(
    span: dict, g: pd.DataFrame, seq: np.ndarray | None, frame_ids: list, foot: list, holds: list
) -> np.ndarray | None:
    """Map a span's frames onto the drawn sequence (kick-aligned for a core, else index order), appending the
    absolute footprint per frame and recording an edge hold. Returns the last assigned ball-relative footprint."""
    idx = span["frames"]
    t = g["time_seconds"].to_numpy(dtype=float)
    last_rel = None
    if seq is None:  # no calibration sequence for this cell: a null footprint (every object off)
        for i in idx:
            frame_ids.append(g.loc[i, "frame_id"])
            foot.append(np.full((4, 2), np.nan))
        return None
    corners = _seq_corners(seq)
    krel = _seq_krel(seq)
    n_seq = len(corners)
    hold_from = None
    for k, i in enumerate(idx):
        if span["seg_kind"] == "core":
            target = t[i] - float(span["t_kick"])
            j = int(np.argmin(np.abs(krel - target)))
            if target > krel.max():  # past the sequence end -> hold its edge
                j = n_seq - 1
                hold_from = hold_from if hold_from is not None else i
        else:
            j = min(k, n_seq - 1)
            if k >= n_seq:
                hold_from = hold_from if hold_from is not None else i
        rel = corners[j]
        last_rel = rel
        bx, by = _f(g.loc[i, "x"]), _f(g.loc[i, "y"])
        foot.append(rel + np.array([bx, by]))
        frame_ids.append(g.loc[i, "frame_id"])
    if hold_from is not None:
        holds.append((g.loc[hold_from, "frame_id"], float(t[idx[-1]] - t[hold_from])))
    return last_rel


def apply_camera_follow(model: CameraFollowModel, frames: pd.DataFrame, chain: FootprintChain) -> pd.DataFrame:
    """Mask objects outside the sampled footprint at their frame (spec section 5.3; never a per-frame draw).

    An object outside its frame's chain footprint gets ``visibility = False``; a null-footprint span (a replay)
    sets every object in those frames to ``visibility = False``. Pure (new frame).
    """
    from silly_kicks.tracking._camera_footprint import in_camera_view_mask

    out = frames.copy()
    foot_by_frame = {fid: chain.footprint[i] for i, fid in enumerate(chain.frame_id)}
    vis = out["visibility"].copy() if "visibility" in out.columns else pd.Series(True, index=out.index)
    for i in out.index:
        fid = out.loc[i, "frame_id"]
        quad = foot_by_frame.get(fid)
        if quad is None:
            continue
        if np.isnan(quad).any():  # a replay / null-footprint span: every object is off camera
            vis.loc[i] = False
            continue
        if bool(out.loc[i].get("is_ball", False)):
            continue
        m = in_camera_view_mask([_f(out.loc[i, "x"])], [_f(out.loc[i, "y"])], quad[None, :, :], tol_m=0.0)
        if m[0] is not pd.NA and not bool(m[0]):
            vis.loc[i] = False
    out["visibility"] = vis
    return out


# ---------------------------------------------------------------------------
# Off-camera position replay (R2) and footprint gate (section 16.1)
# ---------------------------------------------------------------------------

_OFFCAM_BINS: tuple[str, ...] = ("<1s", "1-3s", "3-10s", ">10s")


@dataclass
class RedetectionModel:
    """Parameters-only off-camera re-detection error model (ADR-011; spec section 5.3, R2).

    Per time-off-camera bin, the empirical jump distances (m) and the jump angles relative to the player's last
    velocity (radians). Fitted on SkillCorner calibration matches.
    """

    jump_m: dict[str, list[float]] = field(default_factory=dict)
    angle_rad: dict[str, list[float]] = field(default_factory=dict)
    calibration_matches: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "jump_m": self.jump_m,
            "angle_rad": self.angle_rad,
            "calibration_matches": list(self.calibration_matches),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RedetectionModel:
        return cls(dict(d["jump_m"]), dict(d["angle_rad"]), tuple(d.get("calibration_matches", ())))


def fit_redetection_error(sc_frames: pd.DataFrame, *, calibration_matches: Sequence[str]) -> RedetectionModel:
    """Fit the off-camera re-detection error model on SkillCorner calibration matches (spec section 5.3, R2)."""
    cal = {str(m) for m in calibration_matches}
    jump: dict[str, list[float]] = {b: [] for b in _OFFCAM_BINS}
    angle: dict[str, list[float]] = {b: [] for b in _OFFCAM_BINS}
    sub = sc_frames[sc_frames["game_id"].astype(str).isin(cal)] if not sc_frames.empty else sc_frames
    players = sub[~sub.get("is_ball", pd.Series(False, index=sub.index)).astype(bool)] if not sub.empty else sub
    for _pid, gp in players.groupby("player_id", sort=False):
        for length_s, jmp, ang in _spell_jumps_with_angle(gp):
            b = _offcam_bin(length_s)
            jump[b].append(jmp)
            angle[b].append(ang)
    return RedetectionModel(jump, angle, tuple(sorted(cal)))


def _spell_jumps_with_angle(g: pd.DataFrame) -> list[tuple[float, float, float]]:
    """Per off-camera spell: (length_s, jump_m, angle relative to the last pre-spell velocity)."""
    g = g.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    det = _detected(g)
    t = g["time_seconds"].to_numpy(dtype=float)
    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    period = g["period_id"].to_numpy()
    out: list[tuple[float, float, float]] = []
    i = 0
    n = len(g)
    while i < n:
        if det[i]:
            i += 1
            continue
        j = i
        while j < n and not det[j]:
            j += 1
        if j >= n or period[j] != period[i] or i == 0:
            i = j
            continue
        length_s = float(t[j] - t[i - 1])
        jmp = float(np.hypot(x[j] - x[j - 1], y[j] - y[j - 1]))
        vang = np.arctan2(y[i - 1] - y[i - 2], x[i - 1] - x[i - 2]) if i >= 2 else 0.0
        jang = np.arctan2(y[j] - y[j - 1], x[j] - x[j - 1])
        out.append((length_s, jmp, float(((jang - vang + np.pi) % (2 * np.pi)) - np.pi)))
        i = j
    return out


def replay_offcamera_error(model: RedetectionModel, frames: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Displace off-camera spells by a sampled re-detection error (spec section 5.3, R2). Seeded.

    For each off-camera spell of length L, sample an end error ``e`` from L's bin and an angle; displace the
    position along a path growing from 0 at the spell start to ``e`` at the spell end. Pure (new frame).
    """
    rng = np.random.default_rng(seed)
    out = frames.copy()
    players_mask = ~out.get("is_ball", pd.Series(False, index=out.index)).astype(bool)
    for _pid, gp in out[players_mask].groupby("player_id", sort=False):
        gp = gp.sort_values(["period_id", "time_seconds"])
        det = _detected(gp)
        idx = gp.index.to_numpy()
        t = gp["time_seconds"].to_numpy(dtype=float)
        i = 0
        n = len(gp)
        while i < n:
            if det[i]:
                i += 1
                continue
            j = i
            while j < n and not det[j]:
                j += 1
            length_s = float(t[min(j, n - 1)] - t[i - 1]) if i > 0 else float(t[min(j, n - 1)] - t[i])
            b = _offcam_bin(length_s)
            samples = model.jump_m.get(b, [])
            angs = model.angle_rad.get(b, [])
            if samples:
                e = float(rng.choice(samples))
                ang = float(rng.choice(angs)) if angs else float(rng.uniform(-np.pi, np.pi))
                run = idx[i:j]
                span_t = t[i:j]
                frac = (
                    (span_t - span_t[0]) / max(span_t[-1] - span_t[0], 1e-6)
                    if len(span_t) > 1
                    else np.ones(len(span_t))
                )
                out.loc[run, "x"] = out.loc[run, "x"].to_numpy() + e * np.cos(ang) * frac
                out.loc[run, "y"] = out.loc[run, "y"].to_numpy() + e * np.sin(ang) * frac
            i = j
    return out


def footprint_profile_gate(frames: pd.DataFrame, footprint: pd.DataFrame, profile: DataProfile) -> dict:
    """The section 16.1 camera-footprint gate (CL-16, DL ruling C-PLAN-19).

    Passes when the detected-object inside-footprint share clears ``RATIFIED["footprint_inside_share"]``
    (0.99). A failing profile keeps the current approximation, marked ``visibility_model="approximation"``; a
    passing one uses ``"camera_follow"``. Returns the share, the pass flag, the model mark and the profile key.
    """
    from silly_kicks.tracking._camera_footprint import footprint_inside_share

    share = footprint_inside_share(frames, footprint, tol_m=float(RATIFIED["footprint_tol_m"]))
    passed = bool(np.isfinite(share) and share >= float(RATIFIED["footprint_inside_share"]))
    return {
        "data_profile": _profile_key(profile),
        "footprint_inside_share": float(share),
        "passed": passed,
        "visibility_model": "camera_follow" if passed else "approximation",
    }


__all__ = [
    "R2_CENSUS_DATASET",
    "R2_OBE_PERIOD_CONFLICTS",
    "R2_PERIOD_RELABELLED",
    "R2_PERIOD_REPAIRED_MATCHES",
    "R2_SKILLCORNER_MATCHES",
    "REGIONS",
    "RESEARCH_ROOT",
    "S0_SOURCE_VALUES",
    "SEAM_KINDS",
    "CameraFollowModel",
    "DataProfile",
    "FidelityStat",
    "FootprintChain",
    "GiPairingBarError",
    "GiPairingCalibrationError",
    "GiPairingLagError",
    "R2CensusDeviationError",
    "RedetectionModel",
    "apply_camera_follow",
    "assert_gi_direction_corpus",
    "assert_known_paired",
    "assert_pairing_bar_holds",
    "assert_pairing_lag_holds",
    "assert_r2_period_census",
    "build_footprint_chain",
    "coverage_report",
    "data_profile_for",
    "dedup_course_tier",
    "degrade",
    "derive_pairing_bar",
    "dfl_first_touch_truth",
    "event_phase",
    "fidelity_report",
    "fit_camera_follow",
    "fit_redetection_error",
    "footprint_profile_gate",
    "gi_pairing_report",
    "location_field_characterisation",
    "lomo_folds",
    "measure_profile_stats",
    "pairing_bar_table",
    "pairing_lag_table",
    "pff_truth",
    "r2_census_applies",
    "read_upstream",
    "redetection_error_stats",
    "region_labels",
    "replay_offcamera_error",
    "simulate_skillcorner_anchor",
    "simulate_skillcorner_penalty_semantics",
    "stoppage_time",
    "video_census",
    "write_run_aggregate",
]
