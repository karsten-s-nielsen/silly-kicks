"""W5.7 real-data role gate for the shape-graph position labels (spec 2026-11-08 tracking-batch section 4.5 W5.7,
owner ruling R15; plan Task 11A).

Every match of the public SkillCorner set (``scripts._corpus.RAW_REDISTRIBUTABLE["skillcorner"]``, 20 matches), periods
1-2, every 10th frame. Three arms per match:

- final: ``compute_shape_graphs(frames).players`` on the ``"ok"`` team-frames;
- baseline: the 4.139 shape graph (``scripts/_shape_graph_v4139_baseline.py``) per team-frame, with the direction
  from the frame label (the handoff's variant B);
- mirror: ``compute_shape_graphs(reflect_frames(frames))``, which must give the final labels exactly (rule 3).

The truth is ``match.json`` ``players[].player_role.acronym`` through the committed table ``ROLE_VERTICAL_GROUP``.
A scored player's modal labels are compared with it, pooled over the 20 matches, in four cells (vertical / lateral x
home / away). The pre-registered gate (rules 1-3) is :func:`gate`.

A LOCAL-ONLY pipeline (owner ruling 2026-11-11): it reads pining and runs on the DGX (spec R22), so it has no HF
registry entry (``tests/scripts/test_compute_contract.py`` ``_LOCAL_ONLY``) and is never launched on HF.

    python -m scripts.validate_shape_graph_roles run --out docs/research/shape-graph-positions

exits 1 when the gate fails.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import pathlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from scripts._corpus import RAW_REDISTRIBUTABLE
from scripts._frame_reflection import reflect_frames
from scripts._label_path_flags import label_flag_spies
from scripts._provenance import CODE_IDENTITY_PATHS, git_provenance, git_tree_hash, require_clean_tree
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, done_results
from scripts._stages import main as stages_main

REPO = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = "skillcorner"
PERIODS = (1, 2)
SAMPLE_EVERY = 10  # the handoff's sampling: frame_id % 10 == 0 right after the load
NULL_SEED = 20261108
N_DRAWS = 10_000
TEAM_FRAME_KEYS = ["game_id", "period_id", "frame_id", "team_id"]
CELLS = ("vertical_home", "vertical_away", "lateral_home", "lateral_away")

#: The committed truth table (plan Task 11A step 1): every outfield acronym of the 20 public rosters, with the
#: handoff's back and forward sets (``repro_shape_graph_roles.py:91-92``); every other outfield acronym is "mid".
#: GK and SUB are outside the scored population. An acronym outside the table raises (fail-closed, AG4).
ROLE_VERTICAL_GROUP: Mapping[str, str] = {
    # back (the handoff's set)
    "LB": "back",
    "RB": "back",
    "CB": "back",
    "LCB": "back",
    "RCB": "back",
    "LWB": "back",
    "RWB": "back",
    # forward (the handoff's set; ST, LCF and RCF are not in the 2026-11-11 census)
    "CF": "fwd",
    "LW": "fwd",
    "RW": "fwd",
    "LF": "fwd",
    "RF": "fwd",
    "ST": "fwd",
    "LCF": "fwd",
    "RCF": "fwd",
    # mid (every other outfield acronym of the census, listed explicitly)
    "AM": "mid",
    "DM": "mid",
    "LDM": "mid",
    "RDM": "mid",
    "LM": "mid",
    "RM": "mid",
}

_VERTICAL_GROUP = {"B": "back", "DM": "mid", "M": "mid", "AM": "mid", "F": "fwd"}
_SIDE = {"L": "L", "LC": "L", "C": "C", "RC": "R", "R": "R"}
_DIRECTION = {"ltr": 1.0, "rtl": -1.0}


# --------------------------------------------------------------------------------------------------------------------
# Truth, population, modal labels
# --------------------------------------------------------------------------------------------------------------------


def role_group(acronym: str) -> str:
    """The vertical group of a role acronym; ``ValueError`` for one outside the committed table."""
    if acronym not in ROLE_VERTICAL_GROUP:
        raise ValueError(f"unknown SkillCorner role acronym {acronym!r}: add it to ROLE_VERTICAL_GROUP")
    return ROLE_VERTICAL_GROUP[acronym]


def role_side(acronym: str) -> str | None:
    """``"L"`` / ``"R"`` from the acronym's first letter; ``None`` (not scored for side) otherwise."""
    return "L" if acronym.startswith("L") else "R" if acronym.startswith("R") else None


def scored_population(roster: Sequence[Mapping[str, Any]], *, labelled: set[str]) -> frozenset[str]:
    """The spec predicate (A7-SPEC-86): role not GK and not SUB, a non-null ``start_time``, and at least one
    labelled outfield row (``labelled``: player ids as strings)."""
    out = set()
    for p in roster:
        acronym = (p.get("player_role") or {}).get("acronym")
        pid = str(p["id"])
        if acronym in ("GK", "SUB") or p.get("start_time") is None or pid not in labelled:
            continue
        out.add(pid)
    return frozenset(out)


def _mode(values: Sequence[str]) -> str:
    """The most common value; a tie goes to the value seen first (``Counter.most_common``, the handoff's rule)."""
    return collections.Counter(values).most_common(1)[0][0]


def is_tied(values: Sequence[str]) -> bool:
    """True when the top two values have the same count."""
    top = collections.Counter(values).most_common(2)
    return len(top) == 2 and top[0][1] == top[1][1]


def modal_vertical(labels: Sequence[str]) -> str:
    """The mode of the five-level vertical label, then its group (TB-SPEC-54)."""
    return _VERTICAL_GROUP[_mode(labels)]


def modal_side(labels: Sequence[str]) -> str:
    """Each lateral label to its side first (L, LC -> L; R, RC -> R; C -> C), then the mode."""
    return _mode([_SIDE[h] for h in labels])


# --------------------------------------------------------------------------------------------------------------------
# The null, the interval, the gate
# --------------------------------------------------------------------------------------------------------------------


def null_agreements(scored: pd.DataFrame, *, n_draws: int = N_DRAWS, seed: int = NULL_SEED) -> np.ndarray:
    """The pooled agreement of ``n_draws`` draws, each shuffling ``truth`` within every (match_id, team_id) block
    (``rng.permutation`` per block; the predictions do not move)."""
    truth = scored["truth"].to_numpy(dtype=object)
    pred = scored["pred"].to_numpy(dtype=object)
    block_key = scored["match_id"].astype(str) + "\x1f" + scored["team_id"].astype(str)
    blocks = [np.flatnonzero(block_key.to_numpy() == b) for b in sorted(set(block_key))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    shuffled = truth.copy()
    for d in range(n_draws):
        for idx in blocks:
            shuffled[idx] = truth[idx][rng.permutation(len(idx))]
        out[d] = float(np.mean(shuffled == pred))
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson 95 % interval of ``k`` successes in ``n``."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def block_permutation_null(scored: pd.DataFrame, *, n_draws: int = N_DRAWS, seed: int = NULL_SEED) -> dict:
    """Rule 1's statistic for one cell (``scored``: ``match_id``, ``team_id``, ``truth``, ``pred``): the observed
    pooled agreement, its Wilson interval, the null's 95th percentile, and ``p = (1 + #{null >= obs}) / (1 +
    draws)``."""
    n = len(scored)
    k = int(np.sum(scored["truth"].to_numpy(dtype=object) == scored["pred"].to_numpy(dtype=object)))
    obs = k / n if n else math.nan
    null = null_agreements(scored, n_draws=n_draws, seed=seed) if n else np.full(n_draws, math.nan)
    low, high = wilson(k, n)
    return {
        "agreement": obs,
        "n": n,
        "null_p95": float(np.percentile(null, 95)) if n else math.nan,
        "p": (1 + int(np.sum(null >= obs))) / (1 + n_draws) if n else math.nan,
        "wilson_low": low,
        "wilson_high": high,
        "n_draws": n_draws,
        "seed": seed,
    }


def gate(metrics: Mapping[str, Any]) -> list[str]:
    """The failed pre-registered rules (spec W5.7 rules 1-3); empty on a pass.

    1. each cell's agreement is above its null's 95th percentile;
    2. each cell's final agreement is at least the baseline's;
    3. no scored team-frame's final labels differ from its mirror's.
    """
    failed = []
    for cell in CELLS:
        c = metrics["cells"][cell]
        if not c["agreement"] > c["null_p95"]:
            failed.append(f"rule1_beats_null:{cell}")
    for cell in CELLS:
        c = metrics["cells"][cell]
        if not c["agreement"] >= c["baseline_agreement"]:
            failed.append(f"rule2_no_regression:{cell}")
    if metrics["mirror"]["n_differing_team_frames"] != 0:
        failed.append("rule3_mirror_invariance")
    return failed


# --------------------------------------------------------------------------------------------------------------------
# The arms
# --------------------------------------------------------------------------------------------------------------------


def _team_frames(frames: pd.DataFrame):
    """``(key, rows, positions, direction)`` per team-frame of the shape-graph selection, in first-appearance
    order; ``direction`` is +1.0 / -1.0 from the first row's label, else None."""
    from silly_kicks.tracking._shape_graph import _outfield_selection

    sel = _outfield_selection(frames)
    for key, rows in sel.groupby(TEAM_FRAME_KEYS, sort=False, observed=True):
        xy = rows[["x", "y"]].to_numpy(dtype=np.float64)
        label = rows["team_attacking_direction"].iloc[0] if "team_attacking_direction" in rows.columns else None
        yield key, rows, xy, _DIRECTION.get(label) if isinstance(label, str) else None


def baseline_labels(frames: pd.DataFrame) -> pd.DataFrame:
    """The baseline arm: the 4.139 shape graph per team-frame, direction from the frame label (variant B). A
    team-frame with no direction, fewer than 3 players or no edges gets no label (``None``)."""
    from scripts import _shape_graph_v4139_baseline as baseline

    out = []
    for key, rows, xy, direction in _team_frames(frames):
        labels: list = [None] * len(rows)
        if direction is not None and len(xy) >= 3:
            sg = baseline.compute_shape_graph(xy)
            if len(sg.edges):
                labels = list(baseline.infer_positions(sg, xy, direction))
        for pid, lab in zip(rows["player_id"], labels, strict=True):
            out.append((*key, pid, None if lab is None else lab.vertical, None if lab is None else lab.horizontal))
    return pd.DataFrame(out, columns=[*TEAM_FRAME_KEYS, "player_id", "vertical", "horizontal"])


def mirror_report(plain: pd.DataFrame, mirrored: pd.DataFrame, *, faces_differ: Callable[[tuple], bool]) -> list[dict]:
    """The team-frames whose labels differ between the frames and their reflection (rule 3), in row order, each
    with its key and whether the two graphs' face sets differ (the known near-cocircular class)."""
    cols = ["vertical", "horizontal", "position_label"]
    same = (plain[cols].fillna("<NA>").to_numpy() == mirrored[cols].fillna("<NA>").to_numpy()).all(axis=1)
    out = []
    seen = set()
    for i in np.flatnonzero(~same):
        key = tuple(plain[TEAM_FRAME_KEYS].iloc[int(i)].tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": [_plain(v) for v in key], "faces_differ": bool(faces_differ(key))})
    return out


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


# --------------------------------------------------------------------------------------------------------------------
# One match (the ItemStage unit)
# --------------------------------------------------------------------------------------------------------------------


@functools.cache
def _manifest() -> dict[str, dict]:
    from scripts import _loader_pining as lp

    return {str(m["id"]): m for m in lp._list_matches(PROVIDER, lp._resolve_token(None), lp._base_url())}


def _refs(ctx) -> list[str]:
    ids = sorted(str(m) for m in RAW_REDISTRIBUTABLE[PROVIDER])
    k = ctx.params["max_matches"]
    return ids if k is None else ids[:k]


def _load(match_id: str, ctx):
    """Resume-before-load (ADR-052): the artifacts of one match through pining, the native frames (no velocity),
    and its ``match.json``."""
    from scripts import _loader_pining as lp
    from scripts._driver_sources import resolve_cache_dir

    root = resolve_cache_dir(ctx.params["cache_dir"])
    if root is None:
        raise ValueError("validate_shape_graph_roles: pass --cache-dir (or set SILLY_KICKS_CORPUS_CACHE_DIR)")
    dest = pathlib.Path(root) / PROVIDER / match_id
    dest.mkdir(parents=True, exist_ok=True)
    tok, base = lp._resolve_token(None), lp._base_url()
    paths = lp._download_artifacts(
        PROVIDER, match_id, _manifest()[match_id]["artifacts"], tok, base, dest, use_cache=True
    )
    frames, _report = lp.build_skillcorner_frames(paths, match_id, None, with_velocity=False)
    return match_id, frames, lp._read_json(paths["metadata"])


def _sample(frames: pd.DataFrame) -> pd.DataFrame:
    keep = frames["period_id"].isin(PERIODS).to_numpy() & (frames["frame_id"].to_numpy() % SAMPLE_EVERY == 0)
    return frames[keep].reset_index(drop=True)


def _rows(kind: str, records: list[dict]) -> list[dict]:
    return [{"kind": kind, "payload": json.dumps(r, sort_keys=True, default=_plain)} for r in records]


def _work(item, ctx) -> pd.DataFrame:
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.tracking import compute_shape_graph, compute_shape_graphs, infer_positions

    match_id, frames, meta = item
    frames = _sample(frames)
    home = str(meta["home_team"]["id"])
    roster = meta["players"]
    by_id = {str(p["id"]): p for p in roster}

    final_batch = compute_shape_graphs(frames)
    mirror_batch = compute_shape_graphs(reflect_frames(frames))
    final = final_batch.players
    ok = final[final["shape_graph_source"] == "ok"]
    base = baseline_labels(frames)

    def side_of(team_id) -> str:
        return "home" if str(canonical_id(team_id)) == home else "away"

    def faces_differ(key) -> bool:
        return set(final_batch.graph(*key).faces) != set(mirror_batch.graph(*key).faces)

    mirror = mirror_report(ok, mirror_batch.players.loc[ok.index], faces_differ=faces_differ)

    labelled = {str(canonical_id(p)) for p in ok["player_id"]}
    scored = scored_population(roster, labelled=labelled)
    players = []
    final_pid = ok["player_id"].map(lambda p: str(canonical_id(p)))
    base_ok = base[base["vertical"].notna()]
    base_pid = base_ok["player_id"].map(lambda p: str(canonical_id(p)))
    for pid in sorted(scored):
        acronym = by_id[pid]["player_role"]["acronym"]
        mine = ok[final_pid.to_numpy() == pid]
        theirs = base_ok[base_pid.to_numpy() == pid]
        verticals, horizontals = mine["vertical"].tolist(), mine["horizontal"].tolist()
        players.append(
            {
                "match_id": match_id,
                "team_id": str(by_id[pid]["team_id"]),
                "side": "home" if str(by_id[pid]["team_id"]) == home else "away",
                "player_id": pid,
                "acronym": acronym,
                "truth_vertical": role_group(acronym),
                "truth_side": role_side(acronym),
                "final_vertical_label": _mode(verticals),
                "final_vertical": modal_vertical(verticals),
                "final_side": modal_side(horizontals),
                "base_vertical": modal_vertical(theirs["vertical"].tolist()) if len(theirs) else None,
                "base_side": modal_side(theirs["horizontal"].tolist()) if len(theirs) else None,
                "tied": is_tied(verticals) or is_tied([_SIDE[h] for h in horizontals]),
            }
        )

    # diagnostics: the label-path classes per side (W0 counts the same flags)
    diag: dict[tuple[str, str], int] = collections.Counter()
    for key, _rows_tf, xy, direction in _team_frames(frames):
        if direction is None or len(xy) < 3:
            continue
        sg = compute_shape_graph(xy)
        if not len(sg.edges):
            continue
        with label_flag_spies() as fl:
            infer_positions(sg, xy, direction)
        side = side_of(key[3])
        diag[(side, "n_team_frames")] += 1
        diag[(side, "pitch_thirds")] += bool(fl.get("pitch_thirds"))
        diag[(side, "lateral_fallback")] += bool(fl.get("lateral_fallback"))
        diag[(side, "lateral_fallback_odd_n")] += bool(fl.get("lateral_fallback")) and len(xy) not in (5, 10)
    diag_records = [{"side": s, "metric": m, "value": int(v)} for (s, m), v in sorted(diag.items())]
    return pd.DataFrame(
        _rows("player", players)
        + _rows("mirror", [{"match_id": match_id, **m} for m in mirror])
        + _rows("diag", diag_records),
        columns=["kind", "payload"],
    )


# --------------------------------------------------------------------------------------------------------------------
# The reduce: the null, the gate, metrics.json
# --------------------------------------------------------------------------------------------------------------------


def _records(results: Mapping[str, pd.DataFrame], kind: str) -> list[dict]:
    out = []
    for key in sorted(results):
        df = results[key]
        out += [json.loads(p) for p in df.loc[df["kind"] == kind, "payload"]]
    return out


def _cell_frame(players: pd.DataFrame, cell: str, arm: str) -> pd.DataFrame:
    axis, side = cell.split("_")
    p = players[players["side"] == side]
    truth_col, pred_col = ("truth_vertical", f"{arm}_vertical") if axis == "vertical" else ("truth_side", f"{arm}_side")
    p = p[p[truth_col].notna() & p[pred_col].notna()]
    return pd.DataFrame(
        {"match_id": p["match_id"], "team_id": p["team_id"], "truth": p[truth_col], "pred": p[pred_col]}
    ).reset_index(drop=True)


def _report_work(_key, ctx) -> FileSet:
    prov = require_clean_tree(git_provenance(), allow_dirty=True)
    results = done_results(ctx, "match")
    players = pd.DataFrame(_records(results, "player"))
    mirror = _records(results, "mirror")
    diag = _records(results, "diag")
    n_draws = ctx.params["n_draws"]
    cells = {}
    for cell in CELLS:
        stats = block_permutation_null(_cell_frame(players, cell, "final"), n_draws=n_draws, seed=NULL_SEED)
        base = _cell_frame(players, cell, "base")
        stats["baseline_agreement"] = float(np.mean(base["truth"] == base["pred"])) if len(base) else math.nan
        stats["baseline_n"] = len(base)
        cells[cell] = stats
    diagnostics: dict[str, dict] = {}
    for rec in diag:
        side = diagnostics.setdefault(rec["side"], {})
        side[rec["metric"]] = side.get(rec["metric"], 0) + rec["value"]
    for side in ("home", "away"):
        part = players[players["side"] == side]
        diagnostics.setdefault(side, {})["confusion"] = {
            acr: dict(sorted(collections.Counter(grp["final_vertical_label"]).items()))
            for acr, grp in part.groupby("acronym", sort=True)
        }
        diagnostics[side]["tied_players"] = int(part["tied"].sum())
    metrics = {
        "cells": cells,
        "mirror": {"n_differing_team_frames": len(mirror), "differing": mirror},
        "diagnostics": diagnostics,
        "n_matches": len(results),
        "n_scored_players": len(players),
        "null": {"seed": NULL_SEED, "n_draws": n_draws},
        "run_commit": prov["commit"],
        "run_tree_dirty": bool(prov["dirty"]),
        "code_identity": git_tree_hash(paths=CODE_IDENTITY_PATHS, repo=REPO),
    }
    metrics["failed_rules"] = gate(metrics)
    metrics["passed"] = not metrics["failed_rules"]
    return FileSet({"metrics.json": (json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n").encode()})


PIPELINE = Pipeline(
    name="validate_shape_graph_roles",
    module="scripts.validate_shape_graph_roles",
    params=(
        Param("cache_dir", default=None, help="the pining artifact cache root"),
        Param("max_matches", type=int, default=None, help="the first k of the 20 sorted match ids (toy/debug)"),
        Param("n_draws", type=int, default=N_DRAWS, help="the permutation-null draws (spec: 10,000)"),
    ),
    stages=(
        ItemStage(
            name="match",
            refs=_refs,
            key=lambda ref: ref,
            load=_load,
            work=_work,
            params=("max_matches",),
            declare=lambda p: {"metric": "shape_graph_roles", "schema": 1},
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("match",),
            reduce=lambda r, ctx: r["only"],
            params=("n_draws",),
        ),
    ),
    output="report",
)


def _run_main(argv: Sequence[str]) -> int:
    """``run --out <dir>``: the whole DAG into ``<dir>/_store``, ``metrics.json`` exported to ``<dir>``; exit 1 when
    the gate fails."""
    ap = argparse.ArgumentParser(prog="validate_shape_graph_roles run", add_help=False)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    known, rest = ap.parse_known_args(argv)
    known.out.mkdir(parents=True, exist_ok=True)
    rc = stages_main(
        PIPELINE,
        [*rest, "--run-all", "--allow-dirty", "--store", str(known.out / "_store"), "--out", str(known.out)],
    )
    if rc:
        return rc
    metrics = json.loads((known.out / "metrics.json").read_text(encoding="utf-8"))
    return 0 if metrics["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        return _run_main(argv[1:])
    return stages_main(PIPELINE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
