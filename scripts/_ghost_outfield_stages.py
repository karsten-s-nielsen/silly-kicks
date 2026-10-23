"""ADR-129: the reusable `extract` stage of `scripts.train_ghost_outfield`; its closure must not contain
the trainer (`test_stage_modules`). The fit-side stages (cv/fit/assemble) stay in the trainer and import
`feature_set_for_variant` from here."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Literal

import pandas as pd

from scripts._stages import ItemStage

_FeatureSet = Literal["faithful", "position_only"]


def feature_set_for_variant(variant: str) -> _FeatureSet:
    """Map the CLI ``--variant`` onto the model feature set."""
    return "position_only" if variant == "position_only" else "faithful"


def extraction_inputs(variant: str, subsample_fps: float | None = 1.0) -> dict:
    """DECLARED shard-generation inputs (the digest ``for_each`` keys shards on).

    ``feature_set`` MUST be here (the 4.77.1 stale-shard rule): a position_only run drops the 4
    velocity features, so its shards carry a different column set and are NOT interchangeable with a
    faithful run's. ``subsample_fps`` MUST be here too: it changes the ROW SET (fewer, temporally
    thinned frames), so a 1 fps shard is not interchangeable with a 25 fps one. Bumping the
    extractor's output schema means bumping ``extractor`` here.
    """
    return {
        "schema": "ghost-outfield-1",
        "feature_set": feature_set_for_variant(variant),
        # v2 = possession-conditioned extraction (BOTH teams per frame, live team_in_possession); a v1
        # in-possession-only shard is NOT interchangeable, so the token change forces a fresh generation.
        "extractor": "ghost_outfield_v2",
        "subsample_fps": subsample_fps,
    }


def _subsample_frames(frames: pd.DataFrame, subsample_fps: float | None) -> pd.DataFrame:
    """Keep ~``subsample_fps`` frames/sec per ``(game_id, period_id)`` --- a VERBATIM mirror of the
    ghost-GK trainer's subsampler (``prepare_ghost_gk_training_data``).

    Tracking is ~25 fps and consecutive frames are near-duplicates, so a league-average positioning
    model trains on 1 fps by default: ~25x fewer, highly-redundant rows -> a tractable fit with no
    meaningful signal loss. No-op when ``subsample_fps`` is ``None``/non-positive, ``frame_rate`` is
    absent/non-positive, or the computed step is 1.
    """
    if subsample_fps is None or "frame_rate" not in frames.columns or len(frames) == 0:
        return frames
    fr = frames["frame_rate"].iloc[0]
    if not (fr > 0 and subsample_fps > 0):
        return frames
    step = max(1, round(fr / subsample_fps))
    if step == 1:
        return frames
    uniq = (
        frames[["game_id", "period_id", "frame_id"]].drop_duplicates().sort_values(["game_id", "period_id", "frame_id"])
    )
    keep_mask = uniq.groupby(["game_id", "period_id"]).cumcount() % step == 0
    keep = uniq[keep_mask.values]
    return frames.merge(keep, on=["game_id", "period_id", "frame_id"])


def extract_match(
    frames: pd.DataFrame,
    actions: pd.DataFrame | None,
    feature_set: _FeatureSet,
    *,
    n_rearguard: int = 4,
    home_team_id: int | str | None = None,
    subsample_fps: float | None = None,
):
    """Extract per-(frame, rearguard-slot) features for ONE match; ``None`` on zero rows.

    Frames are thinned to ``subsample_fps`` frames/sec FIRST (``None`` -> keep all), mirroring the
    ghost-GK trainer -- 25 fps tracking is far more (near-duplicate) rows than a mean-positioning model
    needs. Then the in-possession team is resolved from ``team_in_possession`` when present, else the
    ball carrier is inferred (the rearguard's owner). ``home_team_id`` (score perspective) is
    caller-supplied (the tc3 ``meta.json`` home); when ``None`` it falls back to the first non-ball
    team --- consulted only when ``actions`` carry a goal, and only for the sign of ``score_diff``
    (never orientation).
    """
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier
    from silly_kicks.tracking._ghost_outfield import _extract_all_ghost_outfield_features

    frames = _subsample_frames(frames, subsample_fps)
    non_ball = frames[~frames["is_ball"].astype(bool)]
    ids = non_ball["team_id"].dropna()
    if ids.empty:
        return None
    home = home_team_id if home_team_id is not None else ids.iloc[0]
    carrier = None
    if "team_in_possession" not in frames.columns:
        carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)
        c = infer_ball_carrier(frames, **carrier_params)
        carrier = c[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]
    feats = _extract_all_ghost_outfield_features(
        frames,
        actions,
        home_team_id=home,
        carrier=carrier,
        feature_set=feature_set,
        n_rearguard=n_rearguard,
        both_teams=True,  # possession-conditioned training: model BOTH teams' deepest-n per frame
    )
    if feats is None or len(feats) == 0:
        return None
    # Tag the match's provider for the StratifiedGroupKFold(match+provider) CV + per-provider MAE
    # (spec 6). A match has ONE provider; absent -> "unknown".
    prov = "unknown"
    if "source_provider" in frames.columns:
        present = frames["source_provider"].dropna()
        if len(present):
            prov = str(present.iloc[0])
    feats = feats.copy()
    feats["source_provider"] = prov
    return feats


def _actions_for(frames_path: Path, args) -> pd.DataFrame | None:
    """Locate the SPADL actions parquet mirroring ``frames_path`` under ``--actions-dir`` (or None).

    The tc3 ghost cache is ``{provider}/{game}/frames.parquet`` with a SEPARATE flat
    ``_actions/{game}.parquet`` (the game DIRECTORY name is the actions key --- dir name == game_id ==
    actions filename across gradientsports / skillcorner / idsse, verified on the ghost_cache corpus).
    An earlier resolver only tried ``_actions/{provider}/{game}/actions.parquet`` and
    ``_actions/frames.parquet`` (``rel.stem`` is ``"frames"`` under the nested layout), so it matched
    NOTHING on the real corpus and every ``phase`` / ``score_diff`` feature trained constant-0. The
    flat ``{game}.parquet`` toy layout masked this. Candidates are tried in order.
    """
    if args.actions_dir is None:
        return None
    rel = frames_path.relative_to(args.data_dir)
    candidates: list[Path] = []
    if len(rel.parts) > 1:
        # tc3 cache: {provider}/{game}/frames.parquet -> flat _actions/{game}.parquet (game = dir name)
        candidates.append(args.actions_dir / f"{rel.parts[-2]}.parquet")
        # nested variant: _actions/{provider}/{game}/actions.parquet
        candidates.append(args.actions_dir / rel.parent / "actions.parquet")
    # flat frames layout: {game}.parquet -> _actions/{game}.parquet
    candidates.append(args.actions_dir / f"{rel.stem}.parquet")
    from scripts._driver_sources import read_dataset_parquet

    for cand in candidates:
        if cand.exists():
            return read_dataset_parquet(cand)
    return None


def _home_team_for(frames_path: Path, frames: pd.DataFrame, home_map: dict | None = None) -> int | str | None:
    """The authoritative home team for the ``score_diff`` perspective: a ``--home-teams`` map keyed by
    ``game_id`` (a FLAT for_each generation has no per-match ``meta.json``), else the ``meta.json``
    sibling of a tc3-tree frames parquet, else the first non-ball ``team_id``.

    ``score_diff`` is signed from HOME's perspective, so an arbitrary first-team fallback flips the
    sign on any match whose frame row order leads with the away team (measured on IDSSE:
    ``ids.iloc[0]`` disagreed with ``meta.home_team_id``). ``home_team_id`` is consulted ONLY for the
    score lookup --- orientation is GoalMap-based (ADR-055) --- so a wrong home corrupts the sign of
    one feature, never the geometry. Returns ``None`` on an all-ball / empty frame (the caller drops it).
    """
    if home_map is not None:
        gids = frames["game_id"].dropna()
        if len(gids):
            hv = home_map.get(str(gids.iloc[0]))
            if hv is not None:
                return hv
    meta = frames_path.parent / "meta.json"
    if meta.exists():
        from scripts._driver_sources import read_dataset_json

        try:
            return read_dataset_json(meta)["home_team_id"]
        except (KeyError, ValueError, OSError):
            pass
    ids = frames[~frames["is_ball"].astype(bool)]["team_id"].dropna()
    return ids.iloc[0] if len(ids) else None


def _home_map(ctx) -> dict | None:
    from scripts._driver_sources import home_teams

    return home_teams(ctx.data_root, ctx.params["dataset"]) or None


def _extract_refs(ctx):
    from scripts._driver_sources import staged_flat_refs

    return staged_flat_refs(ctx.data_root, ctx.params["dataset"])


def _extract_load(ref, ctx):
    """``(frames_path, legacy args namespace)``: the legacy ``_work`` reads the frames and resolves the
    actions / home team through ``_actions_for`` / ``_home_team_for`` unchanged (decision P16)."""
    from scripts._driver_sources import dataset_dir

    root = dataset_dir(ctx.data_root, ctx.params["dataset"])
    frames_path = root / ref.files[0]
    ns = types.SimpleNamespace(data_dir=frames_path.parent, actions_dir=root / "_actions")
    return frames_path, ns


def _extract_work(item, ctx):
    from scripts._driver_sources import read_dataset_parquet

    p, ns = item
    frames = read_dataset_parquet(p)
    return extract_match(
        frames,
        _actions_for(p, ns),
        feature_set_for_variant(ctx.params["variant"]),
        n_rearguard=ctx.params["n_rearguard"],
        home_team_id=_home_team_for(p, frames, _home_map(ctx)),
        subsample_fps=ctx.params["subsample_fps"],
    )


def _extract_reduce(results, ctx):
    """The legacy combine: NON-EMPTY shards in sorted key order (``res.shard_keys`` order)."""
    from scripts._stages import combine_frames

    # a multi-provider corpus mixes int and str ids (gradientsports vs idsse / skillcorner): stored as the
    # canonical strings the legacy grouped on
    return combine_frames([f for f in results.values() if len(f)], ids=("game_id", "player_id", "team_id"))


#: The reusable `extract` stage (ADR-129); the lambdas are declared here (``test_stage_modules``).
_EXTRACT = ItemStage(
    name="extract",
    refs=_extract_refs,
    key=lambda ref: ref.stem,  # the legacy flat-layout `_key`: the shard stem; _KEY_EXCEPTIONS
    load=_extract_load,
    work=_extract_work,
    reduce=_extract_reduce,
    # extraction_inputs is the legacy token; n_rearguard is declared too, so a changed rearguard
    # size can never reuse stale shards.
    params=("n_rearguard",),
    declare=lambda p: extraction_inputs(p["variant"], p["subsample_fps"]),
    uses_dataset=True,
    max_failures=None,  # legacy WARNs and continues
)
STAGES = {"extract": _EXTRACT}
