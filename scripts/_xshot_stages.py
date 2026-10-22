"""ADR-129: the reusable `extract` stage of `scripts.train_xshot_occurrence`; its closure must not contain
the trainer (`test_stage_modules`). The fit-side stages (prep/study/cv/select/fit/assemble) stay in the
trainer and use none of these helpers."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from scripts._stages import ItemStage


def _source_key(item):
    """The `for_each` key: a `MatchRef` -> its `.key`; a per-game item or `_PergameRef` ->
    `(provider, match_id)`. Module-level so the key-pin gate can see it (_KEY_EXCEPTIONS)."""
    key = getattr(item, "key", None)
    return key if key is not None else (str(item[0]), str(item[1]))


#: The four per-row arrays `_extract` returns alongside the feature matrix, carried as COLUMNS so
#: one match is one tidy shard. Underscore-prefixed and collision-checked below: a feature named
#: `_y` would be silently overwritten, and the model would train on the label.
_SIDE_COLS = ("_y", "_group", "_provider", "_match_id")


def _extract_declare(p) -> dict:
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    return {
        "extractor": "prepare_xshot_training_data",
        "feature_set": p["feature_set"],
        "horizon_seconds": p["horizon_seconds"],
        "attacking_third_only": True,
        "carrier_params": dict(DEFAULT_CARRIER_PARAMS),
    }


class _PergameRef(NamedTuple):
    """A staged flat unit seen as the legacy per-game item. ``(prov, mid)`` is the legacy `_source_key`:
    the shard's ``source_provider`` and the repackaged ``{provider}-{game}`` directory name, so unit keys
    and the in-job arm match keys stay byte-identical to the 4.138.0 runs."""

    prov: str
    mid: str
    stem: str
    files: tuple[str, ...]


def _extract_refs(ctx) -> list[_PergameRef]:
    from scripts._driver_sources import staged_flat_refs
    from scripts.hf_jobs.datasets import pergame_name

    refs = staged_flat_refs(ctx.data_root, ctx.params["dataset"])
    return [_PergameRef(r.source_provider, pergame_name(r.provider, r.game), r.stem, r.files) for r in refs]


def _extract_load(ref: _PergameRef, ctx):
    """``(prov, mid, shots, frames, home)`` exactly as `repackage` + the legacy `_iter_matches_from_dir`
    built it: shots = the SPADL shot actions of the unit's ``_actions`` shard; home = the first non-null
    ``team_id`` of the frames (the legacy rule)."""
    from scripts._driver_sources import dataset_dir, read_dataset_parquet
    from scripts.hf_jobs.repackage import _id_to_name, _shot_type_ids, _shots_frame

    root = dataset_dir(ctx.data_root, ctx.params["dataset"])
    frames = read_dataset_parquet(root / ref.files[0])
    apath = root / ref.files[1]
    actions = read_dataset_parquet(apath) if apath.exists() else pd.DataFrame(columns=["type_id"])
    shots = _shots_frame(actions, _shot_type_ids(), _id_to_name())
    return ref.prov, ref.mid, shots, frames, frames["team_id"].dropna().iloc[0]


def _extract_work(item, ctx) -> pd.DataFrame | None:
    """One match -> its tidy feature rows + the four side columns (the legacy `_extract._work`), or None
    ("ran, produced no usable row")."""
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, prepare_xshot_training_data

    collision = set(_SIDE_COLS) & set(XSHOT_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")
    prov, mid, shots, frames, home = item
    X, y, groups = prepare_xshot_training_data(
        frames,
        shots,
        home_team_id=home,
        feature_set=ctx.params["feature_set"],
        horizon_seconds=ctx.params["horizon_seconds"],
        attacking_third_only=True,
        carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
    )
    if not len(X):
        return None
    return X.assign(
        _y=np.asarray(y, int),
        _group=np.asarray(groups),
        _provider=str(prov),
        _match_id=str(mid),  # per-row match id (the arm key)
    )


def _extract_counters(item, _frame) -> dict:
    """The ADR-126 integrity inputs of one match (shots by the label's own ``type_id`` set, frames,
    duration): counters only, never in a token or the artifact (ADR-128 plan Task 20)."""
    from scripts._integrity_counters import integrity_counters
    from silly_kicks.tracking._xshot_occurrence import _DEFAULT_SHOT_TYPES

    _prov, _mid, shots, frames, _home = item
    return integrity_counters(frames, shots, _DEFAULT_SHOT_TYPES)


def _extract_reduce(results, ctx) -> pd.DataFrame:
    """The legacy combine: NON-EMPTY shards in the legacy iteration order (sorted per-game dir names)."""
    from scripts._stages import combine_frames

    parts = [f for f in results.values() if len(f)]
    if not parts:
        raise RuntimeError("No usable training data.")
    # `_group` mixes kloppy str and Gradient Sports int game ids: stored as the strings the study groups on
    return combine_frames(parts, ids=("_group",))


def _mid_of(key: str) -> str:
    """The per-game dir name of a joined `_source_key` unit key (``<prov>__<mid>``): the legacy order."""
    return key.partition("__")[2]


#: The reusable `extract` stage (ADR-129); declared here so every callable's ``__module__`` is this module.
_EXTRACT = ItemStage(
    name="extract",
    refs=_extract_refs,
    key=_source_key,
    load=_extract_load,
    work=_extract_work,
    counters=_extract_counters,
    reduce=_extract_reduce,
    reduce_order=_mid_of,
    declare=_extract_declare,
    uses_dataset=True,
    max_failures=0,  # the legacy raised on any failed match
    # ADR-129 S5.9: NO data_files. The Task-14 file-audit hook over every toy proved no reusable stage in the
    # 7 Cycle-1 pipelines opens a bundled `_*_weights/` file (spec R7: declare only what the audit proves a
    # stage reads). The data-membership MECHANISM is gated behaviourally in test_closure_lock, not by a
    # declaration the code does not exercise.
)
STAGES = {"extract": _EXTRACT}
