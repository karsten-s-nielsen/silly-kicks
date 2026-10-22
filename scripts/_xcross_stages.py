"""ADR-129: the reusable `extract` and `probe` stages of `scripts.train_xcross_attempt`; their closure
must not contain the trainer (`test_stage_modules`). The fit-side stages (probe_comparison/prep/study/cv/
select/fit/assemble) stay in the trainer and import the shared helpers (`_csv`, `_mid_of`, `_capture_record`,
`_cohort_files`, `PROBE_KEEP`) from here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from scripts._stages import FileSet, ItemStage, join_key


def _source_key(item):
    """The `for_each` key: a `MatchRef` -> its `.key`; a per-game item or `_PergameRef` ->
    `(provider, match_id)`. Module-level so the key-pin gate can see it (_KEY_EXCEPTIONS)."""
    key = getattr(item, "key", None)
    return key if key is not None else (str(item[0]), str(item[1]))


def _new_probe_cohort() -> dict:
    """One TF-19 probe cohort's capture state (M5): bounded frames/actions copies + provenance."""
    return {"frames": [], "actions": [], "home": None, "matches": [], "match_groups": {}}


#: The four per-row arrays `_extract` returns alongside the feature matrix, carried as COLUMNS so
#: one match is one tidy shard. Underscore-prefixed and collision-checked: a feature named `_y`
#: would be silently overwritten, and the model would train on its own label.
_SIDE_COLS = ("_y", "_group", "_provider", "_match_id")

#: The legacy capture budget per probe cohort (M5: bounded frames copies).
PROBE_KEEP = 2


def _write_probe_sample(ps: Path, cohort: dict, provider_filter: list) -> None:
    """Persist one probe cohort + its provenance meta.json (M5). No-op on an empty cohort
    (no match from the filtered providers seen). Extracted so the write is unit-testable
    without Databricks."""
    if not cohort["frames"]:
        return
    ps.mkdir(parents=True, exist_ok=True)
    pd.concat(cohort["frames"], ignore_index=True).to_parquet(ps / "frames.parquet")
    pd.concat(cohort["actions"], ignore_index=True).to_parquet(ps / "actions.parquet")
    meta = {
        "home_team_id": str(cohort["home"]),
        "probe_matches": cohort["matches"],  # [[provider, match_id], ...]
        "probe_providers": list(provider_filter),  # the capture filter used
        "match_groups": cohort["match_groups"],  # match_id -> [game_id, ...] (M6 fold membership)
    }
    json.dump(meta, open(ps / "meta.json", "w"), indent=2)


def _csv(text: str) -> tuple[str, ...]:
    return tuple(p for p in str(text).split(",") if p)


def _extract_declare(p) -> dict:
    """What determines a feature shard's CONTENT (the legacy `_extract` token inputs, unchanged). The probe
    provider filters are NOT declared: they select which matches are copied into a cohort, not what a
    feature row contains."""
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS

    return {
        "extractor": "prepare_xcross_training_data",
        # feature_set changes the X columns (16 vs 15) -> MUST key the shard generation (4.77.1).
        "feature_set": p["feature_set"],
        "horizon_seconds": p["horizon_seconds"],
        "wide_area_only": True,
        "carrier_params": dict(DEFAULT_CARRIER_PARAMS),
    }


class _PergameRef(NamedTuple):
    """A staged unit seen as the legacy per-game item. ``(prov, mid)`` is the legacy `_source_key`: the
    shard's ``source_provider`` and the per-game directory name (training: the repackaged
    ``{provider}-{game}``; probe: the staged per-game directory), so unit keys and arm keys stay
    byte-identical to the 4.138.0 runs. ``home_map`` says whether the dataset's home map applies."""

    prov: str
    mid: str
    stem: str
    files: tuple[str, ...]
    dataset_param: str


def _extract_refs(ctx) -> list[_PergameRef]:
    from scripts._driver_sources import staged_flat_refs
    from scripts.hf_jobs.datasets import pergame_name

    refs = staged_flat_refs(ctx.data_root, ctx.params["dataset"])
    return [_PergameRef(r.source_provider, pergame_name(r.provider, r.game), r.stem, r.files, "dataset") for r in refs]


def _probe_refs(ctx) -> list[_PergameRef]:
    """The probe dataset's per-game units (`staged_pergame_refs` skips dot-directories such as an
    ``hf download`` ``.cache/`` -- the 4.138.0 crash)."""
    from scripts._driver_sources import staged_pergame_refs

    refs = staged_pergame_refs(ctx.data_root, ctx.params["probe_dataset"])
    return [_PergameRef(r.source_provider, r.stem, r.stem, r.files, "probe_dataset") for r in refs]


def _home_map(ctx) -> dict | None:
    from scripts._driver_sources import home_teams

    return home_teams(ctx.data_root, ctx.params["dataset"]) or None


def _load_match(ref: _PergameRef, ctx):
    """``(prov, mid, actions, frames, home)`` exactly as the legacy `_iter_matches_from_dir` built it.

    xcross USES home to sign score_differential (PA-H1), so the training dataset's ``home_teams.json``
    overrides the first-team heuristic (keyed by game_id; TF-65 C2). The probe pass never had a map."""
    from scripts._driver_sources import dataset_dir, read_dataset_parquet

    root = dataset_dir(ctx.data_root, ctx.params[ref.dataset_param])
    frames = read_dataset_parquet(root / ref.files[0])
    actions = read_dataset_parquet(root / ref.files[1])
    home = None
    home_map = _home_map(ctx) if ref.dataset_param == "dataset" else None
    if home_map is not None:
        gids = frames["game_id"].dropna()
        if len(gids):
            home = home_map.get(str(gids.iloc[0]))
    if home is None:
        home = frames["team_id"].dropna().iloc[0]
    return ref.prov, ref.mid, actions, frames, home


def _prepare(item, ctx):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    collision = set(_SIDE_COLS) & set(XCROSS_FEATURE_NAMES_FAITHFUL)
    if collision:
        raise ValueError(f"side columns {sorted(collision)} collide with feature names")
    _prov, _mid, actions, frames, home = item
    return prepare_xcross_training_data(
        frames,
        actions,
        home_team_id=home,
        feature_set=ctx.params["feature_set"],
        horizon_seconds=ctx.params["horizon_seconds"],
        wide_area_only=True,
        carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 values; shared constant (anti-drift)
    )


def _extract_work(item, ctx) -> pd.DataFrame | None:
    """One match -> its tidy feature rows + the four side columns (the legacy `_extract._work`), or None."""
    prov, mid, *_ = item
    X, y, groups = _prepare(item, ctx)
    if not len(X):
        return None  # "ran, produced no usable row"
    return X.assign(
        _y=np.asarray(y, int),
        _group=np.asarray(groups),
        _provider=str(prov),
        _match_id=str(mid),  # per-row match id (the arm key)
    )


def _extract_counters(item, _frame) -> dict:
    """The ADR-126 integrity inputs of one match (crosses by the label's own ``type_id`` set, frames,
    duration): counters only, never in a token or the artifact (ADR-128 plan Task 20)."""
    from scripts._integrity_counters import integrity_counters
    from silly_kicks.tracking._xcross_attempt import _DEFAULT_CROSS_TYPES

    _prov, _mid, actions, frames, _home = item
    return integrity_counters(frames, actions, _DEFAULT_CROSS_TYPES)


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


def _capture_record(prov, mid, groups) -> pd.DataFrame:
    """A cohort capture as ONE row (an ItemStage result): the legacy cohort's per-match provenance (the
    match's ``game_id`` groups, M6). Its frames are reloaded from the immutable staged dataset."""
    return pd.DataFrame(
        {
            "provider": [str(prov)],
            "match_id": [str(mid)],
            "groups": [json.dumps(sorted({str(g) for g in np.asarray(groups).tolist()}))],
        }
    )


def _probe_work(item, ctx) -> pd.DataFrame | None:
    """One probe match: its capture record when its provider is GATED (``--probe-providers``) and the
    extraction yields rows -- the legacy probe pass's capture rule -- else None."""
    prov, mid, *_rest = item
    if prov not in _csv(ctx.params["probe_providers"]):
        return None
    X, _y, groups = _prepare(item, ctx)
    if not len(X):
        return None
    return _capture_record(prov, mid, groups)


def _cohort_files(ctx, records: list[pd.DataFrame], refs: dict, provider_filter) -> FileSet:
    """The legacy `_write_probe_sample` payload for the FIRST `PROBE_KEEP` captures (sorted key order =
    the legacy iteration order), reloading each match's frames + actions from the immutable staged dataset,
    so a resumed stage rebuilds the identical cohort (the legacy guard `res.skipped` is 0 by construction)."""
    import tempfile

    cohort = _new_probe_cohort()
    for rec in records[:PROBE_KEEP]:
        row = rec.iloc[0]
        ref = refs[join_key((row["provider"], row["match_id"]))]
        _prov, _mid, actions, frames, home = _load_match(ref, ctx)
        cohort["frames"].append(frames.copy())
        cohort["actions"].append(actions.copy())
        cohort["home"] = home
        cohort["matches"].append([str(row["provider"]), str(row["match_id"])])
        cohort["match_groups"][str(row["match_id"])] = json.loads(row["groups"])
    with tempfile.TemporaryDirectory() as tmp:
        ps = Path(tmp) / "sample"
        _write_probe_sample(ps, cohort, list(provider_filter))
        files = {q.name: q.read_bytes() for q in sorted(ps.iterdir())} if ps.exists() else {}
    return FileSet(files)


def _probe_reduce(results, ctx) -> FileSet:
    st = ctx.pipeline.stage("probe")
    assert isinstance(st, ItemStage)  # noqa: S101 -- the pipeline declares it so
    records = [results[k] for k in results if len(results[k])]
    return _cohort_files(ctx, records, st.ref_map(ctx), _csv(ctx.params["probe_providers"]))


#: The reusable `extract` and `probe` stages (ADR-129); declared here so every callable's ``__module__``
#: is this stage module (``test_stage_modules``).
_EXTRACT = ItemStage(
    name="extract",
    refs=_extract_refs,
    key=_source_key,
    load=_load_match,
    work=_extract_work,
    counters=_extract_counters,
    reduce=_extract_reduce,
    reduce_order=_mid_of,
    declare=_extract_declare,
    uses_dataset=True,
    max_failures=0,  # the legacy raised on any failed match
)
_PROBE = ItemStage(
    name="probe",
    refs=_probe_refs,
    key=_source_key,
    load=_load_match,
    work=_probe_work,
    reduce=_probe_reduce,
    reduce_order=_mid_of,
    params=("probe_providers",),
    declare=_extract_declare,
    uses_dataset=True,
    dataset_param="probe_dataset",
    max_failures=0,
)
STAGES = {"extract": _EXTRACT, "probe": _PROBE}
