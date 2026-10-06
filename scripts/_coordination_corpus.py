"""Shared per-match plumbing for the TF-58 coordination drivers D1/D2/D3 (spec 8.3).

One place owns the argparse surface every coordination driver shares (``add_common_args``), the resumable
tracking source (``corpus_source`` -> ``pining_source`` refs + load), the per-stage wall-clock counters the
manifest carries (``StageTimer``, R8), and the per-match metric-table builder every driver melts its shards
from (``match_tables``). Keeping it here -- rather than in each of ``derive_coordination_params`` /
``calibrate_coordination`` / ``validate_team_coordination`` -- is what lets the three drivers reuse ONE corpus
walk and ONE table schema (ADR-052 shared-driver discipline).

Imports of the heavy corpus machinery (``_loader_pining``, ``_partition``) stay INSIDE the functions that need
them, the ``_driver`` house pattern: those modules pull pandas/network deps that the pure helpers here (and
the unit tests that import them) must not require.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import pandas as pd

from silly_kicks.coordination import (
    COORD_WINDOW_COLUMNS,
    COORDINATION_CLUSTER_PLAYER_COLUMNS,
    COORDINATION_CLUSTER_TEAM_COLUMNS,
    COORDINATION_PAIR_COLUMNS,
    COORDINATION_PAIR_PHASE_COLUMNS,
    COORDINATION_RSI_COLUMNS,
    COORDINATION_SPECTRAL_COLUMNS,
    COORDINATION_TEAM_SYNC_COLUMNS,
    CoordinationParams,
    build_coordination_signals,
    period_windows,
    possession_windows_from_actions,
    possession_windows_from_frames,
)
from silly_kicks.coordination._compute import _result_from_signals, rsi_switch_times

#: The TF-58 tracking corpus: SkillCorner (909), GradientSports (64), IDSSE (7) ~= 980 matches (spec 8.3).
TF58_PROVIDERS = ("skillcorner", "gradientsports", "idsse")

#: The CoordinationResult tables melted into every shard, in a fixed order. ``windows`` is carried so the
#: possession-terminal / period columns are available to the D3 hypothesis reducers (H3/H5) at reduce time.
RESULT_TABLES = ("windows", "pair", "pair_phase", "spectral", "cluster_team", "cluster_player", "team_sync", "rsi")

#: The leading tags every block of a long ``match_tables`` frame carries.
LEAD_COLUMNS = ("provider", "match_id", "variant", "table")
#: The H7 switch-event blocks' own columns (``rsi_switch_times`` / ``possession_change_times``).
RSI_SWITCH_COLUMNS = ("game_id", "period_id", "axis", "time")
POSSESSION_CHANGE_COLUMNS = ("game_id", "period_id", "time")
#: Each table of a long ``match_tables`` frame and its OWN columns -- the schema it was melted from.
TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "windows": tuple(COORD_WINDOW_COLUMNS),
    "pair": tuple(COORDINATION_PAIR_COLUMNS),
    "pair_phase": tuple(COORDINATION_PAIR_PHASE_COLUMNS),
    "spectral": tuple(COORDINATION_SPECTRAL_COLUMNS),
    "cluster_team": tuple(COORDINATION_CLUSTER_TEAM_COLUMNS),
    "cluster_player": tuple(COORDINATION_CLUSTER_PLAYER_COLUMNS),
    "team_sync": tuple(COORDINATION_TEAM_SYNC_COLUMNS),
    "rsi": tuple(COORDINATION_RSI_COLUMNS),
    "rsi_switch_times": RSI_SWITCH_COLUMNS,
    "possession_changes": POSSESSION_CHANGE_COLUMNS,
}


def split_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A long ``match_tables`` frame split into per-``table`` frames, each carrying ONLY the lead tags and its own
    schema's columns (:data:`TABLE_COLUMNS`).

    The long frame is an outer union: every table's rows carry every other table's columns as NaN. A reducer that
    merged or filtered on such a foreign column met the windows' ``attacking_team_id`` twice (a ``KeyError`` in
    H3/H5) or read a pair slice through spectral's all-NaN ``team_id`` (empty reliability samples) -- review A-03.
    An unknown table is refused: a reducer must never guess another table's columns. The one splitter every driver
    uses (D2's confirm, D3's reduce).
    """
    if not len(frame) or "table" not in frame.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for table in pd.unique(frame["table"]):
        own = TABLE_COLUMNS.get(str(table))
        if own is None:
            raise ValueError(f"unknown table {table!r} in a match_tables frame (known: {sorted(TABLE_COLUMNS)})")
        keep = [c for c in (*LEAD_COLUMNS, *own) if c in frame.columns]
        out[str(table)] = pd.DataFrame(frame.loc[frame["table"] == table, keep]).reset_index(drop=True)
    return out


def _split_providers(text: str) -> tuple[str, ...]:
    """``"skillcorner,idsse"`` -> ``("skillcorner", "idsse")`` (the ``--providers`` argparse type)."""
    return tuple(part.strip() for part in text.split(",") if part.strip())


def add_common_args(parser) -> None:
    """Add the corpus-driver arguments every coordination driver shares (the ``validate_*`` CLI shape)."""
    parser.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    parser.add_argument("--token", default=None, help="pining token (else resolved from the environment)")
    parser.add_argument("--max-matches", type=int, default=None, help="cap matches per provider (dev)")
    parser.add_argument("--cache-dir", default=None, help="raw-artifact cache dir (or $SILLY_KICKS_CORPUS_CACHE_DIR)")
    parser.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"skillcorner": ["id", ...], ...} pinning WHICH matches this process handles (parallel split).',
    )
    parser.add_argument(
        "--providers",
        type=_split_providers,
        default=TF58_PROVIDERS,
        help="comma-separated provider list (default: the full TF-58 corpus)",
    )
    parser.add_argument(
        "--corpus-json",
        default=None,
        help='JSON {"skillcorner": ["id", ...], ...}: the FULL unsplit corpus of a partitioned run. Required with '
        "--match-ids-json wherever workers' shares are combined, so the combine proves it saw all of it.",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    parser.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")


def load_match_ids(args) -> dict[str, list[str]] | None:
    """The ``--match-ids-json`` mapping (provider -> ids), or ``None`` when the whole corpus is requested."""
    if not args.match_ids_json:
        return None
    return json.loads(Path(args.match_ids_json).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- partitioned passes (spec 8.3, B-1)
#: The columns every pass table carries; a combined table is ordered by them (stable), so a reduce reads the same
#: rows in the same order whether one worker or sixteen produced them.
_ORDER_COLUMNS = ("provider", "match_id")


def params_token(providers, params_for: Callable[[str], CoordinationParams] = CoordinationParams.for_provider) -> str:
    """A digest of the per-provider :class:`CoordinationParams` a pass computes with -- in its ``token_inputs`` so a
    parameter change re-runs the shards (M-4: a token carries the VALUES the work consumes, not only their source)."""
    blob = {
        str(p): {
            f.name: (dict(v) if isinstance(v, Mapping) else v)
            for f in dataclasses.fields(params)
            if f.compare
            for v in (getattr(params, f.name),)
        }
        for p in sorted(providers)
        for params in (params_for(p),)
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def run_params_token(args, params_for: Callable[[str], CoordinationParams] = CoordinationParams.for_provider) -> str:
    """The :func:`params_token` a corpus pass puts in its shard token: over EVERY TF-58 provider (plus any other
    ``--providers`` named), never only the providers this worker's slice holds.

    Every worker of one run must write ONE shard generation -- the combine refuses a mix (B-1) -- yet workers are
    launched with the providers of their own slice. A token over ``args.providers`` alone therefore split one run
    into several generations (the 2026-10-03 MEDIA-PC no-flip reduce refused exactly that). A params change for any
    provider still re-keys every worker: conservative, never stale.
    """
    return params_token(sorted(set(TF58_PROVIDERS) | {str(p) for p in args.providers}), params_for)


def expected_corpus(args, providers) -> set[str] | None:
    """The joined shard keys a combine of this pass must see in full: ``--corpus-json`` restricted to the pass's
    ``providers`` -- or ``None`` for an unpartitioned run, whose one worker's own listing IS the corpus.

    A partitioned run (``--match-ids-json``) without ``--corpus-json`` is refused: nothing else can tell a combine
    that a worker never ran, and spec 8.3 says the full corpus is used, never a partial one.
    """
    from scripts._driver import join_key

    if getattr(args, "corpus_json", None):
        corpus = json.loads(Path(args.corpus_json).read_text(encoding="utf-8"))
        return {join_key((p, str(m))) for p in providers for m in corpus.get(p, [])}
    if getattr(args, "match_ids_json", None):
        raise SystemExit(
            "a partitioned run (--match-ids-json) must pass --corpus-json (the FULL unsplit corpus) so every combine "
            "can prove it saw all of it (spec 8.3: never a partial corpus)"
        )
    return None


def read_artifact(args, name: str, *, hint: str = "") -> tuple[dict, str]:
    """The upstream artifact ``--<name>`` as ``(content, sha256 of its bytes)`` -- refused when absent.

    The artifact handoff (owner ruling M-5, 2026-10-02): a downstream driver computes with the D1/D2 artifact the
    chain hands it, never the in-package module, and records the digest in its shard tokens and manifests.
    """
    path = getattr(args, name, None)
    if not path:
        raise SystemExit(f"this pass needs --{name} (the artifact it computes with; owner ruling M-5).{hint}")
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


#: The manifest fields naming where a pass's params came from; every combine requires the workers to agree on them.
PARAMS_PROVENANCE_FIELDS = ("params_source", "derivation_sha256", "calibration_sha256")
_IN_PACKAGE_HINT = " Pass --in-package-params for a dev run on the committed module instead."


def add_params_args(ap) -> None:
    """The artifact-handoff flags of every pass that computes with the FINAL params (owner ruling M-5)."""
    ap.add_argument("--derivation", default=None, help="D1's derivation.json (the artifact handoff, owner ruling M-5)")
    ap.add_argument(
        "--calibration", default=None, help="D2's calibration.json (the artifact handoff, owner ruling M-5)"
    )
    ap.add_argument(
        "--in-package-params",
        action="store_true",
        help="dev run: compute with the committed CoordinationParams.for_provider instead of the D1/D2 artifacts",
    )


def params_resolver(args) -> tuple[Callable[[str], CoordinationParams], dict]:
    """Where a FINAL-params pass gets each provider's params, plus the provenance its token, manifest and artifact
    carry (:data:`PARAMS_PROVENANCE_FIELDS`).

    By default from the D1/D2 ARTIFACTS (``scripts._coordination_params_codegen.params_from_artifacts``: exactly the
    values commit 2 will commit), so the DGX chain never rewrites the package; ``--in-package-params`` uses the
    committed ``CoordinationParams.for_provider`` (a dev run) and says so. The one resolver D3's passes and the
    numerics gate share (review A-16 / R2-2: the gate once certified the interim in-package values).
    """
    from scripts._coordination_params_codegen import params_from_artifacts

    if getattr(args, "in_package_params", False):
        return CoordinationParams.for_provider, {"params_source": "in_package"}
    derivation, derivation_sha = read_artifact(args, "derivation", hint=_IN_PACKAGE_HINT)
    calibration, calibration_sha = read_artifact(args, "calibration", hint=_IN_PACKAGE_HINT)
    provenance = {
        "params_source": "artifacts",
        "derivation_sha256": derivation_sha,
        "calibration_sha256": calibration_sha,
    }
    return (lambda provider: params_from_artifacts(provider, derivation, calibration)), provenance


def table_pairs(*tables: pd.DataFrame) -> set[tuple[str, str]]:
    """The (provider, match) pairs the given combined pass tables hold -- the population an artifact's aggregates come
    from (a table names the match ``match_id``, or ``game_id``)."""
    out: set[tuple[str, str]] = set()
    for table in tables:
        if not len(table) or "provider" not in table.columns:
            continue
        col = "match_id" if "match_id" in table.columns else ("game_id" if "game_id" in table.columns else None)
        if col is not None:
            out |= {(str(p), str(m)) for p, m in zip(table["provider"], table[col], strict=True)}
    return out


def corpus_visibility_label(pairs, *, token: str | None) -> str:
    """ADR-038 (spec 8.3): an artifact's corpus-visibility label, from the pining manifest's per-match ``visibility``
    -- never the provider name. Fail-closed: a match the manifest does not list is private, and a match may claim
    ``public`` only if it is one of the registered public set (``assert_public_corpus`` refuses otherwise)."""
    from scripts import _loader_pining
    from scripts._corpus import artifact_label, assert_public_corpus

    used = {(str(p), str(m)) for p, m in pairs}
    providers = sorted({p for p, _m in used})
    manifest = _loader_pining.match_visibility(providers, token=token) if providers else {}
    visibility = {key: manifest.get(key, "private") for key in used}
    assert_public_corpus(visibility)
    all_public = bool(visibility) and all(v == "public" for v in visibility.values())
    return artifact_label(providers=set(providers), all_public=all_public)


def visibility_preflight(refs, load, *, attempts: int = 3) -> None:
    """ADR-069 Layer 2 for a raw-artifact corpus (spec 8.3; owner ruling 2026-10-03), run by every corpus pass BEFORE
    any compute: load one match of each detection-aware provider in ``refs`` through the pass's own ``load`` and
    refuse -- with the ADR-069 native-rebuild remedy -- if its ``visibility`` flag was discarded (all null, or the
    column dropped). A loader path that throws the flag away is caught at the start of the pass, never deep in it.

    A match that fails to load for any other reason is a per-match problem the pass itself records, so the next match
    of that provider is tried (up to ``attempts``). The per-match ``detected_mask`` trap still refuses a single
    match's hole during the pass, and every combine refuses the failed key.
    """
    from silly_kicks.tracking._provider_visibility import (
        _DETECTION_AWARE_PROVIDERS,
        _detection_discarded_message,
        assert_detection_aware_visibility,
    )

    tried: dict[str, int] = {}
    checked: set[str] = set()
    for ref in refs:
        provider = ref.provider
        if provider not in _DETECTION_AWARE_PROVIDERS or provider in checked or tried.get(provider, 0) >= attempts:
            continue
        tried[provider] = tried.get(provider, 0) + 1
        try:
            frames = load(ref).frames
        except Exception as exc:  # a per-match load failure is the pass's to record: try the provider's next match
            print(f"visibility pre-flight: {ref.key} did not load ({exc!r}); trying {provider}'s next match")
            continue
        if frames is None:
            continue
        if "visibility" not in frames.columns:
            raise ValueError(_detection_discarded_message(provider))
        players = frames.loc[~frames["is_ball"].to_numpy(dtype=bool), "visibility"]
        assert_detection_aware_visibility(players, provider=provider)
        checked.add(provider)
    # A detection-aware provider that was probed but never verified (every probed match failed to load or carried no
    # frames) is a GAP, not a pass: record it with a warning rather than only the per-match prints (review nit). The
    # per-match detected_mask trap and the combine's failed-key refusal still apply during the pass.
    unverified = sorted(set(tried) - checked)
    if unverified:
        warnings.warn(
            f"visibility pre-flight could not verify {unverified}: every probed match (up to {attempts}) failed to "
            "load or carried no frames, so these providers' visibility flag was NOT checked at the pass boundary.",
            stacklevel=2,
        )


def write_worker_partial(dest, name: str, res, prov, timer, *, tag: str, extra: Mapping | None = None) -> pd.DataFrame:
    """This worker's share of pass ``name``: its own shards as ``<name>.<tag>.parquet`` (written atomically) and a
    ``manifest_<name>.<tag>.json`` naming exactly what it was handed (``listed``) and what became of each key
    (``produced`` / ``excluded_keys`` / ``failed_keys``) -- the evidence :func:`combine_workers` proves completeness
    from. Returns the share."""
    from scripts._driver import shard_path
    from scripts._partition import write_table_atomically

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys]
    non_empty = [f for f in parts if len(f)]
    share = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    write_table_atomically(share, dest / f"{name}.{tag}.parquet", tag=tag)
    manifest = _worker_manifest(res, prov, timer, name=name, tag=tag, extra=extra)
    (dest / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return share


def _worker_manifest(res, prov, timer, *, name: str, tag: str, extra: Mapping | None) -> dict:
    """The completeness-evidence manifest :func:`combine_workers` proves from (what was ``listed`` and what became of
    each key). Shared by the single-share and the per-variant writers -- the evidence is the SAME for every variant of
    one pass (same matches), so a variant share reuses it verbatim (only its ``pass`` name differs)."""
    return {
        **res.manifest(),
        **timer.manifest(),
        "pass": name,
        "partition": tag,
        "listed": list(res.keys),
        "produced": list(res.shard_keys),
        "excluded_keys": sorted(res.exclusions),
        "failed_keys": sorted(res.failures),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **(dict(extra) if extra else {}),
    }


def write_worker_partial_by_variant(
    dest, name_for: Callable[[str], str], res, prov, timer, *, tag: str, variants, extra: Mapping | None = None
) -> None:
    """Like :func:`write_worker_partial` but writes ONE share per ``variant`` value instead of one stacked share
    (reduce-memory, ADR-112 follow-up): a reused-signals layer-a pass emits every post-preparation variant in its
    per-match shards, and concatenating the whole stack materialises ~13x the melt in one frame (the D2 baseline
    OOM). Materialise ONE variant at a time -- the per-match shards are re-read per variant (I/O-for-memory) and
    filtered to that variant, so the worker's peak is one variant's melt, not the stack. ``name_for(variant)`` gives
    the share name; each variant's manifest is the same completeness evidence (identical matches). Nothing is dropped
    -- the union of the per-variant shares is exactly the stacked share."""
    from scripts._driver import shard_path
    from scripts._partition import write_table_atomically

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        parts: list[pd.DataFrame] = []
        for k in res.shard_keys:
            df = pd.read_parquet(shard_path(res.shard_dir, k))
            sub = df[df["variant"] == variant] if len(df) else df
            if len(sub):
                parts.append(sub.reset_index(drop=True))
        share = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        name = name_for(variant)
        write_table_atomically(share, dest / f"{name}.{tag}.parquet", tag=tag)
        manifest = _worker_manifest(res, prov, timer, name=name, tag=tag, extra=extra)
        (dest / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _read_concat_categorical(paths: list[Path]) -> pd.DataFrame:
    """Read + concat per-worker shares with the string columns as a SHARED sorted ``CategoricalDtype``
    (reduce-memory, rev 3). Each share's string columns are read dictionary-encoded (pyarrow) -> ``category``
    (so the wide object melt is never materialised); the per-share categories are unioned and SORTED so
    ``sort_values``/``groupby`` order matches the object path exactly -- NOT append-order ``union_categoricals``.
    The concat stays categorical; a downstream reduce must be categorical-safe (``observed=True`` etc.)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames: list[pd.DataFrame] = []
    cats: dict[str, set] = {}
    for p in paths:
        schema = pq.read_schema(p)
        str_cols = [
            n
            for n, t in zip(schema.names, schema.types, strict=True)
            if pa.types.is_string(t) or pa.types.is_large_string(t)
        ]
        df = pq.read_table(p, read_dictionary=str_cols).to_pandas()
        if not len(df):
            continue
        for c in str_cols:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                cats.setdefault(c, set()).update(df[c].cat.categories.tolist())
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    dtypes = {c: pd.CategoricalDtype(categories=sorted(v), ordered=False) for c, v in cats.items()}
    for df in frames:
        for c, dt in dtypes.items():
            if c in df.columns:
                df[c] = df[c].astype(dt)
    return pd.concat(frames, ignore_index=True)


def combine_workers(
    dest, name: str, *, expected: set[str] | None, consistent: tuple[str, ...] = (), categorical: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Every worker's share of pass ``name``, combined -- or a refusal naming what is missing (spec 8.3, B-1).

    Refuses unless: the workers ran ONE shard generation on ONE commit; no key failed; no key was handed to two
    workers; every worker accounted for every key it was handed; every worker recorded the SAME value for each
    manifest field in ``consistent`` (e.g. a calibrated width, an artifact digest); and the union of the shares is
    exactly ``expected`` (``--corpus-json``; an unpartitioned run is one worker tagged ``all``). The combined table is
    sorted (stably) by provider and match, so its rows do not depend on how the corpus was split. Returns
    ``(table, summary)``; the summary carries the population checked and the summed per-stage timings (R8).
    """
    from scripts._partition import UNPARTITIONED_TAG

    dest = Path(dest)
    prefix = f"manifest_{name}."
    manifests = {
        path.name[len(prefix) : -len(".json")]: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(dest.glob(f"{prefix}*.json"))
    }
    if not manifests:
        raise SystemExit(f"no worker manifest for pass {name!r} in {dest}: run the pass first")
    problems: list[str] = []
    generations = {m["generation"] for m in manifests.values()}
    commits = {m["run_commit"] for m in manifests.values()}
    if len(generations) != 1:
        problems.append(f"the workers ran different shard generations {sorted(generations)}")
    if len(commits) != 1:
        problems.append(f"the workers ran different commits {sorted(commits)}")
    failed = sorted(k for m in manifests.values() for k in m["failed_keys"])
    if failed:
        problems.append(f"{len(failed)} key(s) failed, e.g. {failed[:5]} -- re-run their worker to retry them")
    owner: dict[str, str] = {}
    shared: set[str] = set()
    for tag, m in manifests.items():
        for k in m["listed"]:
            if k in owner:
                shared.add(k)
            owner[k] = tag
        unaccounted = set(m["listed"]) - set(m["produced"]) - set(m["excluded_keys"]) - set(m["failed_keys"])
        if unaccounted:
            problems.append(f"worker {tag!r} neither produced nor excluded {len(unaccounted)} key(s) it was handed")
    agreed: dict[str, object] = {}
    for field in consistent:
        values = {json.dumps(m.get(field), sort_keys=True, default=str) for m in manifests.values()}
        if len(values) != 1:
            problems.append(f"the workers disagree on {field!r}: {sorted(values)}")
        else:
            agreed[field] = next(iter(manifests.values())).get(field)
    if shared:
        problems.append(f"{len(shared)} key(s) were handed to more than one worker, e.g. {sorted(shared)[:5]}")
    listed = set(owner)
    if expected is None and len(manifests) > 1:
        problems.append(f"{len(manifests)} worker shares but no --corpus-json to prove they cover the corpus")
    elif expected is None and next(iter(manifests)) != UNPARTITIONED_TAG:
        # review R2-1: one PARTITION's share is a slice of the corpus, not the corpus -- only an unpartitioned run
        # (one worker tagged 'all') may stand for it without --corpus-json
        problems.append(
            f"one worker share tagged {next(iter(manifests))!r} (a partition) but no --corpus-json to prove it covers "
            f"the corpus -- an unpartitioned run is tagged {UNPARTITIONED_TAG!r}"
        )
    if expected is not None and listed != expected:
        missing, extra = sorted(expected - listed), sorted(listed - expected)
        problems.append(f"the shares miss {len(missing)} corpus key(s) (e.g. {missing[:5]}) and add {len(extra)}")
    if problems:
        raise SystemExit(
            f"refusing to combine pass {name!r} in {dest}: " + "; ".join(problems) + ". A stale manifest from an "
            "earlier, differently split run in the same --out also lands here: use a fresh --out per run."
        )
    paths = [dest / f"{name}.{tag}.parquet" for tag in sorted(manifests)]
    if categorical:
        # reduce-memory (rev 3): string cols -> shared sorted CategoricalDtype, concat stays categorical.
        table = _read_concat_categorical(paths)
    else:
        shares = [pd.read_parquet(p) for p in paths]
        non_empty = [s for s in shares if len(s)]
        table = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    order = [c for c in _ORDER_COLUMNS if c in table.columns]
    if order:
        table = table.sort_values(order, kind="mergesort").reset_index(drop=True)
    stage_seconds: dict[str, float] = {}
    for m in manifests.values():
        for stage, seconds in (m.get("stage_seconds") or {}).items():
            stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(seconds)
    summary = {
        "pass": name,
        "n_workers": len(manifests),
        "partitions": sorted(manifests),
        "n_listed": len(listed),
        # the exact corpus the shares cover, independent of the split (D2 keys its objective identity on it, C27)
        "population_digest": hashlib.sha256("\n".join(sorted(listed)).encode("utf-8")).hexdigest()[:16],
        "n_attempted": sum(int(m.get("n_attempted", 0) or 0) for m in manifests.values()),
        "n_produced": sum(len(m["produced"]) for m in manifests.values()),
        "n_excluded": sum(len(m["excluded_keys"]) for m in manifests.values()),
        # the actual excluded keys (not just the count), for a reduce to check against a DECLARED set (exclusions nit)
        "excluded_keys": sorted({k for m in manifests.values() for k in m["excluded_keys"]}),
        "n_failed": 0,
        "population_checked_against": "corpus_json" if expected is not None else "unpartitioned_listing",
        "generation": next(iter(generations)),
        "run_commit": next(iter(commits)),
        "run_tree_dirty": any(bool(m["run_tree_dirty"]) for m in manifests.values()),
        "run_tree_state": sorted({str(m.get("run_tree_state")) for m in manifests.values()}),
        "stage_seconds": stage_seconds,
        "consistent": agreed,
    }
    return table, summary


#: The CLOSED vocabulary of reasons a corpus match may be DECLARED excluded with (exclusions nit). An exclusion for any
#: other reason -- or an UNDECLARED exclusion -- fails the gate: a PASS must cover (scored union declared_excluded) ==
#: corpus, never a silent subset. The owner ratifies / extends this set.
COORD_EXCLUSION_REASONS = frozenset({"no_tracking", "empty_after_filter"})


def read_declared_exclusions(path: str | None) -> dict[str, str]:
    """The DECLARED exclusions ``{joined_key: reason}`` from ``path`` (a JSON input, the authority -- NOT the run's own
    ``.excluded.json`` output, which would be circular). ``None`` -> empty (the strictest default: any exclusion fails
    until declared). Every reason must be in :data:`COORD_EXCLUSION_REASONS`."""
    if not path:
        return {}
    declared = {str(k): str(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
    bad = sorted(k for k, r in declared.items() if r not in COORD_EXCLUSION_REASONS)
    if bad:
        raise SystemExit(
            f"--declared-excluded uses a reason outside {sorted(COORD_EXCLUSION_REASONS)}: "
            f"{[(k, declared[k]) for k in bad]}"
        )
    return declared


def assert_exclusions_declared(excluded_keys, declared: Mapping[str, str]) -> dict[str, str]:
    """Refuse unless every actually-excluded key is DECLARED with a vocabulary reason; return the ``{key: reason}`` map
    of the exclusions in play (for the verdict). An undeclared exclusion is a silent-subset PASS -- the R2-1 class."""
    undeclared = sorted(set(excluded_keys) - set(declared))
    if undeclared:
        raise SystemExit(
            f"refusing a PASS over a subset: {len(undeclared)} excluded key(s) are not DECLARED "
            f"(e.g. {undeclared[:5]}). Add them to --declared-excluded with a reason, or re-run so they are scored."
        )
    return {k: declared[k] for k in sorted(excluded_keys)}


def corpus_source(args) -> tuple[list, Callable]:
    """``(refs, load)`` for ``for_each`` over the requested TF-58 tracking corpus (ADR-052 D14).

    ``providers_for_slice`` narrows the provider list to those this partition actually owns, so N workers
    sliced on disjoint ``--match-ids-json`` sets never re-load one another's providers in full (the measured
    ``_wanted_for_provider`` trap). Everything else is ``pining_source``: the cache dir is resolved once here.
    """
    from scripts._loader_pining import pining_source, resolve_cache_dir
    from scripts._partition import providers_for_slice

    match_ids = load_match_ids(args)
    providers = providers_for_slice(list(args.providers), match_ids)
    return pining_source(
        providers,
        match_ids=match_ids,
        max_per_provider=args.max_matches,
        cache_dir=resolve_cache_dir(args.cache_dir),
        token=args.token,
    )


class StageTimer:
    """Accumulate wall-clock seconds per named stage; fold them into the manifest (R8).

    Use as a context-manager factory -- ``with timer("load"): ...`` -- so a driver can time its stages
    (load, windows, signals, families, reduce) without threading counters by hand. Times for repeated
    stages accumulate, so a per-match loop reports total time in each stage across the corpus.

    Examples
    --------
    >>> timer = StageTimer()
    >>> with timer("windows"):
    ...     _ = 1 + 1
    >>> "windows" in timer.as_dict()
    True
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._seconds[stage] = self._seconds.get(stage, 0.0) + (time.perf_counter() - t0)

    def as_dict(self) -> dict[str, float]:
        """A copy of the accumulated per-stage seconds."""
        return dict(self._seconds)

    def manifest(self) -> dict[str, dict[str, float]]:
        """The ``stage_seconds`` block a driver merges into its manifest."""
        return {"stage_seconds": dict(self._seconds)}


def build_windows(frames: pd.DataFrame, actions: pd.DataFrame | None, params: CoordinationParams) -> pd.DataFrame:
    """Period windows plus possession windows (from events when present, else frames) -- built ONCE (spec 8.3)."""
    built = [period_windows(frames)]
    if actions is not None and len(actions):
        built.append(possession_windows_from_actions(actions, frames, n_phases=params.n_phases))
    else:
        built.append(possession_windows_from_frames(frames, n_phases=params.n_phases, params=params))
    return pd.concat(built, ignore_index=True)


def match_tables(
    loaded,
    params: CoordinationParams,
    *,
    n_surrogates: int,
    stoppage_evidence: Literal["auto", "ball_state", "events", "none"] = "auto",
    detection: Literal["auto", "detection_aware", "fully_observed"] = "auto",
    variants: Mapping[str, CoordinationParams] | None = None,
    include_switch_events: bool = False,
    timer: StageTimer | None = None,
) -> pd.DataFrame:
    """One match's coordination metric tables as a long, JSON-able frame (spec 8.3).

    Builds the windows and the filtered/resampled/oriented signals ONCE, then runs all seven families per
    ``params`` variant. Post-preparation variants (Welch segment, vector-coding epsilon, surrogate shift,
    analysis band) reuse the prepared signals via ``dataclasses.replace(signals, params=v)`` -- the D2 sweep
    reuse (spec 8.4). Variants that change a *preparation* parameter (Butterworth cutoff, resample rate) must NOT
    be passed here; D2 runs a separate corpus pass per preparation level so their signals are rebuilt.

    ``n_surrogates`` is forced on every variant (0 for the heavy D2 layer-a passes; 199 for D3). The result
    has ``provider``, ``match_id``, ``variant`` and ``table`` leading columns, then that table's own columns;
    rows from tables with different schemas coexist with NaN-filled absent columns (outer union).

    With ``include_switch_events`` (D2-confirm + D3, for H7), each variant also emits ``rsi_switch_times`` and
    ``possession_changes`` blocks -- the RSI sign-switch times and the attacking-team-change times H7 tests -- so
    the reducer never needs the raw signals at reduce/confirm time (spec 8.5).

    ``timer`` (the driver's ``StageTimer``) receives the per-stage breakdown (R8): ``windows``, ``signals``, one
    ``family.<name>`` counter per coordination family plus ``family.combine``, ``melt`` and ``switch_events``. The
    returned frame never depends on it.
    """
    frames = loaded.frames
    if frames is None or not len(frames):
        raise ValueError(f"match_tables needs tracking frames; {loaded.provider}/{loaded.match_id} has none")
    stage = timer if timer is not None else _untimed
    actions = loaded.actions
    base = dataclasses.replace(params, n_surrogates=n_surrogates)
    with stage("windows"):
        windows = build_windows(frames, actions, base)
    with stage("signals"):
        signals = build_coordination_signals(
            frames,
            windows=windows,
            params=base,
            actions=actions,
            stoppage_evidence=stoppage_evidence,
            detection=detection,
        )
    variant_params = (
        {"base": base}
        if variants is None
        else {name: dataclasses.replace(vp, n_surrogates=n_surrogates) for name, vp in variants.items()}
    )
    changes = None
    if include_switch_events:
        with stage("switch_events"):
            changes = possession_change_times(windows)
    parts: list[pd.DataFrame] = []
    for name, vp in variant_params.items():
        vsignals = signals if vp == base else dataclasses.replace(signals, params=vp)
        result = _result_from_signals(vsignals, on_stage=lambda family: stage(f"family.{family}"))
        with stage("melt"):
            parts.append(_melt_result(result, loaded.provider, loaded.match_id, name))
        if include_switch_events:
            with stage("switch_events"):
                parts.append(_switch_event_blocks(vsignals, changes, loaded.provider, loaded.match_id, name))
    with stage("melt"):
        return pd.concat(parts, ignore_index=True) if parts else _empty_long_frame()


def _untimed(_stage: str) -> contextlib.AbstractContextManager[None]:
    """The ``StageTimer`` stand-in when a caller passes no timer: every stage is a no-op context."""
    return contextlib.nullcontext()


def possession_change_times(windows: pd.DataFrame) -> pd.DataFrame:
    """Per (game, period), the start times of possession windows where the attacking team changes (H7, spec 8.5)."""
    poss = windows[windows["window_kind"] == "possession"].sort_values(["game_id", "period_id", "start_time_s"])
    rows: list[dict[str, object]] = []
    for _key, g in poss.groupby(["game_id", "period_id"], sort=True, observed=True):
        att = g["attacking_team_id"].to_numpy()
        times = g["start_time_s"].to_numpy(dtype=float)
        game = g["game_id"].iloc[0]
        period = int(g["period_id"].iloc[0])
        for i in range(1, len(att)):
            if not pd.isna(att[i]) and not pd.isna(att[i - 1]) and att[i] != att[i - 1]:
                rows.append({"game_id": game, "period_id": period, "time": float(times[i])})
    return pd.DataFrame(rows, columns=list(POSSESSION_CHANGE_COLUMNS))


def _switch_event_blocks(
    vsignals, changes: pd.DataFrame | None, provider: object, match_id: object, variant: str
) -> pd.DataFrame:
    """The ``rsi_switch_times`` + ``possession_changes`` melted blocks for one variant (H7 inputs)."""

    def _tag(block: pd.DataFrame, table: str) -> pd.DataFrame:
        n = len(block)
        lead = pd.DataFrame(
            {"provider": [provider] * n, "match_id": [match_id] * n, "variant": [variant] * n, "table": [table] * n}
        )
        return pd.concat([lead, block.reset_index(drop=True)], axis=1)

    switches = _tag(rsi_switch_times(vsignals), "rsi_switch_times")
    poss = _tag(
        changes if changes is not None else pd.DataFrame(columns=["game_id", "period_id", "time"]),
        "possession_changes",
    )
    return pd.concat([switches, poss], ignore_index=True)


def _melt_result(result, provider: object, match_id: object, variant: str) -> pd.DataFrame:
    """Stack a CoordinationResult's tables into one long frame, tagged (provider, match_id, variant, table)."""
    parts: list[pd.DataFrame] = []
    for table in RESULT_TABLES:
        tbl = getattr(result, table)
        if tbl is None or not len(tbl):
            continue
        block = tbl.copy()
        block.insert(0, "table", table)
        block.insert(0, "variant", variant)
        block.insert(0, "match_id", match_id)
        block.insert(0, "provider", provider)
        parts.append(block)
    if not parts:
        out = _empty_long_frame()
        out["match_id"] = pd.Series([match_id] * 0)
        return out
    return pd.concat(parts, ignore_index=True)


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["provider", "match_id", "variant", "table"])
