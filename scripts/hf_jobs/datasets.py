"""Declared datasets and staging records (ADR-128, spec 5.5 / 5.7, decision P2).

`datasets.toml` is human-written: repo, kind, raw + provenance (ADR-132), access, and (for open data) the pinned
commit.
`datasets.lock.json` is written ONLY by `record`, which measures the staged dataset: its revision, its
unit list, and the expected corpus identity computed with the SAME `_corpus_identity.corpus_identity` the
in-job trainer uses, over the SAME raw names (INV-9: the expectation exists before any job runs).
This module runs off-HF (operator, preflight, adopt); in-job code reads only `locked()` through
`_stages._dataset_identity` and `_driver_sources`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
DECLARED = HERE / "datasets.toml"
LOCK = HERE / "datasets.lock.json"
KINDS = ("flat_tc3", "pergame", "statsbomb_open")
ACCESS = ("mount", "download", "open")
#: ADR-132 (corpus widening spec 6.2): question 1 -- may the RAW data be redistributed?
RAW = ("redistributable", "restricted")
#: Question 2 -- can anyone reproduce an artifact from data they can obtain? CHECKED at `record` for tracking data.
PROVENANCE = ("public", "extended")


@dataclasses.dataclass(frozen=True)
class Declared:
    name: str
    repo: str
    kind: str
    raw: str
    provenance: str
    access: str
    revision: str | None
    description: str
    #: A flat dataset ships ``home_teams.json`` unless declared otherwise (the SB360 freeze-frame corpus:
    #: the receiver reads no home-team map).
    home_teams: bool = True
    #: A mount's most concurrent whole-file reads per job (the dataset read gate, `scripts._dataset_reads`):
    #: required for ``access = "mount"`` and for a ``snapshot_repo``, refused otherwise.
    read_concurrency: int | None = None
    #: An open dataset's staged, read-only HF copy of its pinned revision (HFU spec 14.7): jobs mount it.
    snapshot_repo: str | None = None


def _declared_entry(name: str, d: dict) -> Declared:
    if "visibility" in d:
        raise ValueError(f"{name}: `visibility` is retired (ADR-132); declare `raw` and `provenance`")
    if "raw" not in d or "provenance" not in d:
        raise ValueError(f"{name}: every dataset declares both `raw` and `provenance` (ADR-132)")
    if d["raw"] not in RAW:
        raise ValueError(f"{name}: raw = {d['raw']!r} is not one of {RAW}")
    if d["provenance"] not in PROVENANCE:
        raise ValueError(f"{name}: provenance = {d['provenance']!r} is not one of {PROVENANCE}")
    cap, snapshot = d.get("read_concurrency"), d.get("snapshot_repo")
    if snapshot is not None and d["access"] != "open":
        raise ValueError(f'{name}: snapshot_repo applies to access = "open" only (access={d["access"]!r})')
    if d["access"] == "mount" or snapshot is not None:
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{name}: a mounted dataset must declare read_concurrency = <int >= 1>, got {cap!r}")
    elif cap is not None:
        raise ValueError(
            f'{name}: read_concurrency applies to a mount only (access = "mount" or a snapshot_repo), '
            f"access={d['access']!r}"
        )
    return Declared(
        name,
        d["repo"],
        d["kind"],
        d["raw"],
        d["provenance"],
        d["access"],
        d.get("revision"),
        d.get("description", ""),
        bool(d.get("home_teams", True)),
        cap,
        snapshot,
    )


def declared() -> dict[str, Declared]:
    from scripts._toml import load_toml

    raw = load_toml(DECLARED)["datasets"]
    return {n: _declared_entry(n, d) for n, d in raw.items()}


def _read_lock(path: pathlib.Path = LOCK) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def locked(name: str, *, path: pathlib.Path = LOCK) -> dict:
    lock = _read_lock(path)
    if name not in lock:
        raise KeyError(
            f"dataset {name!r} has no staging record; run `python -m scripts.hf_jobs datasets record {name}`"
        )
    return lock[name]


def pergame_name(provider: str, game: str) -> str:
    """The per-game directory name the xshot/xcross extraction keys on (`repackage.flat_to_pergame`)."""
    return f"{provider}-{game}" if game else provider


def staged_units(name: str) -> list[dict]:
    return list(locked(name)["units"])


def expected_identity(name: str, naming: str) -> dict:
    """The corpus identity an artifact trained on dataset ``name`` must record (spec 5.7)."""
    rec = locked(name)
    ident = rec["identity"].get(naming)
    if ident is None:
        raise KeyError(f"{name}: no {naming!r} identity recorded (namings: {sorted(rec['identity'])})")
    return ident


def _source_provider(fs: Any, path: str) -> str:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        if "source_provider" not in pf.schema_arrow.names or pf.num_row_groups == 0:
            return "unknown"
        col = pf.read_row_group(0, columns=["source_provider"])["source_provider"]
        return str(col[0].as_py()) if len(col) else "unknown"


#: The id columns whose dtype differs across providers in the staged data (measured, ADR-128): a combined
#: multi-provider stage output mixes int and str values in them.
_ID_COLUMNS = ("game_id", "player_id", "team_id")


def _id_kind(arrow_type: str) -> str | None:
    """``"int"`` for a numeric arrow id type, ``"str"`` for a string (or dictionary-of-string) one, None for
    an all-null column (no evidence)."""
    s = str(arrow_type)
    if s == "null":
        return None
    if (
        s in ("string", "large_string")
        or s.startswith("dictionary<values=string")
        or s.startswith("dictionary<values=large_string")
    ):
        return "str"
    if s.startswith(("int", "uint", "double", "float")):
        return "int"
    raise ValueError(f"unexpected arrow id type {s!r}")


def _unit_files(rec: dict, stem: str) -> tuple[str, str]:
    if rec["kind"] == "flat_tc3":
        return f"shards/{rec['generation']}/{stem}.parquet", f"_actions/{stem}.parquet"
    return f"{stem}/frames.parquet", f"{stem}/actions.parquet"


def _unit_provider(rec: dict, unit: dict) -> str:
    return unit["stem"].split("__", 1)[0] if rec["kind"] == "flat_tc3" else unit["source_provider"]


def measure_id_kinds(name: str, rec: dict, *, repo: str, revision: str, fs: Any = None) -> dict[str, str]:
    """``{provider: "int" | "str"}`` -- the id kind each provider's staged data carries, from the footers of
    one frames file and one actions file per provider at ``revision``. A provider whose id columns disagree
    with each other refuses: no toy profile can mirror it. The toy corpora must match this record
    (`tests/scripts/hf_jobs/test_datasets_lock.py`), so the fixtures can never drift from the real data."""
    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    first: dict[str, str] = {}
    for u in rec["units"]:
        first.setdefault(_unit_provider(rec, u), u["stem"])
    out: dict[str, str] = {}
    for provider, stem in sorted(first.items()):
        kinds = set()
        for rel in _unit_files(rec, stem):
            try:
                with fs.open(f"datasets/{repo}@{revision}/{rel}", "rb") as fh:
                    schema = pq.read_schema(fh)
            except FileNotFoundError:
                continue  # a unit without an actions file
            kinds |= {_id_kind(str(schema.field(c).type)) for c in _ID_COLUMNS if c in schema.names}
        kinds.discard(None)
        if len(kinds) != 1:
            raise ValueError(f"{name}: provider {provider!r} id columns are not of ONE kind: {sorted(kinds)}")
        out[provider] = kinds.pop()
    return out


def _identities(units: list[dict]) -> dict:
    from scripts._corpus_identity import corpus_identity

    providers = [u["source_provider"] for u in units]
    flat = [u["game"] for u in units]
    pergame = [pergame_name(u["stem"].split("__", 1)[0], u["game"]) for u in units]
    return {
        "flat": corpus_identity(providers, flat, all_public=False),
        "pergame": corpus_identity(providers, pergame, all_public=False),
    }


def staged_repos(d: Declared) -> list[str]:
    """The Hub repos this repo STAGES for dataset ``d`` (corpus widening spec 6.1): its ``repo`` unless it is an
    upstream open source we do not own (``access = "open"``), and its ``snapshot_repo``. Each must be private."""
    return [r for r in ((d.repo if d.access != "open" else None), d.snapshot_repo) if r]


def require_private(repo: str, *, api: Any) -> None:
    """Refuse unless the Hub reports ``repo`` private (ADR-132: staging GS frames in a private repo is not
    redistribution). Fail-closed: an API error, or an unknown visibility, refuses too."""
    try:
        private = api.dataset_info(repo).private
    except Exception as exc:  # fail-closed: no answer is not "private"
        raise ValueError(f"{repo}: cannot confirm a private repo ({exc}); refusing (fail-closed)") from exc
    if private is not True:
        raise ValueError(f"{repo}: not a private Hub repo (private={private!r}); a staged repo must be private")


def record(
    name: str,
    *,
    policy: Any,
    api: Any = None,
    fs: Any = None,
    now: str | None = None,
    lock_path: pathlib.Path = LOCK,
) -> dict:
    """Measure the staged dataset ``name`` and write its lock entry. Off-HF, read-only on the Hub.

    ``policy`` (ADR-132) supplies ``check_provenance(name, declared, units)`` and ``public_ids(units)``: the
    launcher-only ``arm_policy`` module. It is INJECTED, never imported here, because the in-job trainers import
    this module and must stay blind to the corpus taxonomy (the I1 gate, ADR-123 Amendment B)."""
    d = declared()[name]
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if d.kind == "statsbomb_open":
        from scripts._corpus_identity import open_data_identity
        from scripts._sb_open_data import all_open_competitions, list_open_data_refs, pin_open_data_revision

        pin_open_data_revision(d.revision)
        comps = all_open_competitions()
        refs = list_open_data_refs(comps)
        keys = [r.match_id for r in refs]
        entry: dict[str, Any] = {
            "repo": d.repo,
            "revision": d.revision,
            "kind": d.kind,
            "n_units": len(keys),
            "units_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            # The corpus scope verify checks (HFU spec 14.3): the digest every event-only artifact records,
            # and the pinned (competition, season) set.
            "identity": {"open": open_data_identity(keys)},
            "competitions": sorted([int(c), int(s)] for c, s in comps),
            "recorded_at": stamp,
        }
        if d.snapshot_repo is not None:
            entry["snapshot"] = _snapshot_record(d, api)
    else:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        rev = api.dataset_info(d.repo, revision=None).sha
        files = api.list_repo_files(d.repo, repo_type="dataset", revision=rev)
        if d.kind == "flat_tc3":
            shards = sorted(f for f in files if f.startswith("shards/") and f.endswith(".parquet"))
            gens = sorted({f.split("/")[1] for f in shards})
            if len(gens) != 1 or (d.home_teams and "home_teams.json" not in files):
                raise ValueError(
                    f"{name}: declared kind flat_tc3 needs ONE shards/<gen>/ (and home_teams.json unless declared "
                    f"`home_teams = false`); saw gens={gens}"
                )
            stems = [pathlib.PurePosixPath(f).stem for f in shards]
            paths = {s: f"datasets/{d.repo}@{rev}/shards/{gens[0]}/{s}.parquet" for s in stems}
            generation = gens[0]
        elif d.kind == "pergame":
            dirs = sorted({f.split("/")[0] for f in files if f.endswith("/frames.parquet") and not f.startswith(".")})
            if not dirs:
                raise ValueError(f"{name}: declared kind pergame needs <game>/frames.parquet files")
            stems = dirs
            paths = {s: f"datasets/{d.repo}@{rev}/{s}/frames.parquet" for s in stems}
            generation = None
        else:  # pragma: no cover - guarded by KINDS
            raise ValueError(d.kind)
        if fs is None:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
        units = []
        for s in stems:
            game = s.split("__", 1)[1] if "__" in s else s
            units.append({"stem": s, "source_provider": _source_provider(fs, paths[s]), "game": game})
        entry = {
            "repo": d.repo,
            "revision": rev,
            "kind": d.kind,
            "generation": generation,
            "home_teams": d.home_teams if d.kind == "flat_tc3" else None,
            "units": units,
            "n_units": len(units),
            "identity": _identities(units),
            "public_ids": policy.public_ids(units),
            "recorded_at": stamp,
        }
        entry["provider_id_kinds"] = measure_id_kinds(name, entry, repo=d.repo, revision=rev, fs=fs)
        policy.check_provenance(name, d, units)
    for repo in staged_repos(d):  # ADR-132 (spec 6.1)
        require_private(repo, api=api if api is not None else _hf_api())
    lock = _read_lock(lock_path)
    lock[name] = entry
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _snapshot_record(d: Declared, api: Any) -> dict:
    """The lock's ``snapshot`` entry (HFU spec 14.7): the snapshot repo's current HF sha and its
    ``_snapshot.json``. Refuses a snapshot of another revision than the declared data pin."""
    import huggingface_hub

    if api is None:
        api = huggingface_hub.HfApi()
    repo = d.snapshot_repo
    if repo is None:
        raise ValueError(f"{d.name}: no snapshot_repo declared")
    rev = api.dataset_info(repo, revision=None).sha
    meta_path = huggingface_hub.hf_hub_download(repo, "_snapshot.json", repo_type="dataset", revision=rev)
    meta = json.loads(pathlib.Path(meta_path).read_text(encoding="utf-8"))
    if meta["source_revision"] != d.revision:
        raise ValueError(
            f"{d.name}: snapshot {repo}@{rev} has source_revision {meta['source_revision']}, "
            f"the pinned revision is {d.revision}"
        )
    return {
        "revision": rev,
        "source_revision": meta["source_revision"],
        "n_files": meta["n_files"],
        "bytes": meta["bytes"],
        "unreadable": sorted(meta.get("unreadable", {})),  # upstream files that never parse (HFU spec 14.7)
    }


def stage_files(ctx: Any, st: Any, keys: Sequence[str] | None) -> list[str]:
    """The dataset-relative files (or fnmatch patterns) stage ``st`` reads for its units ``keys``.

    An ItemStage: its unit refs' ``files`` plus the dataset metadata (`_driver_sources.DATASET_METADATA`),
    nothing for no key. A plain Stage: its declared ``dataset_files`` (``keys`` is ignored). ONE source for
    `fetch` and the read-audit gate (`tests/scripts/test_dataset_read_audit.py`), which proves the set
    covers every file a unit reads."""
    from scripts._driver_sources import DATASET_METADATA
    from scripts._stages import ItemStage

    if isinstance(st, ItemStage):
        refs = st.ref_map(ctx)
        files = {f for k in keys or () for f in getattr(refs[k], "files", ())}
        return sorted(files | set(DATASET_METADATA)) if files else []
    if st.dataset_files is None:
        raise ValueError(f"{st.name}: a dataset Stage must declare dataset_files")
    return sorted(set(st.dataset_files(ctx)))


def fetch_patterns(
    ctx: Any, name: str, stages: Sequence[str], shard: tuple[int, int], profile: int | None
) -> list[str]:
    """The files a ``download`` job must fetch for dataset ``name`` (resume-before-download, spec 5.3 item
    3): over every stage the job runs that reads ``name``. The FIRST stage takes the job's ``shard`` (and,
    for a profile job, only its `profile_sample`), exactly as `_stages.main` and the executor run it; a
    later grouped stage runs whole. Done and excluded units fetch nothing; a reduced plain Stage fetches
    nothing."""
    from scripts._driver import shard_select
    from scripts._stages import ItemStage, output_present, profile_sample, sorted_units, unit_states

    out: set[str] = set()
    for idx, sname in enumerate(stages):
        st = ctx.pipeline.stage(sname)
        if not st.uses_dataset or ctx.params.get(st.dataset_param) != name:
            continue
        if isinstance(st, ItemStage):
            keys = sorted_units(ctx, st)
            if idx == 0:
                keys = shard_select(keys, shard[0], shard[1])
                if profile:
                    keys = profile_sample(keys, profile)
            states = unit_states(ctx, st, keys, verify=False)
            out.update(stage_files(ctx, st, [k for k in keys if states[k] not in ("done", "excluded")]))
        elif not output_present(ctx, sname, verify=False):
            out.update(stage_files(ctx, st, None))
    return sorted(out)


def fetch(
    name: str,
    *,
    dest: pathlib.Path,
    module: str,
    stages: Sequence[str],
    shard: str,
    profile: int | None,
    store: str,
    data_root: str,
    args: list[str],
) -> list[str]:
    """Download ONLY the files this job's units will read (`fetch_patterns`) into ``dest``."""
    import importlib

    from scripts._provenance import code_identity, git_provenance
    from scripts._stages import _parse_shard, build_parser, make_context, open_store

    pipeline = importlib.import_module(module).PIPELINE
    ns = build_parser(pipeline).parse_args([*args, "--plan", "--store", store, "--data-root", data_root])
    params = {p.name: getattr(ns, p.name) for p in pipeline.params}
    git = git_provenance()
    ctx = make_context(
        pipeline,
        params,
        {**git, "code_identity": code_identity(git)},
        store=open_store(store),
        data_root=pathlib.Path(data_root),
    )
    patterns = fetch_patterns(ctx, name, stages, _parse_shard(shard), profile)
    if patterns:
        from huggingface_hub import snapshot_download

        rec = locked(name)
        snapshot_download(
            rec["repo"], repo_type="dataset", revision=rec["revision"], local_dir=str(dest), allow_patterns=patterns
        )
    return patterns


def main(argv: list[str] | None = None) -> int:
    from scripts.hf_jobs.liveness import start_job_watch

    start_job_watch("fetch")  # in an HF job only (the download pre-step, HFU spec §11.1)
    ap = argparse.ArgumentParser(prog="python -m scripts.hf_jobs datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("name")
    k = sub.add_parser("id-kinds", help="measure each provider's id kind at the LOCKED revision into the lock")
    k.add_argument("name")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    for flag in ("--dest", "--module", "--stages", "--shard", "--store", "--data-root"):
        f.add_argument(flag, required=True)
    f.add_argument("--profile", type=int, default=None)
    f.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        print(json.dumps(locked(a.name), indent=2))
    elif a.cmd == "id-kinds":
        lock = _read_lock()
        rec = lock[a.name]
        rec["provider_id_kinds"] = measure_id_kinds(a.name, rec, repo=rec["repo"], revision=rec["revision"])
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(rec["provider_id_kinds"]))
    else:
        rest = [x for x in a.rest if x != "--"]
        got = fetch(
            a.name,
            dest=pathlib.Path(a.dest),
            module=a.module,
            stages=[s for s in a.stages.split(",") if s],
            shard=a.shard,
            profile=a.profile,
            store=a.store,
            data_root=a.data_root,
            args=rest,
        )
        print(f"fetched {len(got)} pattern(s) for {a.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
