# Campaign Throughput, Reduce I/O and Content-Addressed Reuse — Implementation Plan (Cycle 1b)

> **For the implementing session:** execute inline with superpowers:executing-plans. House rules override the
> skill defaults: NO subagents, NO worktrees, NO per-task commits. There is ONE code commit (amended only with an
> explicit owner yes) and ONE artifact commit. Every `git commit`, `git push`, PR update, Hub publish, image push and
> tag push needs its own explicit owner yes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make an ADR-128 campaign fast and resumable at scale: a linear reduce, DAG-parallel critical-path
scheduling, a makespan planner, sound content-addressed reuse across commits with attestation, a prebuilt job
image, and model-based acceptance bars.

**Architecture:** The stage contract (`scripts/_stages.py`) gains a linear, verified, concurrent reduce and a
per-stage code identity built from a locked closure. The controller (`scripts/hf_jobs/campaign.py`) becomes
DAG-parallel with action ownership, a critical-path grant order and a makespan planner. Stores become
content-addressed per `{job}-{tier}`; ADR-129 adopts by token equality plus attestation and a canary.

**Tech stack:** Python 3.10 (local `.venv`) / 3.12 (HF image), pandas, pyarrow, `huggingface_hub` 2.2, `uv==0.9.28`,
GitHub Actions, GHCR, HF Jobs `cpu-upgrade`.

**Revision:** r7 — Cycle 1 cut (owner, 2026-10-21): branch off `main`, the E-5 export also writes stage costs; r6 — both reviewers APPROVE r5; r6 applies B's r4 CONSIDER (an incremental fill) only; r5 = owner ruling 12; addresses plan reviews r1–r4; r2 addressed plan review r1 (A APPROVE, 0 findings; B REQUEST CHANGES, 1 BLOCKING, 7 SHOULD FIX,
7 CONSIDER; change log at the end).

**Spec (approved r10, 2026-10-21; owner rulings 11 and 12 recorded in r11/r12):** `D:\Development\_drafts\2026-10-21-campaign-throughput-design.md`. After Task 1
it lives at `docs/superpowers/specs/2026-10-21-campaign-throughput-design.md`. Read the spec section each task cites.
The spec's sections are cited as "S5.9" and so on.

## Global Constraints

- Branch `feat/compute-campaign-throughput`, created off `main` AFTER PR #288 (the Cycle 1 code commit, amended with
  the Cycle 1 outcome docs) is squash-merged (owner ruling 2026-10-21: the Cycle 1 artifact work is cut, so the tree
  is free once that merge lands). Until then: write NOTHING in the repo folder.
- HF Jobs flavor `cpu-upgrade` only (INV-4). Exploration spend on `cpu-upgrade` is pre-approved; every publish is not.
- Never pip-install into `.venv` or `.venv-hfjobs`. `uv run` needs `--no-project`.
- Bash commands over 30 s use `run_in_background`, or `timeout <= 30000`. Never `gh run watch`; use Monitor.
- Local verification = targeted tests + `uvx ruff@0.15.7 check silly_kicks/ tests/ scripts/` +
  `uvx ruff@0.15.7 format --check silly_kicks/ tests/ scripts/` + `pyright` scoped to changed files. CI is the
  full-suite gate. Never run the full local suite.
- Do not edit the working tree while the controller or a parity test runs (code identity, S2.7).
- Prose (docs, comments, commit text) follows ASD-STE100. Commit trailer `Co-Authored-By: Claude Opus 5.5
  <noreply@anthropic.com>`; no `Claude-Session:` trailer.
- No version bump in the code commit (S10). ADR number ADR-129: re-confirm on `origin` at the commit gate.
- Spec constants (copy exactly): `concurrency_cap = 20`; store read cap default 4; the declared bars of S7.2
  (1.25, 0.75–1.25, 1.5, 1.5, 1.1, all); the 5 % fewest-jobs rule (S5.4); the 10 % store-probe rule (S7.1);
  attempt id `%Y%m%dT%H%M%SZ_<4 hex>`; action id `%Y%m%dT%H%M%SZ_<8 hex>`; R8 fallback `<stages>__<action id>`;
  startup probe = 5 jobs, median.
- Reviewer reports go to `D:\Development\_reviews\`.

## Engineering resolutions found while planning (for the plan reviewers and the owner)

These come from reading the code at `74d57ba`. Each one keeps the spec's intent. E-1 and E-2 change a mechanism the
spec names; the owner RULED on both on 2026-10-21 (spec owner ruling 11), and the spec carries the rulings.

- **E-1 (owner ruling 11a) — the image build: the house publish pattern plus provenance.** A `workflow_dispatch`
  workflow can only start when the workflow file exists on the DEFAULT branch, and this branch does not reach `main`
  before the campaign. The image workflow therefore mirrors `publish.yml` (a tag trigger and a protected
  environment, `publish.yml:3-5`, `:62`):
  1. it triggers on the push of a tag `job-image/env-<12 hex>` (the derived tag of S5.12), from the tagged commit;
     it refuses a tag that differs from the derived tag, and an image tag that exists;
  2. its publish job runs in `environment: job-image`, whose required reviewer is the owner: nothing publishes
     until the owner approves in the Actions UI (the recorded approval is the publish yes);
  3. a repository ruleset restricts creation of `job-image/*` tags to the owner;
  4. `actions/attest-build-provenance` signs a SLSA build-provenance attestation for the pushed digest, and the
     preflight `check_image` verifies it (`gh attestation verify`), not only the digest;
  5. the owner sets the GHCR package public once (it holds only public third-party packages).
  The tag prefix never starts with `v`, so it cannot trigger the PyPI publish.
- **E-2 (owner ruling 11b) — the S7.1 item 3 reuse demonstration runs the REAL path.** The first `cycle1b` campaign
  attests every token fresh, so without this step the ancestor-attestation path and the canary would never run on
  real HF in this cycle. Two steps, both at the Cycle 1b ARTIFACT commit (no extra commit):
  1. before the push: `campaign plan --with-replica 1` gives the S7.1 item 3 table (per stage: reused, or re-run
     because it declares an adopted weight file) and the cost of any re-run;
  2. after the owner-approved push and a green CI: `campaign launch --with-replica 1` at that HEAD. Every attested
     stage runs one canary unit on real HF, reading its upstream from the main store, and must be byte-identical;
     the main run takes zero units and the replica skips every attested stage. A stage that re-runs because it
     declares an adopted weight file launches only after the owner sees its cost. A canary mismatch quarantines the
     token and is an owner finding before any merge.
- **E-3 — the HF `name` field is a label.** `launch.py:33-35` records that HF validates the job name as `labels.name`.
  So a job carries 8 labels today (7 + `name`) and 9 with `action`. The R8 probe (Task 0.2) launches a job with 9.
- **E-4 — every reduce already runs in a subprocess** (`executor reduce`, `executor.py:243-258`). "In-job" (S5.2,
  S7.2) therefore means "inside a unit job" as opposed to "in a separate reduce job". The launcher marks a separate
  reduce job with `SILLY_KICKS_ACTION_KIND=reduce` (a new `neutral` variable in the S5.9 registry).
- **E-5 — per-unit history for the first Cycle 1b campaign.** The committed Cycle 1 summary is written by `74d57ba`
  code; owner ruling 2026-10-21 cut Cycle 1's artifact commit, so no Cycle 1 summary is committed at all, and S5.7's
  history has nothing to read for `cycle1b`. Task 22 (Step 3, before any edit of the image lock) exports the
  per-unit costs from the Cycle 1 stores (read-only) into `docs/research/hf_campaigns/cycle1.per_unit.json` on a clean
  tree, and the amend commits it beside the image lock. From Cycle 1b on, `campaign summary` writes per-unit costs
  into the summary itself.
- **E-6 — the preflight image check.** `preflight.check_lock` resolves the image on Docker Hub
  (`preflight.py:58-91`). The prebuilt image lives in GHCR, so `check_lock` keeps only the `uv lock --check` half, and
  the new `check_image` resolves the locked digest in the locked registry.

---

## Phase 0 — measurements that need no repo change (run while Cycle 1 finishes)

All Phase 0 scripts live in the session scratchpad, never in the repo. They read the repo; they write nothing there.

### Task 0.1: Time the controller reads (S2.7, S5.11 first step)

**Files:** scratchpad `time_reads.py` only.

- [ ] **Step 1: Write the timing script**

```python
# scratchpad/time_reads.py — run from the repo root with .venv-hfjobs/Scripts/python
import json, sys, time
sys.path.insert(0, ".")
from huggingface_hub import HfApi
from scripts.hf_jobs import campaign as cp
from scripts.hf_jobs.store import BucketStore

c = cp.load_campaign("scripts/hf_jobs/campaigns/cycle1.toml")
commit = cp._head()
inst = next(i for i in cp.instances(c, commit) if i.job == "win_probability")
store = BucketStore(c.bucket, inst.run_key, api=HfApi())
t0 = time.monotonic(); plan = cp.read_plan(inst, bucket=c.bucket); t1 = time.monotonic()
cp.live_history(plan, store, {}); t2 = time.monotonic()
print(json.dumps({"read_plan_s": round(t1 - t0, 1), "live_history_s": round(t2 - t1, 1)}))
```

- [ ] **Step 2: Run it** (background; it takes about 30 minutes)

Run: `.venv-hfjobs/Scripts/python <scratchpad>/time_reads.py` with `run_in_background: true`.
Expected: one JSON line. Record both numbers in the S5.11 evidence note (Task 23 summary).

### Task 0.2: Probe the HF label count (R8, E-3)

- [ ] **Step 1: Launch one job with 9 labels** (8 named labels + the job name)

```python
# scratchpad/label_probe.py
from huggingface_hub import HfApi
labels = {k: "probe" for k in ("campaign", "run_key", "job", "variant", "stage", "shard", "commit")}
labels["action"] = "20261021T143000Z_1a2b3c4d"
info = HfApi().run_job(image="python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0",
                       command=["python", "-c", "print('ok')"], flavor="cpu-upgrade", timeout=600,
                       labels=labels, name="label-probe-9")
print(info.id)
```

- [ ] **Step 2: Record the result.** Accepted (the job reaches COMPLETED): R8 is closed, the `action` label stays.
  Refused (HTTP 400): Task 6 implements the R8 fallback `<stages>__<action id>` in the `stage` label instead of
  the `action` label. Record the job id or the 400 text.

### Task 0.3: Store read probe (S7.1 item 1)

The Cycle 1 win_probability extract generation holds about 8000 small files. The probe mounts that bucket prefix
READ-ONLY and reads it at caps 1, 2, 4, 8 and 16. It must not start while a win_probability reduce job is running
on that store, and it never writes there.

- [ ] **Step 1: Write and launch the probe job**

```python
# scratchpad/store_probe.py
from huggingface_hub import HfApi, Volume
# A RESULT read, as the reduce does it (Task 2 `read_results`): the data file, its `.sha256` sidecar, the digest
# check and the parquet decode -- not one bare file read (r1 B CCTP-PLAN-06).
CODE = r'''
import concurrent.futures as cf, hashlib, io, json, pathlib, sys, time
import pyarrow.parquet as pq
gen = next(p for p in pathlib.Path(sys.argv[1]).iterdir() if p.is_dir())
keys = sorted(p.name[:-8] for p in gen.iterdir() if p.name.endswith(".parquet") and not p.name.startswith("_"))
out = {"n_results": len(keys)}
def one(k):
    try:
        data = (gen / f"{k}.parquet").read_bytes()
        side = (gen / f"{k}.sha256").read_bytes().decode().strip()
        if hashlib.sha256(data).hexdigest() != side:
            return -2
        pq.read_table(io.BytesIO(data))
        return len(data)
    except OSError:
        return -1
for cap in (1, 2, 4, 8, 16):
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(cap) as ex:
        got = list(ex.map(one, keys))
    out[str(cap)] = {"wall_s": round(time.monotonic() - t0, 2), "errors": sum(1 for g in got if g == -1),
                     "bad_digest": sum(1 for g in got if g == -2)}
print(json.dumps(out))
'''
import pathlib as _pl, re as _re, shlex
lock = _pl.Path("uv.lock").read_text(encoding="utf-8")
pyarrow = _re.search(r'name = "pyarrow"\nversion = "([^"]+)"', lock).group(1)  # the locked version
rk = "cycle1-74d57ba-win_probability-public"
# Mount the WHOLE bucket read-only and read the run-key prefix under it (no sub-path form is assumed).
info = HfApi().run_job(image="python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0",
    command=["bash", "-c", f"pip install -q pyarrow=={pyarrow} && python -c {shlex.quote(CODE)} /scratch/{rk}/win_probability/extract"],
    flavor="cpu-upgrade", timeout=7200, name="store-read-probe",
    volumes=[Volume(type="bucket", source="silly-kicks/hf-refit-scratch", mount_path="/scratch", read_only=True)])
print(info.id)
```

- [ ] **Step 2: Apply the S7.1 rule.** The safe cap = the largest cap with `errors == 0` and a wall within 10 % of
  the best wall. Record `per_result_read_s = wall_at_the_safe_cap / n_results`: the wall per RESULT at that cap
  (data + sidecar + digest + decode, with the cap's concurrency already in the wall; never divide by the cap again).
  That is the S5.2 cost input `Constraints.store_read_s`. Both values go into `datasets.toml [store]` in Task 3, with
  this evidence in a comment.

### Task 0.4: Confirm the `uv sync --check` spelling at `uv==0.9.28` (S5.12)

- [ ] **Step 1:** Run `uvx uv@0.9.28 sync --help` (timeout 30000) and find the flag that checks without changing the
  environment (`--check`, or `--locked --dry-run`). Record the exact spelling for Task 18.

---

## Phase 1 — branch and drafts

### Task 1: Create the sub-branch and move the drafts in

**Precondition:** PR #288 is squash-merged to `main` (owner ruling 2026-10-21: Cycle 1's Tasks 24–28 are superseded
by this cycle; ADR-128 "Cycle 1 outcome").

**Files:**
- Create: `docs/superpowers/specs/2026-10-21-campaign-throughput-design.md` (moved from `D:\Development\_drafts\`)
- Create: `docs/superpowers/plans/2026-10-21-campaign-throughput-plan.md` (moved from `D:\Development\_drafts\`)

- [ ] **Step 1:** `git switch main && git pull --ff-only && git switch -c feat/compute-campaign-throughput`
- [ ] **Step 2:** Copy both drafts into the paths above (byte copy). Delete nothing in `_drafts` (the reviewers'
  reports cite it).
- [ ] **Step 3:** Re-confirm ADR-129 is free: `git fetch origin && git grep -l "ADR-129" $(git branch -r --format='%(refname:short)' | grep -v HEAD) -- docs/superpowers/adrs || true`. Expected: no file. If one
  exists, stop and ask the owner for the number.

---

## Phase 2 — the stage contract (`scripts/_stages.py`, the read gate, the ledger)

### Task 2: One listing, one verified read, write after verify (S5.1, INV-3, INV-11)

**Files:**
- Modify: `scripts/_stages.py` (`unit_states`, `read_result`, `_verified`, `reduce_stage`; new `UnitStates`,
  `_read_verified`, `_decode`, `read_results`, `quarantine_corrupt`)
- Create: `tests/scripts/test_reduce_store_access.py`
- Modify: `tests/_scale_guarded.py` (register the guard in `SCALE_GUARDED` at `:19`, as its other entries do)

**Interfaces:**
- Produces: `class UnitStates(dict[str, str])` with attribute `names: frozenset[str]`;
  `unit_states(ctx, st, keys, *, verify) -> UnitStates`;
  `read_result(store, base, *, names: Collection[str] | None = None) -> Any`;
  `read_results(store, gen, keys, *, names, cap: int) -> tuple[dict[str, Any], list[str]]` (results in `keys`
  order, corrupt keys);
  `quarantine_corrupt(store, gen, key, names) -> None`;
  `CORRUPT_DIR = ".corrupt"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_reduce_store_access.py
"""INV-11: a reduce lists each generation once and reads each result file once, verified, before it writes."""

from __future__ import annotations

import collections

import pandas as pd
import pytest

from scripts import _stages as s


class CountingStore(s.LocalStore):
    def __init__(self, root):
        super().__init__(root)
        object.__setattr__(self, "calls", collections.Counter())

    def names(self, rel_dir):
        self.calls[("names", rel_dir)] += 1
        return super().names(rel_dir)

    def read_bytes(self, rel):
        self.calls[("read", rel)] += 1
        return super().read_bytes(rel)


def _pipeline(n: int, kind: str) -> s.Pipeline:
    def work(k, ctx):
        if kind == "fileset":
            return s.FileSet({"a.bin": k.encode(), "b.bin": (k * 2).encode()})
        # real mixed id types: int ids from one provider, str ids from another (Cycle 1 defect class)
        gid = int(k) if int(k) % 2 else f"J{k}"
        return pd.DataFrame({"game_id": [gid], "x": [float(k)]})

    def reduce(results, ctx):
        if kind == "fileset":
            return {k: sorted(v.files) for k, v in results.items()}
        return s.combine_frames(list(results.values()), ids=("game_id",))

    st = s.Stage(name="a", units=lambda c: [s.int_key(i) for i in range(n)], work=work, reduce=reduce)
    return s.Pipeline(name="t", module="tests.scripts._toy_pipeline", stages=(st,))


def _ctx(tmp_path, n, kind):
    store = CountingStore(tmp_path)
    prov = {"commit": "c" * 40, "tree_state": "clean", "code_identity": "c" * 40}
    ctx = s.make_context(_pipeline(n, kind), {}, prov, store=store, data_root=None)
    for k in s.sorted_units(ctx, ctx.pipeline.stage("a")):
        s.run_unit(ctx, ctx.pipeline.stage("a"), k, worker="w0")
    store.calls.clear()
    return ctx, store


@pytest.mark.parametrize("kind", ["parquet", "fileset"])
@pytest.mark.parametrize("n", [50, 100, 200])
def test_reduce_lists_once_and_reads_each_file_once(tmp_path, n, kind):
    ctx, store = _ctx(tmp_path, n, kind)
    s.reduce_stage(ctx, "a")
    gen = ctx.generation("a")
    assert store.calls[("names", gen)] == 1
    reads = {rel: c for (op, rel), c in store.calls.items() if op == "read"}
    assert max(reads.values()) == 1, {r: c for r, c in reads.items() if c > 1}
    # every result file was read: data + sidecar (+ 2 members for a fileset)
    per_unit = 2 + (2 if kind == "fileset" else 0)
    assert sum(1 for r in reads if r.startswith(gen + "/0")) == n * per_unit


def test_a_corrupt_result_writes_nothing_and_returns_its_unit_to_pending(tmp_path):
    ctx, store = _ctx(tmp_path, 20, "parquet")
    gen = ctx.generation("a")
    victim = tmp_path / gen / "0007.parquet"
    victim.write_bytes(victim.read_bytes()[:-3] + b"xyz")
    with pytest.raises(s.StageIncompleteError, match="0007"):
        s.reduce_stage(ctx, "a")
    names = store.names(gen)
    assert "_summary.json" not in names and "_inputs.json" not in names
    assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0007"], verify=False)["0007"] == "pending"
    assert any(p.name.startswith("0007") for p in (tmp_path / gen / s.CORRUPT_DIR).rglob("*"))
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0007", worker="w0")
    s.reduce_stage(ctx, "a")
    assert "_output.parquet" in store.names(gen)


def test_unit_states_carries_its_listing(tmp_path):
    ctx, store = _ctx(tmp_path, 3, "parquet")
    states = s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000", "0001", "0002"], verify=False)
    assert dict(states) == {"0000": "done", "0001": "done", "0002": "done"}
    assert "0000.parquet" in states.names and isinstance(states.names, frozenset)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/scripts/test_reduce_store_access.py -q -p no:cacheprovider`
Expected: FAIL (`names` called n+1 times; reads repeat; `CORRUPT_DIR` missing).

- [ ] **Step 3: Implement**

```python
# scripts/_stages.py — replace `_verified`, `read_result`, `unit_states`; add the helpers below

CORRUPT_DIR = ".corrupt"


class UnitStates(dict):
    """``{key: state}`` plus the generation listing it was computed from (``names``), so a caller that
    reads results next reuses ONE listing (INV-11)."""

    names: frozenset[str] = frozenset()


def _read_verified(store: Store, base: str, suffix: str) -> tuple[bytes, dict[str, bytes]] | None:
    """The result bytes and (for a FileSet) its member bytes, each read ONCE and checked against its digest;
    None when any part is absent or fails its digest."""
    data = store.read_bytes(base + suffix)
    side = store.read_bytes(base + ".sha256")
    if data is None or side is None or _sha(data) != side.decode("ascii").strip():
        return None
    members: dict[str, bytes] = {}
    if suffix == ".fileset.json":
        for name, want in json.loads(data)["files"].items():
            payload = store.read_bytes(f"{base}.files/{name}")
            if payload is None or _sha(payload) != want:
                return None
            members[name] = payload
    return data, members


def _verified(store: Store, base: str, suffix: str) -> bool:
    return _read_verified(store, base, suffix) is not None


def _decode(suffix: str, data: bytes, members: Mapping[str, bytes]) -> Any:
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(io.BytesIO(data))
    if suffix == ".json":
        return json.loads(data)
    return FileSet(dict(members))


def read_result(store: Store, base: str, *, names: Collection[str] | None = None) -> Any:
    """The verified result at ``base``, or ``MISSING``. ``names`` is the caller's listing of the result's
    directory; without it the directory is listed here (one listing per call)."""
    rel_dir, _, stem = base.rpartition("/")
    suffix = _present_suffix(set(names) if names is not None else store.names(rel_dir), stem)
    if suffix is None:
        return MISSING
    got = _read_verified(store, base, suffix)
    return MISSING if got is None else _decode(suffix, *got)


def quarantine_corrupt(store: Store, gen: str, key: str, names: Collection[str]) -> None:
    """Move every file of unit ``key`` (result, sidecar, FileSet members) under ``<gen>/.corrupt/<utc>/`` --
    copy, then delete -- so the unit is ``pending`` again (INV-3) and the bytes stay for diagnosis."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = f"{gen}/{CORRUPT_DIR}/{stamp}"
    own = [n for n in names if n == f"{key}.sha256" or any(n == f"{key}{sfx}" for sfx in _RESULT_SUFFIXES)]
    sfx = _present_suffix(set(names), key)
    if sfx == ".fileset.json":
        raw = store.read_bytes(f"{gen}/{key}{sfx}")
        for member in json.loads(raw)["files"] if raw else ():
            rel = f"{key}.files/{member}"
            payload = store.read_bytes(f"{gen}/{rel}")
            if payload is not None:
                store.write_bytes(f"{dest}/{rel}", payload)
                store.delete(f"{gen}/{rel}")
    for name in own:
        payload = store.read_bytes(f"{gen}/{name}")
        if payload is not None:
            store.write_bytes(f"{dest}/{name}", payload)
        store.delete(f"{gen}/{name}")


def read_results(
    store: Store, gen: str, keys: Sequence[str], *, names: Collection[str], cap: int
) -> tuple[dict[str, Any], list[str]]:
    """Read and verify every result of ``keys`` through a bounded thread pool (``cap`` readers, each under the
    store read gate). Returns the results in ``keys`` order -- never completion order (INV-1) -- and the
    keys whose bytes failed their digest."""
    import concurrent.futures

    from scripts._dataset_reads import store_read_slot

    listing = set(names)

    def one(k: str) -> tuple[str, Any]:
        sfx = _present_suffix(listing, k)
        if sfx is None:
            return k, MISSING
        with store_read_slot():
            got = _read_verified(store, f"{gen}/{k}", sfx)
        return k, (MISSING if got is None else _decode(sfx, *got))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(cap))) as ex:
        got = dict(ex.map(one, keys))
    corrupt = [k for k in keys if got[k] is MISSING]
    return {k: got[k] for k in keys if got[k] is not MISSING}, corrupt


def unit_states(
    ctx: RunContext, st: AnyStage, keys: Sequence[str], *, verify: bool, names: Collection[str] | None = None
) -> UnitStates:
    """``done`` | ``excluded`` | ``failed`` | ``pending`` per key, plus the listing (``.names``). ``verify``
    re-hashes every present result (the in-job resume check, INV-3); a failing one is deleted with a warning
    and is pending. ``names`` is a listing the caller already holds (`plan`, Task 5)."""
    gen = ctx.generation(st.name)
    listing = set(names) if names is not None else ctx.store.names(gen)
    out = UnitStates()
    deleted = False
    for k in keys:
        suffix = _present_suffix(listing, k)
        if suffix is not None and f"{k}.sha256" in listing:
            if not verify or _verified(ctx.store, f"{gen}/{k}", suffix):
                out[k] = "done"
                continue
            warnings.warn(f"checksum failure for {gen}/{k}{suffix}; deleting it and redoing the unit", stacklevel=2)
            for sfx in (suffix, ".sha256"):
                ctx.store.delete(f"{gen}/{k}{sfx}")
            deleted = True
        if f"{k}.excluded.json" in listing:
            out[k] = "excluded"
        elif f"{k}.failed.json" in listing:
            out[k] = "failed"
        else:
            out[k] = "pending"
    out.names = frozenset(ctx.store.names(gen) if deleted else listing)
    return out
```

Then rewrite the read part of `reduce_stage`:

```python
def reduce_stage(ctx: RunContext, name: str) -> None:
    """Write the stage output from every unit result, in ``reduce_order``. Reads and verifies EVERY result
    first (one listing, one read per file, INV-11); writes the summary, inputs and output only when all pass.
    A corrupt result moves to ``.corrupt/`` and its unit is pending again. Idempotent and atomic."""
    from scripts._dataset_reads import store_read_cap

    st = ctx.pipeline.stage(name)
    keys = sorted_units(ctx, st)
    states = unit_states(ctx, st, keys, verify=False)
    pending = [k for k, v in states.items() if v == "pending"]
    failed = [k for k, v in states.items() if v == "failed"]
    if pending:
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: {len(pending)} unit(s) pending, e.g. {pending[:3]}")
    if st.max_failures is not None and len(failed) > st.max_failures:
        raise StageIncompleteError(
            f"{ctx.pipeline.name}/{name}: {len(failed)} unit(s) failed (tolerance {st.max_failures}): {failed[:5]}"
        )
    gen = ctx.generation(name)
    done = sorted((k for k, v in states.items() if v == "done"), key=st.reduce_order or (lambda k: k))
    started = time.time_ns()
    results, corrupt = read_results(ctx.store, gen, done, names=states.names, cap=store_read_cap() or 4)
    if corrupt:
        for k in corrupt:
            warnings.warn(f"checksum failure for {gen}/{k}; moving it to {CORRUPT_DIR}/ (the unit re-runs)", stacklevel=2)
            quarantine_corrupt(ctx.store, gen, k, states.names)
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: corrupt result(s) {corrupt[:5]} moved; re-run them")
    excluded = {}
    for k in keys:
        if states[k] == "excluded":
            raw = ctx.store.read_bytes(f"{gen}/{k}.excluded.json")
            excluded[k] = json.loads(raw)["reason"] if raw else "unknown"
    summary = {  # unchanged keys
        "stage": name, "token": ctx.tokens[name], "n_units": len(keys), "n_done": len(done),
        "n_excluded": len(excluded), "n_failed": len(failed), "failed_keys": failed, "excluded": excluded,
    }
    ctx.store.write_bytes(f"{gen}/_summary.json", _canon_json(summary))
    ctx.store.write_bytes(f"{gen}/_inputs.json", _canon_json(ctx.token_inputs[name]))
    output = st.reduce(results, ctx) if st.reduce is not None else summary
    write_result(ctx.store, f"{gen}/{_OUTPUT}", output)
    write_reduce_record(ctx, name, started_ns=started, n_results=len(results))  # Task 4
```

(`write_reduce_record` lands in Task 4; until then omit that last line.) Add `from collections.abc import Collection`.

Register the guard in `tests/_scale_guarded.py` (`SCALE_GUARDED`, `:19`) following its existing entries, with
`assert_subquadratic_growth` over `reduce_stage` at n = 50/100/200 counting `names` + `read_bytes` calls.

- [ ] **Step 4: Update the other `read_result` callers that loop over units** (S6.1): `scripts/build_territory_ranking_census.py`,
  `scripts/train_ghost_gk.py`, `scripts/train_xcross_attempt.py`, `scripts/hf_jobs/adopt.py`. For each call inside a
  loop over keys, list the directory once before the loop and pass `names=`. Find them with
  `git grep -n "read_result(" -- scripts`.

- [ ] **Step 5: Run the targeted tests**

Run: `.venv/Scripts/python -m pytest tests/scripts/test_reduce_store_access.py tests/scripts/test_stage_contract.py tests/scripts/test_combine_frames.py tests/scripts/test_extract_reduce_mixed_ids.py tests/scripts/hf_jobs/test_executor.py tests/test_scale_guards.py tests/_scale_guarded.py -q -p no:cacheprovider`
(background if over 30 s). Expected: PASS.

### Task 3: The store read gate (S5.1)

**Files:**
- Modify: `scripts/_dataset_reads.py`; `scripts/hf_jobs/datasets.toml`; `scripts/hf_jobs/datasets.py`;
  `scripts/hf_jobs/launch.py` (`job_env`)
- Test: `tests/scripts/test_dataset_reads.py`, `tests/scripts/hf_jobs/test_launch_typed.py`

**Interfaces:**
- Produces: `STORE_READ_CAP_ENV = "SILLY_KICKS_STORE_READ_CAP"`; `store_read_cap() -> int | None`;
  `store_read_slot()` (context manager); `datasets.store_settings() -> StoreSettings(read_concurrency: int,
  per_result_read_s: float)`.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/test_dataset_reads.py — add
def test_the_store_gate_is_separate_from_the_dataset_gate(tmp_path, monkeypatch):
    from scripts import _dataset_reads as dr

    monkeypatch.setenv(dr.READ_CAP_ENV, "1")
    monkeypatch.setenv(dr.STORE_READ_CAP_ENV, "1")
    monkeypatch.setenv(dr.READ_GATE_DIR_ENV, str(tmp_path))
    with dr.read_slot():  # a held dataset slot never blocks a store read
        with dr.store_read_slot():
            pass
    assert dr.store_read_cap() == 1


def test_store_read_cap_refuses_a_non_positive_value(monkeypatch):
    import pytest

    from scripts import _dataset_reads as dr

    monkeypatch.setenv(dr.STORE_READ_CAP_ENV, "0")
    with pytest.raises(ValueError, match="positive"):
        dr.store_read_cap()
```

```python
# tests/scripts/hf_jobs/test_launch_typed.py — add
def test_every_job_sets_the_store_read_cap():
    from scripts._dataset_reads import STORE_READ_CAP_ENV
    from scripts.hf_jobs import datasets, launch

    env = launch.job_env(_spec())  # the file's existing JobSpec helper
    assert env[STORE_READ_CAP_ENV] == str(datasets.store_settings().read_concurrency)
```

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/test_dataset_reads.py tests/scripts/hf_jobs/test_launch_typed.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Implement.** In `_dataset_reads.py`, generalize the cap and slot by a gate name:

```python
STORE_READ_CAP_ENV = "SILLY_KICKS_STORE_READ_CAP"


def _cap(env: str) -> int | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError(f"{env} must be a positive integer, got {raw!r}") from None
    if cap < 1:
        raise ValueError(f"{env} must be a positive integer, got {raw!r}")
    return cap


def read_cap() -> int | None:
    return _cap(READ_CAP_ENV)


def store_read_cap() -> int | None:
    return _cap(STORE_READ_CAP_ENV)


_HELD = {"dataset": threading.local(), "store": threading.local()}


@contextlib.contextmanager
def _slot(kind: str, cap: int | None) -> Iterator[None]:
    held = _HELD[kind]
    if cap is None or getattr(held, "depth", 0):
        held.depth = getattr(held, "depth", 0) + 1
        try:
            yield
        finally:
            held.depth -= 1
        return
    gate = _gate_dir() / kind
    gate.mkdir(parents=True, exist_ok=True)
    fh = _acquire(gate, cap)
    held.depth = 1
    try:
        yield
    finally:
        held.depth = 0
        try:
            _unlock(fh)
        finally:
            fh.close()


def read_slot():
    """One dataset-mount read slot (unchanged contract)."""
    return _slot("dataset", read_cap())


def store_read_slot():
    """One store (bucket mount) read slot, a gate SEPARATE from the dataset gate (S5.1)."""
    return _slot("store", store_read_cap())
```

The dataset gate's lock files move from `<gate dir>/slot-i.lock` to `<gate dir>/dataset/slot-i.lock`. Nothing reads
the old path outside this module (`git grep -n "slot-" -- scripts tests`).

In `datasets.toml` add (values from Task 0.3; the comment carries the evidence):

```toml
# ADR-128 Cycle 1b (S5.1, S7.1): the job-wide cap on concurrent reads from the bucket STORE mount, and the
# measured wall per RESULT read at that cap (data + sidecar + digest + parquet decode; S5.2). Measured <date> by the store read probe (job <id>):
# caps 1/2/4/8/16 -> <walls>, errors <...>. Safe cap = the largest with no error, within 10 % of the best.
[store]
read_concurrency = <from Task 0.3>
per_result_read_s = <from Task 0.3>
```

In `datasets.py` add `StoreSettings` (a frozen dataclass) and `store_settings()` reading `[store]` from `DECLARED`.
In `launch.job_env` add `env[STORE_READ_CAP_ENV] = str(datasets.store_settings().read_concurrency)`.

- [ ] **Step 4:** Re-run Step 2's command. Expected: PASS.

### Task 4: Per-attempt identity, job phases, cost aggregates and reduce records (S5.2, S5.3, S5.11, S7.2)

**Files:**
- Create: `scripts/hf_jobs/phases.py`
- Modify: `scripts/hf_jobs/launch.py` (`label_of`, `parse_stage_label`, `labels_for`, `JobSpec`, `job_env`,
  `job_command`), `scripts/_stages.py` (`run_unit`, `write_reduce_record`), `scripts/hf_jobs/executor.py`
  (`_run`, `worker_main`), `scripts/hf_jobs/ledger.py` (`job_startups`, new `cost_aggregates`,
  `reduce_records`)
- Test: `tests/scripts/hf_jobs/test_ledger.py`, `tests/scripts/hf_jobs/test_launch_typed.py`,
  `tests/scripts/hf_jobs/test_executor.py`

**Interfaces:**
- Produces (in `scripts/hf_jobs/phases.py`):
  `ATTEMPT_ENV = "SILLY_KICKS_ATTEMPT_ID"`, `PHASE_FILE_ENV = "SILLY_KICKS_PHASE_FILE"`,
  `ACTION_KIND_ENV = "SILLY_KICKS_ACTION_KIND"`;
  `job_rel(label: str, attempt: str) -> str` = `_ledger/_jobs/{label}.{attempt}.json`;
  `cost_rel(pipeline, stage, token, label, attempt) -> str` = `_ledger/{pipeline}/{stage}/{token}/_costs/{label}.{attempt}.json`;
  `reduce_rel(pipeline, stage, token, label, attempt) -> str` = `_ledger/{pipeline}/{stage}/{token}/_reduce/{label}.{attempt}.json`;
  `publish_first_unit(store) -> None` (once per process).
- Produces (in `launch.py`): `label_of(labels) -> str` (`job.variant.stage.shard[.action]`);
  `parse_stage_label(value: str) -> tuple[tuple[str, ...], str | None]`; `JobSpec.action_id: str | None = None`,
  `JobSpec.action_kind: str = "run"`.
- Produces (in `ledger.py`): `cost_aggregates(store, pipeline, stage, token) -> dict[str, dict]` (unit key ->
  record); `reduce_records(store, pipeline, stage, token) -> list[dict]`.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/hf_jobs/test_launch_typed.py — add
def test_label_of_appends_the_action_and_tolerates_an_old_record():
    from scripts.hf_jobs.launch import label_of, parse_stage_label

    old = {"job": "g", "variant": "default", "stage": "cv", "shard": "0of5"}
    assert label_of(old) == "g.default.cv.0of5"  # every Cycle 1 job record (spec r10, B R9-01)
    assert label_of({**old, "action": "20261021T143000Z_1a2b3c4d"}) == "g.default.cv.0of5.20261021T143000Z_1a2b3c4d"
    assert parse_stage_label("cache-cv") == (("cache", "cv"), None)
    assert parse_stage_label("cache-cv__20261021T143000Z_1a2b3c4d") == (("cache", "cv"), "20261021T143000Z_1a2b3c4d")


def test_the_job_command_exports_a_fresh_attempt_id_and_marks_phases():
    from scripts.hf_jobs import launch

    cmd = launch.job_command(_spec())[2]
    assert "SILLY_KICKS_ATTEMPT_ID" in cmd and "date -u +%Y%m%dT%H%M%SZ" in cmd
    for phase in ("script_start", "after_fetch"):  # `after_env_check` arrives with the image check (Task 18)
        assert f"phases mark {phase}" in cmd
```

```python
# tests/scripts/hf_jobs/test_ledger.py — add
def test_two_actions_with_the_same_n_and_a_retry_never_share_a_file(tmp_path):
    from scripts._stages import LocalStore
    from scripts.hf_jobs import ledger, phases

    st = LocalStore(tmp_path)
    labels = ["g.default.extract.0of3.20261021T143000Z_aaaaaaaa", "g.default.extract.0of3.20261021T150000Z_bbbbbbbb"]
    for label, attempt, key in ((labels[0], "20261021T143001Z_0001", "u1"), (labels[0], "20261021T143500Z_0002", "u2"),
                                (labels[1], "20261021T150001Z_0003", "u3")):
        rel = phases.cost_rel("g", "extract", "tok", label, attempt)
        st.write_bytes(rel, json.dumps({key: {"outcome": "done", "wall_s": 1.0, "started_ns": 1, "finished_ns": 2}}).encode())
    got = ledger.cost_aggregates(st, "g", "extract", "tok")
    assert sorted(got) == ["u1", "u2", "u3"]
    assert len(list((tmp_path / "_ledger/g/extract/tok/_costs").iterdir())) == 3


def test_job_startups_reads_an_old_job_record_without_an_action_label():
    import datetime

    from scripts.hf_jobs import ledger

    iso = lambda t: datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()  # noqa: E731
    job = {"labels": {"job": "g", "variant": "v", "stage": "cv", "shard": "0of1"}, "started_at": iso(100), "finished_at": iso(900)}
    rec = {"job_label": "g.v.cv.0of1", "started_ns": int(160e9)}
    assert ledger.job_startups([job], [rec]) == [60.0]
```

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/hf_jobs/test_launch_typed.py tests/scripts/hf_jobs/test_ledger.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Implement `phases.py`**

```python
"""Per-attempt job identity and the job phase record (ADR-128 Cycle 1b, S5.3, S7.2).

The job script makes an ATTEMPT ID at start (`SILLY_KICKS_ATTEMPT_ID`, `%Y%m%dT%H%M%SZ_<4 hex>`): an HF retry
starts a new script, so it gets a new id. Every per-job file name carries ``<in-job label>.<attempt id>``, so no
per-job file has two writers. The bash job script appends phase timestamps to a local file
(`SILLY_KICKS_PHASE_FILE`); the first unit of the process publishes them, with its own start, to the store."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

ATTEMPT_ENV = "SILLY_KICKS_ATTEMPT_ID"
PHASE_FILE_ENV = "SILLY_KICKS_PHASE_FILE"
ACTION_KIND_ENV = "SILLY_KICKS_ACTION_KIND"
LABEL_ENV = "SILLY_KICKS_JOB_LABEL"
_LOCK = threading.Lock()
_PUBLISHED = False


def job_rel(label: str, attempt: str) -> str:
    return f"_ledger/_jobs/{label}.{attempt}.json"


def cost_rel(pipeline: str, stage: str, token: str, label: str, attempt: str) -> str:
    return f"_ledger/{pipeline}/{stage}/{token}/_costs/{label}.{attempt}.json"


def reduce_rel(pipeline: str, stage: str, token: str, label: str, attempt: str) -> str:
    return f"_ledger/{pipeline}/{stage}/{token}/_reduce/{label}.{attempt}.json"


def identity() -> tuple[str, str] | None:
    """(in-job label, attempt id), or None outside a job (a laptop, CI)."""
    label, attempt = os.environ.get(LABEL_ENV), os.environ.get(ATTEMPT_ENV)
    return (label, attempt) if label and attempt else None


def mark(name: str) -> None:
    """Append ``{"phase": name, "t": epoch_s}`` to the local phase file (the bash script calls this)."""
    path = os.environ.get(PHASE_FILE_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"phase": name, "t": time.time()}) + "\n")


def publish_first_unit(store) -> None:
    """At the first unit start of the ATTEMPT: write the job phase record (the local phases plus ``first_unit``)
    to ``job_rel``. ONE writer per attempt: the first process to create the local marker
    ``<phase file>.first`` (``O_CREAT | O_EXCL`` on the job's local disk, which a retry does not share) writes;
    every other worker of the attempt skips (r1 B CCTP-PLAN-12)."""
    global _PUBLISHED
    ident = identity()
    path = os.environ.get(PHASE_FILE_ENV)
    with _LOCK:
        if _PUBLISHED or ident is None or not path:
            return
        _PUBLISHED = True
    try:
        os.close(os.open(path + ".first", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return  # another worker of this attempt is the writer
    rel = job_rel(*ident)
    phases = {}
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                phases.setdefault(row["phase"], row["t"])
    phases["first_unit"] = time.time()
    store.write_bytes(rel, (json.dumps({"label": ident[0], "attempt": ident[1], "phases": phases}, sort_keys=True) + "\n").encode())


if __name__ == "__main__":  # `python -m scripts.hf_jobs.phases mark <name>`
    if len(sys.argv) == 3 and sys.argv[1] == "mark":
        mark(sys.argv[2])
        raise SystemExit(0)
    raise SystemExit("usage: python -m scripts.hf_jobs.phases mark <phase>")
```

- [ ] **Step 4: Implement the launcher changes**

```python
# launch.py
def parse_stage_label(value: str) -> tuple[tuple[str, ...], str | None]:
    """THE parser of the `stage` label (every reader goes through it): ``a-b`` or, under the R8 fallback,
    ``a-b__<action id>``. Stage names never contain ``-`` or ``__`` (`Pipeline.validate`), and an action id
    never contains ``-``."""
    stages, sep, action = value.partition("__")
    return tuple(s for s in stages.split("-") if s), (action if sep else None)


def label_of(labels: Mapping[str, str]) -> str:
    stages, folded = parse_stage_label(labels["stage"])
    base = f"{labels['job']}.{labels['variant']}.{'-'.join(stages)}.{labels['shard']}"
    action = labels.get("action") or folded
    return f"{base}.{action}" if action else base  # an old record (no action) keeps today's value
```

`JobSpec` gains `action_id: str | None = None` and `action_kind: str = "run"`. `labels_for` adds
`"action": spec.action_id` when it is set (or, under the R8 fallback from Task 0.2, folds it into `stage`).
`job_env` adds `ACTION_KIND_ENV: spec.action_kind` and `PHASE_FILE_ENV: "/tmp/sk-phases.jsonl"`.
`job_command` starts with:

```python
    lines = [
        "set -euo pipefail",
        'export SILLY_KICKS_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(od -An -N2 -tx1 /dev/urandom | tr -d " \\n")"',
        f"{PYTHON_BIN} -c 'import json,time;open(\"/tmp/sk-phases.jsonl\",\"a\").write(json.dumps({{\"phase\":\"script_start\",\"t\":time.time()}})+\"\\n\")'",
        # ... the existing git lines ...
    ]
```

and, after the git checkout line, `".venv/bin/python -m scripts.hf_jobs.phases mark after_fetch"` once the venv
exists (with the prebuilt image of Task 18 the venv exists before the fetch; until Task 18, place the
`after_fetch` mark after `uv sync`). After the environment check (Task 18) add `phases mark after_env_check`; after
each dataset fetch line add `phases mark after_dataset_fetch`. Update the test helper strings to match.

`ledger.job_startups` keeps its contract; it calls `label_of`, which now tolerates a missing `action`.

- [ ] **Step 5: Write cost aggregates in the executor and reduce records in the reduce**

In `executor.worker_main`, extend each outcome row with the unit's `wall_s`, `peak_rss_bytes`, `started_ns`,
`finished_ns`, `worker` and `inner_workers` (`run_unit` returns them: change its return to
`tuple[str, dict]` = (outcome, ledger record) and update its callers). In `LocalExecutor._run`, after `_dispatch`,
write ONE aggregate for this job's units of this stage:

```python
        ident = phases.identity()
        if ident is not None and order:
            rows = {r["key"]: {k: r.get(k) for k in ("outcome", "wall_s", "peak_rss_bytes", "started_ns",
                                                     "finished_ns", "worker", "inner_workers")}
                    for r in _read_outcomes(outdir)}  # read before the temp dir is removed
            ctx.store.write_bytes(
                phases.cost_rel(ctx.pipeline.name, st.name, ctx.tokens[st.name], *ident),
                _canon_json(rows),
            )
```

(`_dispatch` owns `outdir`; return the rows from `_dispatch` with `relaunched` and `order` instead of reading
after cleanup.) In `run_unit`, call `phases.publish_first_unit(ctx.store)` before `started = time.time_ns()`.

In `_stages.py` add:

```python
def write_reduce_record(ctx: RunContext, name: str, *, started_ns: int, n_results: int) -> None:
    """The S5.2 reduce ledger record. ``in_job`` is False only in a separate reduce job (E-4)."""
    from scripts._dataset_reads import store_read_cap
    from scripts.hf_jobs import phases

    ident = phases.identity()
    if ident is None:
        return
    rec = {
        "stage": name, "token": ctx.tokens[name], "started_ns": started_ns, "finished_ns": time.time_ns(),
        "wall_s": (time.time_ns() - started_ns) / 1e9, "n_results": n_results,
        "read_cap": store_read_cap() or 4,
        "in_job": os.environ.get(phases.ACTION_KIND_ENV, "run") != "reduce",
    }
    ctx.store.write_bytes(phases.reduce_rel(ctx.pipeline.name, name, ctx.tokens[name], *ident), _canon_json(rec))
```

Add `ledger.cost_aggregates` (one listing of the `_costs/` directory, one read per file, merged by unit key; for a
unit in several files keep the row with the latest `finished_ns`) and `ledger.reduce_records` (the same for
`_reduce/`).

- [ ] **Step 6:** Run: `.venv/Scripts/python -m pytest tests/scripts/hf_jobs/test_launch_typed.py tests/scripts/hf_jobs/test_ledger.py tests/scripts/hf_jobs/test_executor.py tests/scripts/test_stage_contract.py -q -p no:cacheprovider` (background). Expected: PASS.

### Task 5: Linear `--plan` and live history from aggregates (S5.11)

**Files:** Modify `scripts/_stages.py` (`plan`, `output_present`, `_ready`), `scripts/hf_jobs/campaign.py`
(`live_history`). Test: `tests/scripts/test_reduce_store_access.py` (extend).

- [ ] **Step 1: Failing test**

```python
def test_plan_and_live_history_are_linear_with_one_crashed_job(tmp_path):
    """S5.11: --plan lists each generation once and reads no result bytes; live history reads one file per job
    aggregate plus unit records only for the units of the job that wrote no aggregate."""
    import json as _json

    from scripts.hf_jobs import campaign as cp, phases

    ctx, store = _ctx(tmp_path, 200, "parquet")
    gen = ctx.generation("a")
    keys = sorted(store.names(gen))
    units = [k[:-8] for k in keys if k.endswith(".parquet")]
    for j in range(4):  # 5 jobs x 40 units; job 4 crashed before writing its aggregate
        rows = {u: {"outcome": "done", "wall_s": 1.0, "finished_ns": 1} for u in units[j * 40:(j + 1) * 40]}
        store.write_bytes(phases.cost_rel("t", "a", ctx.tokens["a"], f"t.d.a.{j}of5.X", "A"), _json.dumps(rows).encode())
    store.calls.clear()
    p = s.plan(ctx)
    assert store.calls[("names", gen)] == 1
    assert not any(op == "read" and rel.endswith(".parquet") for (op, rel) in store.calls)
    store.calls.clear()
    cp.live_history(p, store, {})
    unit_reads = [rel for (op, rel) in store.calls if op == "read" and rel.endswith(".ledger.json")]
    assert len(unit_reads) == 40  # only job 4's units
```

- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3: Implement.** In `plan(ctx)`, build `listing: dict[str, frozenset[str]]` lazily (one `names` call per
  generation per `plan` call) and pass it to a new `output_present(ctx, name, *, verify, names=None)` and to
  `unit_states` (add `names: Collection[str] | None = None` to `unit_states`; when given, it skips its own listing).
  In `campaign.live_history`, replace `unit_records(store, gen)` with: `aggs = ledger.cost_aggregates(store, pipeline,
  stage, token)`; `missing = done_units - set(aggs)` where `done_units = set(st["units"]) - set(st["pending_units"]) -
  set(st["failed_units"])` (excluded units have no record either; ignore a missing record for them); read
  `<gen>/<k>.ledger.json` only for `missing`. Build `UnitCost` from the merged rows (`ledger.unit_cost` accepts dicts
  with `outcome`, `wall_s`, `peak_rss_bytes`).
- [ ] **Step 4:** Run it plus `tests/scripts/hf_jobs/test_campaign.py`. Expected: PASS.

---

## Phase 3 — controller, planner, scheduler, errors

### Task 6: Actions, ownership, DAG-parallel actions and partial grants (S5.3)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`Action`, `next_actions`, `_job_specs`, `_running`, new
`make_action_id`, `Ownership`, `read_ownership`, `write_action_record`), `scripts/hf_jobs/store.py` (prefix
allowlist, Task 15 finishes it). Test: `tests/scripts/hf_jobs/test_campaign.py`.

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Ownership: all_units: bool; keys: frozenset[str]`;
  `make_action_id(instance_id: str, stages: Sequence[str], tokens: Sequence[str], partition, now: datetime) -> str`;
  `Action.action_id: str | None`; `Action.owns_all: bool`;
  `next_actions(plan, owned: Mapping[str, Ownership], history, c, *, lpt=True, grant: Callable[[Action], int] | None = None) -> list[Action]`;
  `action_rel(pipeline, stage, token, action_id) -> str` = `_actions/{pipeline}/{stage}/{token}/{action_id}.json`.

- [ ] **Step 1: Failing tests** (add to `test_campaign.py`; reuse `_st` and `_plan`)

```python
_NOONE: dict = {}


def test_independent_ready_stages_get_actions_in_the_same_step():
    """S5.3: a -> {b, c}; {a, c} -> d. With a reduced, b and c both act; d waits for both."""
    p = _plan([_st("a", reduced=True), _st("b", needs=("a",)), _st("c", needs=("a",)), _st("d", needs=("a", "c"), ready=False)])
    hist = {n: {"cost": UnitCost(10, 12, 15, 2**30, 8), "efficiency": 1.0, "per_unit": None} for n in "bcd"}
    acts = C.next_actions(p, _NOONE, hist, Constraints())
    assert sorted(a.stages for a in acts) == [("b",), ("c",)]


def test_the_real_ghost_gk_graph_launches_cv_and_fit_together():
    p = _plan([_st("cache", n=1, reduced=True), _st("cv", n=5, needs=("cache",)), _st("fit", n=1, needs=("cache",)),
               _st("importance", n=1, needs=("cache", "fit"), ready=False, inner=True)])
    hist = {"cv": {"cost": UnitCost(400, 500, 600, 2**30, 5), "efficiency": 1.0, "per_unit": None}}
    acts = C.next_actions(p, _NOONE, hist, Constraints())
    assert {a.stages[0] for a in acts} == {"cv", "fit"}


def test_no_reduce_and_no_duplicate_while_running_actions_own_every_pending_unit():
    """r1 B CCTP-PLAN-01 (BLOCKING): when running LPT jobs own every pending unit, the stage WAITS. A reduce comes
    only when no unit is pending (and so none is owned)."""
    st = _st("x", n=40)
    hist = {"x": {"cost": UnitCost(60, 70, 80, 2**30, 40), "efficiency": 1.0, "per_unit": None}}
    owned = {("x", "t_x"): C.Ownership(False, frozenset(st["pending_units"]))}
    assert C.next_actions(_plan([st]), owned, hist, Constraints()) == []
    done = _st("x", n=40, done=40)
    acts = C.next_actions(_plan([done]), {}, hist, Constraints())
    assert [a.kind for a in acts] == ["reduce"]


def test_a_second_grant_with_the_same_n_writes_under_a_new_action_id():
    a = C.partition_rel("g", "x", "t", "20261021T143000Z_aaaaaaaa", 0, 2)
    b = C.partition_rel("g", "x", "t", "20261021T143100Z_bbbbbbbb", 0, 2)
    assert a != b and a.startswith("_partitions/g/x/t/20261021T143000Z_aaaaaaaa/")


def test_a_round_robin_action_never_runs_partially():
    """A `download` instance (no partition files) launches all of a stage's jobs or waits (S5.3)."""
    a = C.Action("run", ("x",), 4, 8, 1, 1800, owns_all=True)
    assert C.allocate([("i1", a)], budget=3) == []
    assert [k for _, _, k in C.allocate([("i1", a)], budget=4)] == [4]


def test_a_running_job_maps_to_its_action_through_its_label_even_from_another_campaign():
    """Ownership reads every job on the store's run key (r4 B R4-04), mapped by the `action` label."""
    jobs = [types.SimpleNamespace(id="j1", labels={"run_key": "g-o", "action": "A", "campaign": "other"},
                                  status=types.SimpleNamespace(stage="RUNNING")),
            types.SimpleNamespace(id="j2", labels={"run_key": "g-o", "action": "B", "campaign": "cycle1b"},
                                  status=types.SimpleNamespace(stage="COMPLETED"))]
    records = [{"pipeline": "g", "stage": "x", "token": "t", "keys": ["u1"], "owns_all": False, "action_id": "A"},
               {"pipeline": "g", "stage": "x", "token": "t", "keys": ["u2"], "owns_all": False, "action_id": "B"}]
    owned = C.read_ownership_from_records(C.mark_running(records, jobs))
    assert owned[("x", "t")].keys == frozenset({"u1"})


def test_a_running_profile_owns_its_stage():
    owned = {("x", "t_x"): C.Ownership(True, frozenset())}
    assert C.next_actions(_plan([_st("x", n=40)]), owned, {}, Constraints()) == []


def test_action_ids_match_the_label_pattern_and_differ_for_same_second_actions():
    import datetime

    from scripts.hf_jobs.launch import LABEL_MAX, LABEL_PATTERN

    now = datetime.datetime(2026, 10, 21, 14, 30, tzinfo=datetime.timezone.utc)
    a = C.make_action_id("rk/default", ("cv",), ("t1",), None, now)
    b = C.make_action_id("rk/default", ("fit",), ("t2",), None, now)
    assert a != b and a.startswith("20261021T143000Z_") and "-" not in a
    assert LABEL_PATTERN.fullmatch(a) and len(a) <= LABEL_MAX


def test_two_variants_in_one_store_do_not_block_each_other():
    """Records are keyed by pipeline, stage AND token (S5.3, r4 B R3-03)."""
    a = _st("cv", n=5)
    b = {**_st("cv", n=5), "token": "t_other"}
    owned = C.read_ownership_from_records(
        [{"pipeline": "ghost_gk", "stage": "cv", "token": "t_cv", "keys": a["units"], "owns_all": False,
          "action_id": "X", "running": True}]
    )
    assert owned.get(("cv", "t_cv")) and not owned.get(("cv", b["token"]))
```

(`owned` is keyed by `(stage, token)`; `next_actions` looks up `owned.get((st["name"], st["token"]))`.)

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/hf_jobs/test_campaign.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Implement**

```python
@dataclasses.dataclass(frozen=True)
class Ownership:
    """What running actions own in one stage generation: every unit (profile, grouped, round-robin, reduce)
    or the keys of their partitions / escalations."""

    all_units: bool
    keys: frozenset[str]


def make_action_id(instance_id, stages, tokens, partition, now) -> str:
    """``%Y%m%dT%H%M%SZ_<8 hex>``; the hash names the action itself (instance, stages, tokens, partition), so
    two actions of one instance in one second differ (r8 B R7-02). No ``-``: readers split `stage` on it."""
    digest = hashlib.sha256(
        json.dumps([instance_id, list(stages), list(tokens), partition], sort_keys=True, default=list).encode()
    ).hexdigest()[:8]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{digest}"


def action_rel(pipeline: str, stage: str, token: str, action_id: str) -> str:
    return f"_actions/{pipeline}/{stage}/{token}/{action_id}.json"


def read_ownership_from_records(records: Iterable[Mapping]) -> dict[tuple[str, str], Ownership]:
    """Merge the action records whose jobs still run into ``{(stage, token): Ownership}``."""
    out: dict[tuple[str, str], Ownership] = {}
    for r in records:
        if not r.get("running"):
            continue
        k = (r["stage"], r["token"])
        prev = out.get(k, Ownership(False, frozenset()))
        out[k] = Ownership(prev.all_units or bool(r["owns_all"]), prev.keys | frozenset(r.get("keys") or ()))
    return out
```

`next_actions` rewrite (keep every existing rule; change the walk):

```python
def next_actions(plan, owned, history, c, *, lpt=True, grant=None) -> list[Action]:
    """EVERY action to take for ONE instance now (S5.3): one per stage whose needs are all reduced and whose
    pending units are not all owned by running actions. Grouping, retries, memory escalation, tolerant
    reduce, halt and profile keep their 74d57ba rules per stage. ``grant(action) -> int`` caps an action's
    jobs (the scheduler's budget, S5.5); the action is re-planned at that size."""
    stages = list(plan["stages"])
    reduced = {s["name"] for s in stages if s.get("reduced")}
    grouped: set[str] = set()
    out: list[Action] = []
    for i, st in enumerate(stages):
        if st.get("reduced") or st["name"] in grouped:
            continue
        if not st.get("ready") or st.get("n_units") is None:
            continue  # NOT `return`: a later independent stage may act
        own = owned.get((st["name"], st["token"]))
        if own is not None and own.all_units:
            continue
        a = _stage_action(stages, i, st, reduced, own, history, c, lpt=lpt, grant=grant)  # the 74d57ba body
        if a is None:
            continue
        grouped |= set(a.stages[1:])
        out.append(a)
    return out
```

Move the 74d57ba per-stage body (lines 242-317) into `_stage_action(...) -> Action | None`, with these changes
(r1 B CCTP-PLAN-01, BLOCKING):

```python
    owned_keys = own.keys if own is not None else frozenset()
    pending = set(st.get("pending_units") or ()) | retry          # as in 74d57ba (retry = failed below attempts)
    if pending & owned_keys and not (pending - owned_keys - memory):
        return None   # every pending unit is owned by a running action: WAIT. Never a reduce, never a duplicate.
    todo = sorted(pending - owned_keys - memory)
```

The 74d57ba "no `todo` -> reduce / tolerant reduce / halt" branch is reached only when `pending` is EMPTY (so no unit
is pending or owned). The memory-escalation branch takes `memory - owned_keys`. `plan_stage` gets
`grant=grant(prelim) if grant else None` (Task 7 adds `grant` and the remainder). Set `owns_all=True` for profile,
grouped (`n_units <= 2`), reduce and round-robin (`partition is None`) actions.

`partition_rel(pipeline, stage, token, action_id, i, n) -> str` =
`_partitions/{pipeline}/{stage}/{token}/{action_id}/{i}of{n}.json` (`_job_specs` uses it). `mark_running(records,
jobs) -> list[dict]` sets each record's `running` from the jobs whose `action` label (through
`launch.parse_stage_label` for the R8 fallback) equals its `action_id` and whose status is not TERMINAL.

In `step`, before launching an action: compute `action_id = make_action_id(inst.id, a.stages, [tokens], a.partition,
utcnow)`, write `action_rel(...)` through `store.upload_bytes` with `{"pipeline", "stage", "token", "keys",
"owns_all", "action_id", "kind", "created_at"}`, pass `action_id` into every `JobSpec` of the action, and move the
partition path under the action id: `_partitions/{pipeline}/{stage}/{token}/{action_id}/{i}of{n}.json`.

`read_ownership(store, plan, jobs_by_action) -> dict[(stage, token), Ownership]`: list each stage's
`_actions/{pipeline}/{stage}/{token}/` once, read each record, mark `running = any job with label action ==
action_id is not TERMINAL`, and merge with `read_ownership_from_records`. The job list comes from Task 8's
run-key listing.

Every reader of the `stage` label goes through `launch.parse_stage_label` (`campaign.py:489`, `:716`, `:760`): replace
the `.split("-")` calls.

- [ ] **Step 4:** Run the Step 2 command. Expected: PASS.

### Task 7: The makespan planner and `plan_reduce` (S5.2, S5.4)

**Files:** Modify `scripts/hf_jobs/planner.py`, `scripts/hf_jobs/campaign.py` (`_stage_action` passes `grant`). Test:
`tests/scripts/hf_jobs/test_planner.py`.

**Interfaces:**
- Produces: `Constraints.unit_overhead_s: float = 0.0`; `Constraints.store_read_s: float` (from
  `datasets.store_settings().per_result_read_s`, wired in `campaign.planning_constraints`);
  `plan_stage(..., grant: int | None = None) -> StagePlan`;
  `@dataclass ReducePlan: n_results: int; wall_s: float; timeout_s: int`;
  `plan_reduce(n_results: int, c: Constraints, *, per_result_s: float | None = None) -> ReducePlan`;
  `ideal_stage_wall(per_unit: Mapping[str, float], cost: UnitCost, c: Constraints, *, inner_parallel: bool,
  n_inner: int, efficiency: float | None) -> float` (bar 1).

- [ ] **Step 1: Failing tests**

```python
def test_the_planner_picks_the_shortest_finish_within_the_grant_and_fewest_jobs_within_5_percent():
    c = P.Constraints(startup_s=120, concurrency_cap=20, target_s=3600)
    cost = P.UnitCost(60, 70, 80, 2**30, 100)
    one = P.plan_stage("x", [f"k{i}" for i in range(5)], cost, inner_parallel=False, c=c)
    assert one.n_jobs == 1  # 5 x 60 s fits one job; a second job saves < 5 % after its startup
    many = P.plan_stage("x", [f"k{i}" for i in range(3961)], cost, inner_parallel=False, c=c)
    assert many.n_jobs == 20 and many.predicted_job_wall_s < c.target_s
    capped = P.plan_stage("x", [f"k{i}" for i in range(3961)], cost, inner_parallel=False, c=c, grant=3)
    assert capped.n_jobs == 3 and capped.predicted_job_wall_s <= c.target_s
    taken = {k for part in capped.partition for k in part}
    assert capped.remainder and not taken & set(capped.remainder) and len(taken) + len(capped.remainder) == 3961


def test_cv_folds_fan_out_when_a_shorter_makespan_wins():
    c = P.Constraints(startup_s=120, concurrency_cap=20)
    # 20 GiB peak -> W = 1 per job (RAM-bound, like a heavy cv fold): one job runs the 5 folds in 5 waves (2200 s),
    # 5 jobs run them in one (440 s). With a small peak (W = 8) one job already runs all 5 at once, and the
    # fewest-jobs rule keeps it at 1 job (owner ruling 12: the wave model).
    cost = P.UnitCost(440, 450, 460, 20 * 2**30, 5)
    p = P.plan_stage("cv", [f"{i:04d}" for i in range(5)], cost, inner_parallel=False, c=c)
    assert p.n_jobs >= 2


def test_a_partial_grant_leaves_an_unowned_remainder_within_the_job_length_limit():
    """S5.3 / S5.4 through the controller: k granted jobs take only what fits k jobs under `target_s`; the rest stays
    for the next step (r1 B CCTP-PLAN-02). Numbers (owner ruling 12, the wave model): 40 units x 2000 s, W = 8 -> a job
    of 8 units runs one wave (120 + 2000 = 2120 s); a 9th unit makes 2 waves (4120 s > 3600), so 2 jobs take 16 and
    leave 24. Lives here, not in Task 6, because it needs this task's `grant` and remainder."""
    from scripts.hf_jobs import campaign as C
    from tests.scripts.hf_jobs.test_campaign import _plan, _st

    st = _st("x", n=40)
    hist = {"x": {"cost": P.UnitCost(2000, 2100, 2200, 2**30, 40), "efficiency": 1.0, "per_unit": None}}
    c = P.Constraints(target_s=3600, startup_s=120)
    first = C.next_actions(_plan([st]), {}, hist, c, grant=lambda a: 2)[0]
    assert first.n_jobs == 2 and len(first.partition) == 2 and first.workers == 8
    assert first.predicted_job_wall_s <= c.target_s
    taken = frozenset(k for part in first.partition for k in part)
    assert len(taken) == 16 and all(len(part) == 8 for part in first.partition)
    owned = {("x", "t_x"): C.Ownership(False, taken)}
    second = C.next_actions(_plan([st]), owned, hist, c)
    rest = {k for part in second[0].partition for k in part}
    assert rest == set(st["pending_units"]) - taken


def test_plan_reduce_is_linear_with_and_without_history_and_respects_the_floor():
    c = P.Constraints(store_read_s=0.05, startup_s=60, timeout_floor_s=1800, timeout_safety=1.5)
    assert P.plan_reduce(0, c).timeout_s == 1800 and P.plan_reduce(1, c).timeout_s == 1800
    big = P.plan_reduce(3961, c)
    assert abs(big.wall_s - (60 + 3961 * 0.05)) < 1e-6
    assert P.plan_reduce(3961, c, per_result_s=2.0).timeout_s == int(1.5 * (60 + 3961 * 2.0))
```

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/hf_jobs/test_planner.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Implement.** In `plan_stage`, for the non-inner-parallel branch replace the `n_jobs` formula with a
  search:

```python
        ceiling = max(1, min(n, grant if grant is not None else c.concurrency_cap))
        costs = {k: (per_unit[k] if per_unit else cost.p50_s) + c.unit_overhead_s for k in keys}

        def job_wall(part) -> float:  # owner ruling 12: units run whole, in waves (LPT on the job's W workers)
            return c.startup_s + max(job_makespan([costs[k] for k in part], workers) / eff, cost.p95_s)

        def finish(n_jobs: int) -> tuple[float, tuple[tuple[str, ...], ...]]:
            parts = lpt_partition(costs, n_jobs)
            ordered = tuple(tuple(sorted(p, key=lambda k: (-costs[k], k))) for p in parts)  # LPT order per job
            return max(job_wall(p) for p in ordered), ordered

        trials = {j: finish(j) for j in sorted({1, 2, 3, 4, 5, 8, 10, 12, 16, 20, ceiling} & set(range(1, ceiling + 1)))}
        best = min(w for w, _ in trials.values())
        n_jobs = min(j for j, (w, _) in trials.items() if w <= 1.05 * best)
        wall, parts = trials[n_jobs]
        partition = parts if per_unit or lpt_wanted else None
```

Keep `lpt_wanted` = the caller's `lpt` (pass it into `plan_stage` as a new keyword, default True), so a `download`
instance still gets `partition=None` (round-robin). Apply the same search to the inner-parallel branch over
`n_jobs` in `1..min(n, ceiling)` with `wall = startup + ceil(n/n_jobs) * p50`. Keep `irreducible_floor_s`,
`timeout`, `usd`.

`target_s` is the job length LIMIT (S5.4). When the best plan within the ceiling still has a job over the limit:

```python
        if wall > c.target_s and lpt_wanted:
            # Incremental (r4 B R4-01): units arrive in LPT order (longest first), so a bin's makespan is kept by its
            # W worker loads -- the next unit goes to the least-loaded worker. This IS `job_makespan` of the bin (LPT
            # on W workers), at O(log W) per check instead of re-scheduling the bin: 3961 units plan in milliseconds.
            bins: list[list[str]] = [[] for _ in range(ceiling)]
            loads: list[list[float]] = [[0.0] * max(1, workers) for _ in range(ceiling)]  # heaps of worker loads
            spans = [0.0] * ceiling
            remainder: list[str] = []

            def wall_of(span: float) -> float:
                return c.startup_s + max(span / eff, cost.p95_s)

            for k in sorted(costs, key=lambda k: (-costs[k], k)):   # LPT order, deterministic
                j = min(range(ceiling), key=lambda i: (spans[i], i))
                new_span = max(spans[j], loads[j][0] + costs[k])
                if not bins[j] or wall_of(new_span) <= c.target_s:  # a unit over the limit still runs alone
                    heapq.heapreplace(loads[j], loads[j][0] + costs[k])
                    spans[j] = new_span
                    bins[j].append(k)
                else:
                    remainder.append(k)
            parts = tuple(tuple(b) for b in bins if b)  # each already in LPT order
            n_jobs, partition = len(parts), parts
            wall = max(wall_of(spans[i]) for i, b in enumerate(bins) if b)
            remainder = tuple(sorted(remainder))
        else:
            remainder = ()
```

Add to `planner.py`:

```python
def job_makespan(unit_costs: Sequence[float], workers: int) -> float:
    """The wall of ONE job: LPT list scheduling of its units on its W workers (owner ruling 12). Units run whole, so
    8 units of 2000 s on 8 workers take 2000 s and a 9th makes it 4000 s; a fluid `sum / W` would say 2250 s."""
    loads = [0.0] * max(1, int(workers))
    for cost in sorted(unit_costs, reverse=True):
        i = min(range(len(loads)), key=lambda j: (loads[j], j))
        loads[i] += cost
    return max(loads) if unit_costs else 0.0
```

`workers` and `eff` are the existing `workers` (from `_workers`) and `efficiency or 1.0`; `par` goes. A partition
keeps each job's keys in LPT order (longest first), and **the executor runs a `--shard-keys` list in its given
order**: `LocalExecutor.run_stage` replaces `mine = sorted(set(keys))` (`executor.py:299`) with an order-keeping
de-duplication (`list(dict.fromkeys(keys))`). Workers claim from that list in order, so the job runs the LPT schedule
the model predicts. Outputs never depend on the order (INV-1): the reduce sorts by `reduce_order`. Tests
(`test_planner.py`): `job_makespan([2000.0] * 8, 8) == 2000` and `job_makespan([2000.0] * 9, 8) == 4000`; the incremental
fill's per-bin span equals `job_makespan` of the bin's units (a property check over 50 random cost lists, seeded); a
3961-unit `plan_stage` with `grant=20` runs in under 1 s; a plan of
13 units of 2000 s at W = 8 predicts 2 waves; (`test_executor.py`) a `--shard-keys` list runs in its given order (the
outcomes' completion order follows it at W = 1).

`StagePlan` gains `remainder: tuple[str, ...] = ()`: those units stay pending and unowned, so the next step plans
them (S5.3). A round-robin (`download`) action has no remainder: it launches all its jobs or waits (Task 8
`allocate`).

```python
@dataclasses.dataclass(frozen=True)
class ReducePlan:
    n_results: int
    wall_s: float
    timeout_s: int


def plan_reduce(n_results: int, c: Constraints, *, per_result_s: float | None = None) -> ReducePlan:
    """S5.2: a reduce reads each result once; its wall is linear in the results. Before history, the cost per
    result is the store probe's `store_read_s`; with history, the measured seconds per result."""
    per = c.store_read_s if per_result_s is None else per_result_s
    wall = c.startup_s + max(0, n_results) * per
    return ReducePlan(n_results, wall, int(max(c.timeout_floor_s, c.timeout_safety * wall)))


def ideal_stage_wall(per_unit, cost, c, *, inner_parallel, n_inner, efficiency) -> float:
    """S7.2 bar 1: the stage's wall with an UNLIMITED budget, from measured unit walls (plus the startup)."""
    unlimited = dataclasses.replace(c, concurrency_cap=max(1, len(per_unit)))
    p = plan_stage("ideal", sorted(per_unit), cost, inner_parallel=inner_parallel, n_inner=n_inner,
                   efficiency=efficiency, per_unit=per_unit, c=unlimited)
    return p.predicted_job_wall_s
```

In `campaign.next_actions` (`_stage_action`), the reduce action uses
`plan_reduce(st["done"], c, per_result_s=history.get(hk, {}).get("reduce_per_result_s"))` for its `timeout_s`; the
`2 * c.target_s` goes. `live_history` adds `reduce_per_result_s` from `ledger.reduce_records` (the median of
`wall_s / n_results` over the stage's successful records).

- [ ] **Step 4:** Run the Step 2 command plus `test_campaign.py`. Expected: PASS.

### Task 8: Critical-path scheduler, the run-key job listing, `--with-replica`, `check_concurrency` (S5.5)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`step`, new `remaining_critical_path`, `grant_order`),
`scripts/hf_jobs/launch.py` (new `jobs_for_run_keys`), `scripts/hf_jobs/preflight.py` (new `check_concurrency`).
Create `scripts/hf_jobs/campaigns/cycle1b.toml`, `tests/scripts/hf_jobs/test_campaign_cycle1b.py`.

**Interfaces:**
- Produces: `remaining_critical_path(plan, history, c) -> float`;
  `grant_order(candidates: Sequence[tuple[str, float, Action]]) -> list[tuple[str, Action]]` (instance id,
  critical path, action; sorted by path desc, then instance id);
  `allocate(ordered: Sequence[tuple[str, Action]], *, budget: int) -> list[tuple[str, Action, int]]`;
  `launch.jobs_for_run_keys(api, run_keys: Iterable[str]) -> list[JobInfo]`;
  `preflight.check_concurrency(cap: int, summaries: Iterable[Path], api) -> Check`.

- [ ] **Step 1: Failing tests**

```python
def test_the_grant_order_is_longest_remaining_chain_first_and_deterministic():
    a, b, c3 = (C.Action("run", ("x",), 4, 8, 1, 1800) for _ in range(3))
    order = C.grant_order([("i2", 100.0, a), ("i1", 300.0, b), ("i3", 300.0, c3)])
    assert [i for i, _ in order] == ["i1", "i3", "i2"]
    assert order == C.grant_order(list(reversed([("i2", 100.0, a), ("i1", 300.0, b), ("i3", 300.0, c3)])))


def test_remaining_critical_path_follows_the_dag():
    p = _plan([_st("a", reduced=True), _st("b", needs=("a",)), _st("c", needs=("b",)), _st("d", needs=("a",))])
    hist = {n: {"cost": UnitCost(600, 600, 600, 2**30, 4), "efficiency": 1.0, "per_unit": None} for n in "bcd"}
    cp_bc = C.remaining_critical_path(p, hist, Constraints(startup_s=0, concurrency_cap=1))
    assert cp_bc == pytest.approx(2 * 4 * 600 / 8 + 0, rel=0.5)  # chain b -> c dominates d


def test_allocate_grants_in_order_and_a_zero_budget_launches_nothing():
    a = C.Action("run", ("x",), 4, 8, 1, 1800)
    b = C.Action("run", ("y",), 4, 8, 1, 1800)
    ordered = [("i1", a), ("i2", b)]
    assert C.allocate(ordered, budget=0) == []
    assert [(i, k) for i, _, k in C.allocate(ordered, budget=5)] == [("i1", 4), ("i2", 1)]
    assert [(i, k) for i, _, k in C.allocate(ordered, budget=20)] == [("i1", 4), ("i2", 4)]
```

(`allocate(ordered, *, budget) -> list[tuple[instance_id, Action, granted_jobs]]` is the pure budget walk that
`step` applies; an action with `owns_all` and `n_jobs > granted` is skipped, not split, because a profile, a
grouped job or a round-robin action cannot run partially.)

```python
# tests/scripts/hf_jobs/test_campaign_cycle1b.py
def test_cycle1b_is_the_cycle1_population_at_cap_20():
    from scripts.hf_jobs import campaign as C

    one = C.load_campaign("scripts/hf_jobs/campaigns/cycle1.toml")
    two = C.load_campaign("scripts/hf_jobs/campaigns/cycle1b.toml")
    assert two.name == "cycle1b" and two.concurrency_cap == 20
    assert [(j.job, j.tier, j.dataset, j.variants) for j in two.jobs] == [(j.job, j.tier, j.dataset, j.variants) for j in one.jobs]
```

- [ ] **Step 2:** Run both files. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `remaining_critical_path`: for each not-reduced stage, cost = its planned wall (`plan_stage(...).predicted_job_wall_s`
    with the history cost, or `c.startup_s + c.target_s` when no history); longest path over `needs` by DP in DAG
    order.
  - `step`: (1) list jobs once with `launch.jobs_for_run_keys(api, {every instance's run key})` (labels
    `{"run_key": rk}`, one `list_jobs` call per run key) — ownership and "running" use THIS list, so a concurrent
    campaign on the same store is seen (r4 B R4-04); error history keeps `jobs_for(campaign, commit)`;
    (2) per instance read plan, ownership and history, call `next_actions` with no grant; (3) collect
    `(instance id, remaining_critical_path, action)` for every action; (4) walk `grant_order`, grant
    `min(budget, a.n_jobs)`, re-plan the action at that grant (`next_actions(..., grant=lambda _: k)` for that
    stage), launch, subtract; (5) append each launched action to `_campaign/{campaign}/{commit7}/predictions.jsonl`
    (Task 20 defines the record).
  - `--with-replica N` on `launch`/`resume`: `step` iterates `instances(c, commit)` plus
    `instances(c, commit, replica=N)` plus the canary instances (Task 17) under the one budget.
  - `cycle1b.toml`: a copy of `cycle1.toml` with `name = "cycle1b"` and `concurrency_cap = 20`, and a header comment
    citing S5.5 and the Cycle 1 measurement (18 concurrent accepted).
  - `check_concurrency`: read `max_concurrent` from the committed summaries (Task 20 writes it; for Cycle 1 compute
    it from the summary's `jobs` list: the peak of overlapping `started_at`..`finished_at` windows); warn (ok=True, a
    flag) when `cap` exceeds it. `huggingface_hub` 2.2 exposes no per-account concurrent-job limit in the typed
    surface the preflight already checks (`run_job`, `list_jobs`, `list_jobs_hardware`), so the check records
    "account limit: not exposed by the API" and relies on the measured peak; R3 covers a refusal.
- [ ] **Step 4:** Run the Step 2 tests plus `test_campaign.py`, `test_preflight.py`. Expected: PASS.

### Task 9: Plan from committed history, no profile when history exists (S5.7, E-5)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`committed_history`, `cmd_summary`, new `cmd_history_export`),
`scripts/hf_jobs/ledger.py` (`load_history`). Test: `test_campaign.py`.

- [ ] **Step 1: Failing tests**

```python
def test_committed_history_carries_per_unit_costs_and_suppresses_the_profile(tmp_path, monkeypatch):
    import json

    (tmp_path / "cycle1.json").write_text(json.dumps({"stages": [{"history_key": "hk", "unit_cost":
        {"p50_s": 10, "p95_s": 12, "p99_s": 15, "peak_rss_bytes": 2**30, "n_samples": 40}}]}))
    (tmp_path / "cycle1.per_unit.json").write_text(json.dumps({"run_commit": "x", "run_tree_dirty": False,
        "per_unit": {"hk": {"u0": 99.0, "u1": 1.0}}}))
    monkeypatch.setattr(C, "SUMMARY_DIR", tmp_path)
    hist = C.committed_history()
    assert hist["hk"]["per_unit"] == {"u0": 99.0, "u1": 1.0}
    acts = C.next_actions(_plan([_st("x", n=40, hk="hk")]), {}, hist, Constraints())
    assert acts and acts[0].kind == "run"  # never "profile" with committed history
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.** `committed_history` merges `*.per_unit.json` (`{"per_unit": {history_key: {key: wall}}}`)
  into each history entry; a stage row's own `per_unit` (written by the Cycle 1b `cmd_summary`) wins over the export.
  `_stage_action` passes `per_unit` from history to LPT whenever `lpt`. `cmd_summary` writes `per_unit` per stage row
  (from the merged cost aggregates) and the campaign-level `max_concurrent`. `cmd_history_export(c, commit, *, api)`
  reads each instance's unit ledger records ONCE (this is the one slow, read-only pass over the Cycle 1 stores) and
  writes `docs/research/hf_campaigns/<name>.per_unit.json` with top-level `run_commit` = the Cycle 1 commit and
  `run_tree_dirty: false`; it refuses a dirty tree (`require_clean_tree`). Register the file pattern where
  `test_artifact_provenance_output` expects research JSON (read that test first).
- [ ] **Step 4:** Run. Expected: PASS.

### Task 10: Infra-failure resume and the operator `--clear-halt` (S5.8)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`resume_after_error`, `_trailing_errors`, `main`), `scripts/hf_jobs/store.py`
(`_operator/` prefix). Test: `test_campaign.py`.

- [ ] **Step 1: Failing tests**

```python
def _job(stage="ERROR", msg="", created=1, label_action="A"):
    return types.SimpleNamespace(id=f"j{created}", status=types.SimpleNamespace(stage=stage, message=msg),
                                 created_at=created, labels={"action": label_action})


def test_an_infra_error_resumes_twice_then_halts_with_its_cause():
    plan = _plan([_st("x")])
    c = Constraints(error_resumes=2)
    no_unit = lambda job: True  # noqa: E731 -- the job's phase record has no first_unit
    assert C.resume_after_error(plan, "x", [_job(created=1)], c, infra=no_unit) is None
    why = C.resume_after_error(plan, "x", [_job(created=2), _job(created=1)], c, infra=no_unit)
    assert why and "infra" in why


def test_clear_halt_resets_the_error_count_and_refuses_when_not_halted():
    jobs = [_job(created=3), _job(created=2)]
    assert C._trailing_errors(jobs, operator_after=2.5) == 1
    with pytest.raises(SystemExit, match="not halted"):
        C.clear_halt_guard(halted=False)
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.** `resume_after_error(plan, stage, history, c, *, infra: Callable[[job], bool])`: after the
  `error_resumes` bound (now `_trailing_errors(history, operator_after=<newest operator record time>)`), a job for which
  `infra(job)` is True resumes (`None`); the halt reason names the class: `infra` (no `first_unit` in the job phase
  record), `image` (phase record has `after_fetch` but no `after_env_check`), `timeout`, `unit`. `infra` reads
  `phases.job_rel(label_of(job.labels), attempt)` — list `_ledger/_jobs/` once per step and match by the label prefix.
  `campaign <file> resume --clear-halt <instance>/<stage> --reason "<text>"`: refuse unless the stage's last step
  report is `halt` or its trailing errors reached `error_resumes`; then `upload_bytes(f"_operator/{pipeline}/{stage}/{token}/{utc}.json",
  {"who": git config user.email, "when": utc, "reason": text})`. It launches nothing.
- [ ] **Step 4:** Run. Expected: PASS.

---

## Phase 4 — sound reuse

### Task 11: Stage modules for the reusable set (S5.9 "Where a stage is declared")

**Files:**
- Create: `scripts/_ghost_gk_stages.py`, `scripts/_ghost_outfield_stages.py`, `scripts/_receiver_stages.py`,
  `scripts/_win_probability_stages.py`, `scripts/_xcross_stages.py`, `scripts/_xshot_stages.py`,
  `scripts/_census_stages.py`
- Modify: the 7 pipeline modules (S6.4)
- Create: `tests/scripts/test_stage_modules.py`

**The reusable set at `74d57ba`** (S2.4 plus upstreams): ghost_gk `validate`, `extract`; ghost_outfield `extract`;
receiver `extract`, `pool`; win_probability `refs`, `extract`; xcross `extract`, `probe`; xshot `extract`; census
`refs`, `fit_prepass`, `prefit`, `score`.

- [ ] **Step 1: Failing gate**

```python
# tests/scripts/test_stage_modules.py
"""S5.9: every callable of a stage lives in ONE stage module; a reusable stage's upstreams are stage modules too;
a stage module's closure never contains its trainer."""

from __future__ import annotations

import importlib
import inspect

import pytest

from scripts import _stages as s

REUSABLE = {
    "scripts.train_ghost_gk": ("validate", "extract"),
    "scripts.train_ghost_outfield": ("extract",),
    "scripts.train_receiver_model": ("extract", "pool"),
    "scripts.train_win_probability": ("refs", "extract"),
    "scripts.train_xcross_attempt": ("extract", "probe"),
    "scripts.train_xshot_occurrence": ("extract",),
    "scripts.build_territory_ranking_census": ("refs", "fit_prepass", "prefit", "score"),
}
_FIELDS = ("work", "load", "reduce", "reduce_order", "units", "refs", "key", "counters", "declare", "dataset_files")


def _callables(st):
    for f in _FIELDS:
        fn = getattr(st, f, None)
        if fn is not None and callable(fn) and not inspect.ismethod(fn):
            yield f, fn


@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_each_reusable_stage_is_declared_in_one_stage_module(module):
    pipe = importlib.import_module(module).PIPELINE
    for name in REUSABLE[module]:
        st = pipe.stage(name)
        homes = {f: fn.__module__ for f, fn in _callables(st)}
        assert len(set(homes.values())) == 1, (name, homes)
        home = next(iter(homes.values()))
        assert home != module and importlib.import_module(home).STAGES[name] is st


@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_a_reusable_stage_has_only_reusable_upstreams(module):
    pipe = importlib.import_module(module).PIPELINE
    for name in REUSABLE[module]:
        assert set(pipe.stage(name).needs) <= set(REUSABLE[module]), name

```

(The closure test "a stage module's closure never contains its trainer" needs `closures.module_closure`, so it lands
in Task 12's test file, r1 B CCTP-PLAN-07.)

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/test_stage_modules.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Move each reusable stage.** For each pipeline module, in this order (one at a time; run its own
  tests after each):
  1. Create `scripts/_<pipeline>_stages.py` with the module docstring "ADR-129: the reusable stages of
     `<trainer>`; their closure must not contain the trainer (`test_stage_modules`)."
  2. Move into it: the `Stage`/`ItemStage` declarations of the reusable stages (lambdas included), every function
     those declarations reference, and every helper those functions call that is not used by a fit-side stage. A
     helper used by BOTH sides moves to the stage module, and the trainer imports it from there.
  3. End the stage module with `STAGES = {st.name: st for st in (<the stage objects>)}`.
  4. In the trainer: `from scripts._<pipeline>_stages import STAGES as _REUSABLE` and build `PIPELINE.stages` with
     `_REUSABLE["extract"]` etc. in the SAME order as `74d57ba`. Re-export the moved private names the trainer's
     tests import (`git grep -n "train_ghost_gk\._" tests` lists them), each with a `# re-export: tests` comment.
  5. Check `docs/PRIVATE_CONSUMERS.md` (S6.8) for any moved name; the ghost_gk/outfield entries pin
     `silly_kicks/tracking/_ghost_gk.py` (a library module), not the trainer, so none applies at `74d57ba`; record the
     check in the task note.
  Census: `from _crossed_icc import …` (`build_territory_ranking_census.py:48-51`) stays where it is; the stage module
  imports `scripts._crossed_icc` by its package name.

- [ ] **Step 4:** Run `test_stage_modules.py` plus each pipeline's own tests
  (`git grep -l "<trainer module name>" tests | tr '\n' ' '`). Expected: PASS. Run the parity tests that pin these
  pipelines' outputs (`test_compute_contract.py`, the `_parity.py` users) in the background: every output must stay
  byte-identical (this task moves code only).

### Task 12: Closure computation, the lock and the CI gate (S5.9 "The code closure", "The lock")

**Files:** Create `scripts/hf_jobs/closures.py`, `scripts/hf_jobs/closures.lock.json`, `tests/scripts/test_closure_lock.py`.
Modify `scripts/_stages.py` (`Stage.data_files`, `ItemStage.data_files`, `Stage.hub_files`, `ItemStage.hub_files`),
`scripts/hf_jobs/__main__.py` (`closures`).

**Interfaces:**
- Produces: `module_closure(module: str, *, root: Path = REPO_ROOT) -> tuple[str, ...]` (sorted repo-relative POSIX
  paths); `RUNNER_MODULES = ("scripts._stages", "scripts._driver")`;
  `stage_membership(pipeline: Pipeline, stage: AnyStage, *, root) -> Membership(modules: tuple[str, ...],
  data: tuple[str, ...])`; `LOCK = Path(__file__).with_name("closures.lock.json")`; `load_lock() -> dict`;
  `record(*, root) -> dict`; `main(argv)` with `closures record`.
- `Stage.data_files: tuple[str, ...] = ()`, `Stage.hub_files: tuple[tuple[str, str, str], ...] = ()` (the same on
  `ItemStage`). NOTE: `Stage.dataset_files` (a callable over the DATASET mount) is a different field and stays.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/test_closure_lock.py
"""S5.9: the lock records each stage's closure and data membership; CI recomputes and compares."""

from __future__ import annotations

import pytest

from scripts.hf_jobs import closures


def test_the_lock_matches_the_computed_memberships():
    want = closures.record(write=False)
    have = closures.load_lock()
    assert have == want, "closure lock is stale: run `python -m scripts.hf_jobs closures record`"


def test_the_resolver_runs_on_the_real_tree_with_the_scripts_namespace_package():
    """r2 B R2-03: `scripts` has no __init__.py; the resolver must not crash and must still follow it."""
    import pathlib

    assert not pathlib.Path("scripts/__init__.py").exists()  # the real tree at 74d57ba
    got = closures.module_closure("scripts._stages")
    assert got and all(p.startswith(("scripts/", "silly_kicks/")) for p in got)


def test_a_package_init_that_an_import_executes_is_in_the_closure():
    got = closures.module_closure("scripts._ghost_gk_stages")
    assert "silly_kicks/__init__.py" in got  # `import silly_kicks.tracking...` executes it (R1 measures its width)


def test_a_function_level_import_is_in_the_closure():
    got = closures.module_closure("scripts._stages")
    assert "scripts/_driver.py" in got  # module-level
    assert "scripts/hf_jobs/store.py" in got  # imported inside `open_store` (r1 B CCTP-SPEC-13)


@pytest.mark.parametrize("module", sorted(__import__("tests.scripts.test_stage_modules", fromlist=["REUSABLE"]).REUSABLE))
def test_a_stage_module_closure_never_contains_its_trainer(module):
    import importlib

    from tests.scripts.test_stage_modules import REUSABLE, _callables

    pipe = importlib.import_module(module).PIPELINE
    home = next(iter({fn.__module__ for _, fn in _callables(pipe.stage(REUSABLE[module][0]))}))
    assert module.replace(".", "/") + ".py" not in closures.module_closure(home)


def test_the_census_bare_import_resolves_once():
    got = closures.module_closure("scripts._census_stages")
    assert got.count("scripts/_crossed_icc.py") == 1


def test_one_models_weights_change_only_the_stages_that_declare_them():
    lock = closures.load_lock()
    prefix = "silly_kicks/tracking/_ghost_gk_weights/default"
    readers = {k for k, v in lock.items() if any(p.startswith(prefix) for p in v["data"])}
    others = {k for k, v in lock.items() if k not in readers}
    assert readers and others  # non-vacuity: both sides exist
    for k in others:
        assert not any(p.startswith(prefix) for p in lock[k]["data"])
```

- [ ] **Step 2:** Run. Expected: FAIL (no module).

- [ ] **Step 3: Implement `closures.py`**

```python
"""The locked closure of every pipeline stage (ADR-129, S5.9).

The CODE closure of a stage = the static import closure of its stage module plus the runner modules, restricted to
repo files (``silly_kicks/**``, ``scripts/**``). It follows every ``import`` / ``from ... import`` in EVERY code
object (an `ast.walk` over the whole module, so imports inside functions count -- `modulefinder` semantics), and
it adds every parent package ``__init__.py`` an import executes. Resolution is a stdlib AST resolver, NOT
`modulefinder`: `scripts` is a namespace package (no ``scripts/__init__.py`` at 74d57ba), and `modulefinder` crashes on
it (``AttributeError: 'NoneType' object has no attribute 'is_package'``, r2 B R2-03). Names resolve against the job's
`sys.path` order: the repo root, then ``scripts/`` (the census bare ``from _crossed_icc import ...``). A name that
resolves to no repo file is third-party (``uv.lock`` and the image digest cover it). Files de-duplicate by path.
The DATA membership = the stage's declared ``data_files`` prefixes, expanded with ``git ls-files``. The lock records
MEMBERSHIP only (paths, never digests): digests come from HEAD at plan time."""

from __future__ import annotations

import ast
import functools
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK = pathlib.Path(__file__).with_name("closures.lock.json")
RUNNER_MODULES = ("scripts._stages", "scripts._driver")
_REPO_DIRS = ("silly_kicks", "scripts")


class ClosureViolationError(RuntimeError):
    """A stage process loaded a repo module, opened a tracked repo file, or saw an environment variable outside its
    locked closure / data membership / registry. The unit writes no result."""


def _bases(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return (root, root / "scripts")  # the job's sys.path order (`_stages.main` runs with both)


def _file_of(name: str, root: pathlib.Path) -> pathlib.Path | None:
    """The repo file of module ``name`` (``a.b`` -> ``a/b.py`` or ``a/b/__init__.py``), or None (third-party, or a
    namespace package with no ``__init__.py``)."""
    parts = name.split(".")
    for base in _bases(root):
        stem = base.joinpath(*parts)
        for cand in (stem.with_suffix(".py"), stem / "__init__.py"):
            if cand.is_file() and _inside(cand, root):
                return cand
    return None


def _package_inits(name: str, root: pathlib.Path) -> list[pathlib.Path]:
    """Every ``__init__.py`` of the parents of ``name`` that an import executes (``a.b.c`` -> ``a/__init__.py``,
    ``a/b/__init__.py``), when they exist (a namespace package has none)."""
    out = []
    parts = name.split(".")
    for i in range(1, len(parts)):
        f = _file_of(".".join(parts[:i]), root)
        if f is not None and f.name == "__init__.py":
            out.append(f)
    return out


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts[:-1]) if rel.name == "__init__" else list(rel.parts)
    return ".".join(parts)


def _imports(path: pathlib.Path, root: pathlib.Path) -> set[str]:
    """Every module name ``path`` imports, anywhere in the file; relative imports made absolute."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    me = _module_name(path, root)
    pkg = me if path.name == "__init__.py" else me.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = pkg.split(".")[: len(pkg.split(".")) - node.level + 1] if pkg else []
                base = ".".join([*anchor, *(node.module.split(".") if node.module else [])])
            else:
                base = node.module or ""
            if base:
                names.add(base)
            names |= {f"{base}.{a.name}" if base else a.name for a in node.names if a.name != "*"}  # submodules
    return names


def _inside(path, root: pathlib.Path) -> bool:
    try:
        rel = pathlib.Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in _REPO_DIRS


@functools.lru_cache(maxsize=None)
def module_closure(module: str, *, root: pathlib.Path = REPO_ROOT) -> tuple[str, ...]:
    start = _file_of(module, root)
    if start is None:
        raise ValueError(f"{module}: not a repo module")
    seen: set[pathlib.Path] = set()
    todo = [start, *_package_inits(module, root)]
    while todo:
        f = todo.pop().resolve()
        if f in seen:
            continue
        seen.add(f)
        for name in _imports(f, root):
            hit = _file_of(name, root)
            if hit is not None:
                todo.append(hit)
                todo.extend(_package_inits(name, root))
    return tuple(sorted(p.relative_to(root.resolve()).as_posix() for p in seen))


def _ls_files(prefixes: Sequence[str], root: pathlib.Path) -> tuple[str, ...]:
    if not prefixes:
        return ()
    out = subprocess.run(["git", "ls-files", "--", *prefixes], capture_output=True, text=True, check=True, cwd=root)  # noqa: S603,S607
    return tuple(sorted(line for line in out.stdout.splitlines() if line))


def stage_module(st) -> str:
    return st.work.__module__


def stage_membership(st, *, root: pathlib.Path = REPO_ROOT) -> dict[str, list[str]]:
    mods = set(module_closure(stage_module(st), root=root))
    for r in RUNNER_MODULES:
        mods |= set(module_closure(r, root=root))
    return {"modules": sorted(mods), "data": list(_ls_files(tuple(st.data_files), root))}


def _population() -> list[str]:
    """The compliant pipeline modules, single-sourced in production code (`scripts/_pipeline_population.py`)."""
    from scripts._pipeline_population import compliant

    return [f"scripts.{name}" for name in compliant()]


def record(*, write: bool = True, root: pathlib.Path = REPO_ROOT) -> dict:
    import importlib

    lock: dict[str, dict] = {}
    for mod in _population():
        pipe = importlib.import_module(mod).PIPELINE
        for st in pipe.stages:
            lock[f"{pipe.name}/{st.name}"] = stage_membership(st, root=root)
    if write:
        LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["record"]:
        raise SystemExit("usage: python -m scripts.hf_jobs closures record")
    record()
    return 0
```

**The population moves to production code (decided, r1 B CCTP-PLAN-11).** Create `scripts/_pipeline_population.py`
holding `NOT_A_PIPELINE` and `LEGACY_UNMIGRATED` (moved verbatim from `tests/scripts/test_compute_contract.py:21`,
`:40`, with their approvals) and `population()`, `declares_pipeline()`, `compliant()` (moved from the test's
`_population`, `_declares_pipeline`, `_compliant`, `:106-125`). Those use two helpers of
`tests/scripts/_corpus_load_rules.py` (`all_script_trees`, `_call_name`, `:107-113`; r2 B R2-04) and the test's own
`_calls` (`test_compute_contract.py:102`, r3 B R3-03): all three move into
`scripts/_pipeline_population.py` as `all_script_trees()`, `call_name()` and `calls()`, and `_corpus_load_rules.py` imports them
from there (one definition; its other users keep their imports). `test_compute_contract.py` imports the five
population names from the production module and keeps every assertion. A job never imports `tests/`: a test
(`test_closure_lock.py`) runs `python -c "import scripts._pipeline_population, sys; assert not any(m.startswith('tests')
for m in sys.modules)"` in a fresh interpreter.

`closures.py` defines `ClosureViolationError` (Tasks 12b and 14 import it from here).

Declare `data_files=` on every stage that opens package data. Start from S2.9's list: the `silly_kicks/tracking/_*_weights` readers, `win_probability/_model.py`,
`match_outcome/_dependence.py`, `xtgk/_retention.py`, `scripts/_events_admission.py`'s verdicts JSON. Task 14 Step 6 then runs every toy with the file audit hook and adds
any declaration the hook finds missing. Run `python -m scripts.hf_jobs closures record` and write the lock (again
after Task 14 Step 6).

- [ ] **Step 4:** Run `test_closure_lock.py`. Expected: PASS.
- [ ] **Step 5: The R1 measurement (report to the owner).** Per reusable stage: the module count, any
  `silly_kicks/**/__init__.py` that pulls in modules the stage never uses, and the HARNESS members that reach every
  closure through `_stages.py`'s function-level imports (today `scripts.hf_jobs.executor`, `store`, `datasets`,
  `_parallel_launch`; after Task 13 also `closures` and `env_contract`; r1 B CCTP-PLAN-09). An edit to a harness member
  changes every token, which costs a re-run, never soundness. Report the list; any boundary that excludes harness
  modules from the closure is a spec change and needs an owner ruling.

### Task 12b: The environment registry and its gate (S5.9 "Environment variables")

Task 13's token reads `env_contract.output_values`, so the registry lands first (r1 B CCTP-PLAN-07).

**Files:** Create `scripts/hf_jobs/env_contract.py`, `tests/scripts/test_env_contract.py`.

**Interfaces:**
- Produces: `NEUTRAL`, `OUTPUT`, `FORBIDDEN_IN_JOB` (frozensets of full names), `FORBIDDEN_PREFIXES`,
  `PLATFORM_NAMES`, `UNRESOLVED`, `registered(name) -> bool`, `output_values(environ) -> dict[str, str | None]`,
  `check_job_env(environ) -> None` (raises `closures.ClosureViolationError`),
  `env_reads(tree, *, module_path, search=None) -> list[str | UNRESOLVED]`.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/test_env_contract.py
"""S5.9: every environment read in silly_kicks/ and scripts/ is registered; the AST gate resolves constants."""

from __future__ import annotations

import ast
import pathlib

from scripts.hf_jobs import env_contract as ec


def _reads(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ec.env_reads(tree, module_path=path)  # names, or the sentinel UNRESOLVED


def test_every_env_read_is_registered():
    bad = []
    for root in ("silly_kicks", "scripts"):
        for p in pathlib.Path(root).rglob("*.py"):
            for name in _reads(p):
                if name is ec.UNRESOLVED or not (ec.registered(name) or name in ec.PLATFORM_NAMES):
                    bad.append((str(p), name))
    assert not bad, bad


def test_the_gate_resolves_an_imported_constant(tmp_path):
    (tmp_path / "a.py").write_text("ENV = 'SILLY_KICKS_NEW_THING'\n")
    (tmp_path / "b.py").write_text("import os\nfrom a import ENV\nv = os.environ.get(ENV)\n")
    got = ec.env_reads(ast.parse((tmp_path / "b.py").read_text()), module_path=tmp_path / "b.py", search=[tmp_path])
    assert got == ["SILLY_KICKS_NEW_THING"]


def test_a_forbidden_variable_fails_a_job():
    import pytest

    from scripts.hf_jobs.closures import ClosureViolationError

    with pytest.raises(ClosureViolationError, match="SILLY_KICKS_GHOST_GK_PATH"):
        ec.check_job_env({"SILLY_KICKS_IN_HF_JOB": "1", "SILLY_KICKS_GHOST_GK_PATH": "/x"})
    with pytest.raises(ClosureViolationError, match="SILLY_KICKS_UNKNOWN"):
        ec.check_job_env({"SILLY_KICKS_IN_HF_JOB": "1", "SILLY_KICKS_UNKNOWN": "1"})
```

- [ ] **Step 2:** Run: `.venv/Scripts/python -m pytest tests/scripts/test_env_contract.py -q -p no:cacheprovider`. Expected: FAIL.

- [ ] **Step 3: Implement `env_contract.py`** — the S5.9 table exactly:

```python
NEUTRAL = frozenset({f"SILLY_KICKS_{n}" for n in (
    "RUN_KEY", "JOB_LABEL", "ATTEMPT_ID", "ACTION_KIND", "PHASE_FILE", "IMAGE", "IN_HF_JOB", "DATASET_READ_CAP",
    "DATASET_READ_GATE_DIR", "STORE_READ_CAP", "NUMBA_CACHE", "CORPUS_CACHE_DIR", "IMPORTANCE_PID_LOG", "DEBUG_MODULES",
)})
OUTPUT = frozenset({f"SILLY_KICKS_{n}" for n in (
    "GHOST_FORCE_NUMPY", "DAS_FORCE_NUMPY", "COORDINATION_FORCE_NUMPY", "COORDINATION_REFERENCE_NUMERICS",
    "ASSERT_INVARIANTS", "OPEN_DATA_REVISION",
)})
FORBIDDEN_IN_JOB = frozenset(
    {f"SILLY_KICKS_{n}" for n in ("OPEN_DATA_BASE", "PINING_MANIFEST_DIR", "GHOST_GK_PATH", "GHOST_OUTFIELD_PATH")}
    | {"PINING_API_URL", "PINING_FOR_THE_DATA_TOKEN", "SB_USERNAME", "SB_PASSWORD"}
)
FORBIDDEN_PREFIXES = ("DATABRICKS_", "SK_DAS_")
PLATFORM_NAMES = frozenset({"PATH", "HOME", "TMPDIR", "HF_TOKEN", "HF_HOME", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
                            "HF_HUB_OFFLINE", "NUMBA_CACHE_DIR", "NUMBA_NUM_THREADS",
                            *__import__("scripts._thread_pin", fromlist=["DETERMINISTIC_VARS"]).DETERMINISTIC_VARS})
```

Plus `registered(name)`, `output_values(environ) = {n: environ.get(n) for n in sorted(OUTPUT)}`,
`check_job_env(environ)` (only when `SILLY_KICKS_IN_HF_JOB` is set: any `SILLY_KICKS_*` not registered, any
`FORBIDDEN_IN_JOB` name, any name starting with a `FORBIDDEN_PREFIXES` entry -> `ClosureViolationError` naming it),
and `env_reads(tree, *, module_path, search=None)` (the AST visitor: `os.environ[X]`, `os.environ.get(X)`,
`os.getenv(X)`, `environ.get` after `from os import environ`; `X` a literal, a module-level `Name` bound to a
literal, or an imported `Name` resolved by reading the source module from `search` (default the repo root and
`scripts/`); anything else -> `UNRESOLVED`). Re-run the S5.9 inventory grep first and class any new name it finds
(record each in the task note).

- [ ] **Step 4:** Re-run Step 2's command. Expected: PASS.

### Task 13: Per-stage code identity in the token; `cross_commit_version` deleted (S5.9 "The token")

**Files:** Modify `scripts/_stages.py` (`compute_tokens`, `make_context`, new `stage_code_identity`; delete
`cross_commit_version` from `Stage`/`ItemStage`), the 7 pipeline modules (delete the 11 `cross_commit_version=`),
`docs/howto/compute-pipelines.md:42`, ADR-128 P9 row (Task 19). Tests: `tests/scripts/test_stage_contract.py`,
`tests/scripts/test_compute_contract.py`.

**Interfaces:**
- Produces: `stage_code_identity(pipeline: Pipeline, st: AnyStage, *, root: Path, env_values: Mapping[str, str | None],
  image: str | None, python: str) -> str | None` (None: the stage is absent from the lock).

- [ ] **Step 1: Failing tests** (replace `test_cross_commit_version_replaces_the_code_identity`)

```python
def test_a_clean_tree_token_follows_the_closure_not_the_commit(tmp_path, monkeypatch):
    """S5.9: the code field digests the stage's closure files; a commit that changes no closure file keeps it."""
    from scripts.hf_jobs import closures

    p = _p()
    lock = {"t/a": {"modules": ["m/a.py"], "data": []}, "t/b": {"modules": ["m/b.py"], "data": []}}
    (tmp_path / "m").mkdir()
    (tmp_path / "m/a.py").write_text("A = 1\n")
    (tmp_path / "m/b.py").write_text("B = 1\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    monkeypatch.setattr(closures, "load_lock", lambda: lock)
    ids = lambda: {st.name: s.stage_code_identity(p, st, root=tmp_path, env_values={}, image="img@sha256:1", python="3.12.14") for st in p.stages}  # noqa: E731
    before = ids()
    (tmp_path / "m/b.py").write_text("B = 2\n")
    after = ids()
    assert before["a"] == after["a"] and before["b"] != after["b"]
    assert s.stage_code_identity(p, p.stage("a"), root=tmp_path, env_values={"SILLY_KICKS_DAS_FORCE_NUMPY": "1"},
                                 image="img@sha256:1", python="3.12.14") != before["a"]


def test_a_stage_absent_from_the_lock_keeps_the_commit_identity(tmp_path, monkeypatch):
    from scripts.hf_jobs import closures

    monkeypatch.setattr(closures, "load_lock", lambda: {})
    ctx = _ctx(tmp_path)  # clean prov, pipeline "t" is in no lock
    assert {ctx.token_inputs[n]["code"] for n in ("a", "b")} == {"c" * 40}
    monkeypatch.setenv("SILLY_KICKS_IN_HF_JOB", "1")
    monkeypatch.setattr(s, "_is_compliant", lambda name: True)
    with pytest.raises(closures.ClosureViolationError, match="lock"):
        _ctx(tmp_path / "job")


def test_a_dirty_tree_keys_every_stage_on_the_tree_hash(tmp_path):
    prov = {"commit": "c" * 40, "tree_state": "dirty", "code_identity": "dirty-abc"}
    ctx = s.make_context(_p(), {"x": 1}, prov, store=s.LocalStore(tmp_path), data_root=None)
    assert {ctx.token_inputs[n]["code"] for n in ("a", "b")} == {"dirty-abc"}
```

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement**

```python
def stage_code_identity(pipeline, st, *, root, env_values, image, python) -> str | None:
    """ADR-129: the digest of the stage's locked closure files, its data members, ``uv.lock``, the image digest,
    the Python version and the output-affecting environment values. A commit that changes none of these keeps
    the identity, so its results are reused (INV-10)."""
    from scripts.hf_jobs.closures import load_lock

    entry = load_lock().get(f"{pipeline.name}/{st.name}")
    if entry is None:
        return None  # not in the lock: the caller keeps the commit identity (no reuse across commits)
    h = hashlib.sha256()
    for rel in [*entry["modules"], *entry["data"], "uv.lock"]:
        h.update(rel.encode("utf-8") + b"\0" + (pathlib.Path(root) / rel).read_bytes() + b"\0")
    for repo, rev, path in sorted(getattr(st, "hub_files", ())):
        h.update(f"hub:{repo}@{rev}:{path}\0".encode())
    h.update(json.dumps({"image": image, "python": python, "env": dict(sorted(env_values.items()))}, sort_keys=True).encode())
    return "closure-" + h.hexdigest()
```

**A stage absent from the lock (r1 B CCTP-PLAN-03).** `stage_code_identity` returns `None`, and the token keeps the
commit identity (`code_id`, today's rule): such a stage never reuses across commits, which is the safe side. The lock
covers every compliant pipeline (the Task 12 gate), so in production only toy and test pipelines take this path. In
an HF job (`SILLY_KICKS_IN_HF_JOB`) a COMPLIANT pipeline's stage that is absent from the lock refuses the run
(`ClosureViolationError`): the lock is stale. Tests: the Task 2 pipeline `"t"` and `_stage_harness.PROV` contexts
build with the commit identity and no `KeyError`; a compliant stage absent from the lock in a job refuses.
`_stages._is_compliant(pipeline_name) -> bool` answers "is this pipeline in `_pipeline_population.compliant()`"
(matching by `PIPELINE.name`; cached per process).

`compute_tokens(pipeline, params, code_id, dataset_identity, *, dataset_identities=None, stage_codes=None)`: the
`code` field is `stage_codes.get(st.name) or code_id`. `make_context`: when `prov["tree_state"] == "clean"`, build
`stage_codes` with `stage_code_identity(..., root=REPO_ROOT, env_values=env_contract.output_values(os.environ),
image=_image_identity(), python=platform.python_version())`; else `None` (the dirty rule). `_image_identity()` =
`os.environ.get(IMAGE_ENV) or launch.IMAGE` in this task; Task 18 changes its fallback to `image.image_ref()` (ONE
name, defined in Task 18; r1 B CCTP-PLAN-07). Cache per process (`functools.lru_cache` on a (pipeline name, stage name) key) so `--plan` hashes each file
once.

Delete `cross_commit_version` from both dataclasses and the 11 call sites; update `test_compute_contract` (remove its
`cross_commit_version` checks) and `compute-pipelines.md:42`.

- [ ] **Step 4:** Run `test_stage_contract.py`, `test_compute_contract.py`, `test_trainer_cache_and_providers.py`,
  `test_env_provenance.py`. Expected: PASS.

### Task 14: The parameter gate and the runtime checks (S5.9)

**Files:** Create `scripts/hf_jobs/stage_checks.py`, `tests/scripts/test_closure_runtime_check.py`. Modify `scripts/_stages.py`
(`context_to_json`, new `stage_context_from_json`, `PipelineView`), `scripts/hf_jobs/executor.py` (`worker_main`,
`reduce_main`, `InnerPool` initializer), `tests/scripts/test_stage_modules.py` (the parameter gate).

**Interfaces:**
- Produces (stage_checks): `ClosureViolationError` re-exported from `closures`; `install_stage_checks(stage_key: str,
  *, root: Path, entry: Mapping | None = None, tracked: Collection[str] | None = None) -> None` (``entry`` defaults
  to the lock entry of ``stage_key``); `check_modules(stage_key) -> None`.
- Produces (_stages): `@dataclass PipelineView(name, module, params, stages)`; `stage_context_from_json(text, stage)
  -> tuple[RunContext, AnyStage]`.

- [ ] **Step 1: Failing tests**

(`tests/scripts/test_env_contract.py` is written in Task 12b.)

```python
# tests/scripts/test_closure_runtime_check.py
"""S5.9 runtime checks, in a FRESH interpreter (the worker), fail closed."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_HARNESS = textwrap.dedent(
    """
    import sys, pathlib
    from scripts.hf_jobs.stage_checks import install_stage_checks, check_modules, ClosureViolationError
    entry = {{"modules": {mods!r}, "data": {data!r}}}
    install_stage_checks("t/a", root=pathlib.Path.cwd(), entry=entry, tracked={tracked!r})
    try:
        {body}
        check_modules("t/a")
        print("PASS")
    except ClosureViolationError as exc:
        print("FAIL", exc)
    """
)


def _run(body, mods=("scripts/_stages.py",), data=(), tracked=None, extra_env=None):
    import os

    code = _HARNESS.format(mods=list(mods), data=list(data), tracked=tracked, body=body)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **(extra_env or {})}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    return (out.stdout or "") + (out.stderr or "")


def test_a_dynamic_import_outside_the_closure_fails():
    assert "FAIL" in _run("import importlib; importlib.import_module('scripts._toml')")


def test_a_lazy_import_of_a_closure_module_passes():
    assert "PASS" in _run("import scripts._toml", mods=("scripts/_stages.py", "scripts/_toml.py"))


def test_an_undeclared_tracked_data_file_fails_and_a_member_passes():
    tracked = ["docs/research/skillcorner_s1_event_validity/verdicts.json", "uv.lock"]
    body = "open('docs/research/skillcorner_s1_event_validity/verdicts.json','rb').read()"
    assert "FAIL" in _run(body, tracked=tracked)
    assert "PASS" in _run(body, data=tracked[:1], tracked=tracked)
    assert "PASS" in _run("open('uv.lock','rb').read()", tracked=tracked)


def test_a_third_party_read_in_the_venv_passes():
    """The local and CI venv is `<repo>/.venv`, under the repo root: neither check may flag it (r1 B PLAN-04)."""
    assert "PASS" in _run("import json, pandas; open(pandas.__file__,'rb').read()", tracked=["uv.lock"])


def test_a_module_preloaded_outside_the_closure_fails_at_the_first_check():
    code = "import scripts._toml\n" + _HARNESS.format(mods=["scripts/_stages.py"], data=[], tracked=None, body="pass")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    assert "FAIL" in out.stdout or "ClosureViolationError" in out.stderr


def test_a_sourceless_pyc_of_a_non_closure_module_fails():
    body = textwrap.dedent(
        """
        import importlib.machinery, importlib.util, py_compile, tempfile, pathlib, sys
        pyc = py_compile.compile("scripts/_toml.py", cfile="scripts/__pycache__/_sk_probe_toml.pyc", doraise=True)
        loader = importlib.machinery.SourcelessFileLoader("_sk_probe_toml", pyc)
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("_sk_probe_toml", loader))
        loader.exec_module(mod); sys.modules["_sk_probe_toml"] = mod
        """
    ).replace("\n", "\n        ")
    try:
        assert "FAIL" in _run(body)
    finally:
        import pathlib

        pathlib.Path("scripts/__pycache__/_sk_probe_toml.pyc").unlink(missing_ok=True)


def test_a_toy_corpus_file_under_tests_passes_as_a_dataset_root():
    """A REAL tracked file (r2 B R2-05): it passes under its dataset root and fails without the root."""
    root = "tests/scripts/_fixtures"
    path = f"{root}/d2_calibration_golden.json"  # tracked at 74d57ba
    body = f"open('{path}','rb').read()"
    assert "PASS" in _run(body, tracked=[path, "uv.lock"], extra_env={"SILLY_KICKS_DATA_ROOT": root})
    assert "FAIL" in _run(body, tracked=[path, "uv.lock"])


def test_an_undeclared_hub_download_fails_offline(tmp_path):
    code = ("from huggingface_hub import snapshot_download\n"
            "try:\n    snapshot_download('silly-kicks/ghost-gk', revision='main')\n    print('PASS')\n"
            "except Exception as exc:\n    print('FAIL', type(exc).__name__)")
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert out.stdout.startswith("FAIL")


def test_declared_hub_files_are_prefetched_and_key_the_token(monkeypatch):
    from scripts import _stages as s

    calls = []
    st = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1,
                 hub_files=(("silly-kicks/ghost-gk", "abc123", "weights/model.json"),))
    s.prefetch_hub_files([st], fetch=lambda repo, rev, path: calls.append((repo, rev, path)))
    assert calls == [("silly-kicks/ghost-gk", "abc123", "weights/model.json")]
    p = s.Pipeline(name="t", module="m", stages=(st,))
    st2 = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1,
                  hub_files=(("silly-kicks/ghost-gk", "def456", "weights/model.json"),))
    p2 = s.Pipeline(name="t", module="m", stages=(st2,))
    kw = dict(root=pathlib.Path("."), env_values={}, image="i", python="3.12")
    import scripts.hf_jobs.closures as cl

    monkeypatch.setattr(cl, "load_lock", lambda: {"t/a": {"modules": [], "data": []}})  # never leaks (r2 B R2-06)
    assert s.stage_code_identity(p, st, **kw) != s.stage_code_identity(p2, st2, **kw)


def test_a_stage_unit_that_downloads_undeclared_hub_files_fails_under_the_checks(tmp_path):
    """r2 B R2-09: not only the library -- a UNIT under `install_stage_checks` fails, and records the failure."""
    code = textwrap.dedent(
        f"""
        import pathlib
        from scripts import _stages as s
        from scripts.hf_jobs.stage_checks import install_stage_checks
        def work(k, ctx):
            from huggingface_hub import snapshot_download
            return {{"p": snapshot_download("silly-kicks/ghost-gk", revision="main")}}
        st = s.Stage(name="a", units=lambda c: ["only"], work=work)
        pipe = s.Pipeline(name="t", module="m", stages=(st,))
        prov = {{"commit": "c" * 40, "tree_state": "dirty", "code_identity": "dirty-x"}}
        ctx = s.make_context(pipe, {{}}, prov, store=s.LocalStore(pathlib.Path({str(tmp_path)!r})), data_root=None)
        from scripts.hf_jobs.closures import module_closure  # the REAL runner closure (r3 B R3-01)
        install_stage_checks("t/a", root=pathlib.Path.cwd(), entry={{"modules": list(module_closure("scripts._stages")), "data": []}})
        print("OUTCOME", s.run_unit(ctx, st, "only", worker="w")[0])
        """
    )
    import os

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)
    assert "OUTCOME failed" in out.stdout, out.stdout + out.stderr


def test_the_worker_and_the_reduce_load_a_reusable_stage_without_its_trainer(tmp_path):
    import importlib
    import json
    import os

    from scripts._stages import LocalStore, context_to_json, make_context, sorted_units
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for
    from tests.scripts._stage_harness import PROV, params_from_args

    mod = "scripts.train_ghost_outfield"
    pipe = importlib.import_module(mod).PIPELINE
    data = build_toy_for(mod, tmp_path / "d")
    ctx = make_context(pipe, params_from_args(pipe, TOY_PARAMS[mod]), PROV, store=LocalStore(tmp_path / "s"),
                       data_root=data)
    keys = sorted_units(ctx, pipe.stage("extract"))
    (tmp_path / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
    (tmp_path / "pending.json").write_text(json.dumps(keys), encoding="utf-8")
    for d in ("claims", "outcomes"):
        (tmp_path / d).mkdir()
    # the executor starts workers with HF_HUB_OFFLINE=1 (Task 14); a test that starts them directly sets it too
    env = {**os.environ, "SILLY_KICKS_DEBUG_MODULES": "1", "SILLY_KICKS_STAGE_CHECKS": "1", "HF_HUB_OFFLINE": "1"}
    base = [sys.executable, "-m", "scripts.hf_jobs.executor"]
    worker = subprocess.run([*base, "worker", "--ctx", str(tmp_path / "ctx.json"), "--stage", "extract",
                             "--pending", str(tmp_path / "pending.json"), "--claims", str(tmp_path / "claims"),
                             "--outcomes", str(tmp_path / "outcomes"), "--worker-id", "w0"],
                            capture_output=True, text=True, env=env, timeout=900)
    reduce = subprocess.run([*base, "reduce", "--ctx", str(tmp_path / "ctx.json"), "--stage", "extract"],
                            capture_output=True, text=True, env=env, timeout=900)
    for out in (worker, reduce):
        assert out.returncode == 0, out.stderr[-2000:]
        mods = next(line for line in out.stdout.splitlines() if line.startswith("MODULES:"))
        assert "scripts.train_ghost_outfield" not in mods.split()


def test_an_inner_task_that_opens_an_undeclared_file_fails():
    code = textwrap.dedent(
        """
        from scripts.hf_jobs.executor import InnerPool
        from scripts.hf_jobs.stage_checks import ClosureViolationError
        from tests.scripts._toy_pipeline import open_undeclared
        pool = InnerPool(2, stage_checks=("t/a", {"modules": ["scripts/_stages.py"], "data": []}))
        try:
            pool.map_keyed(open_undeclared, {"k1": (0,), "k2": (1,)})
            print("PASS")
        except ClosureViolationError as exc:
            print("FAIL", exc)
        finally:
            pool.shutdown()
        """
    )
    import os

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}  # the pool initializer refuses a process without it (r2 B R2-02)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert "FAIL" in out.stdout and "BrokenProcessPool" not in out.stdout + out.stderr, out.stdout + out.stderr
```

Add to `tests/scripts/_toy_pipeline.py` (importable by a spawned child as `tests.scripts._toy_pipeline`):

```python
def open_undeclared(_i: int) -> int:
    """An inner task that opens a TRACKED repo file outside any closure (test_closure_runtime_check)."""
    return len(open("docs/research/skillcorner_s1_event_validity/verdicts.json", "rb").read())
```

In `worker_main` and `reduce_main`, at exit when `SILLY_KICKS_DEBUG_MODULES` is set (register it `neutral`):
`print("MODULES: " + " ".join(sorted(sys.modules)))`. `InnerPool.__init__` gains `stage_checks: tuple[str, dict] |
None = None`; when set, the spawn initializer calls `install_stage_checks(key, root=REPO_ROOT, entry=entry)` before
the existing `_inner_init`. The worker passes `(stage key, lock entry)` when it builds the pool.

```python
# tests/scripts/test_stage_modules.py — add the parameter gate (S5.9 "Declared parameters")
@pytest.mark.parametrize("module", sorted(REUSABLE))
def test_every_parameter_a_reusable_stage_reads_moves_its_token(module, tmp_path):
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for
    from scripts.hf_jobs.closures import stage_module
    from tests.scripts._param_reads import param_reads, second_value

    pipe = importlib.import_module(module).PIPELINE
    base = _toy_params(pipe, tmp_path)  # parse TOY_PARAMS[module] through build_parser
    for name in REUSABLE[module]:
        st = pipe.stage(name)
        reads = param_reads(importlib.import_module(stage_module(st)), st)
        for p in reads:
            t0 = s.compute_tokens(pipe, base, "c", **_identities(pipe, base))[1][name]
            moved = {**base, p: second_value(pipe, p, base[p])}
            t1 = s.compute_tokens(pipe, moved, "c", **_identities(pipe, moved))[1][name]
            assert t0 != t1, f"{module}/{name} reads param {p!r} but its token ignores it"


def _identities(pipe, params):
    """The dataset identities built from the parameter VALUES, as `_dataset_identity` keys them in production."""
    others = {st.dataset_param for st in pipe.stages if getattr(st, "uses_dataset", False) and st.dataset_param != "dataset"}
    return {"dataset_identity": {"name": params.get("dataset")},
            "dataset_identities": {n: {"name": params.get(n)} for n in sorted(others)}}
```

Create `tests/scripts/_param_reads.py` with `param_reads(module, st) -> set[str]` (the AST read set: direct
`ctx.params[...]`/`.get(...)`, aliases `p = ctx.params`, and helpers that receive `ctx.params` or `ctx` followed into
repo modules) and `second_value(pipe, name, value)` (the next valid value: a `choices` neighbour, `not value` for a
flag, `value + 1` for an int, `value + "_x"` for a str, a sibling path for a Path). Two rules (r1 B CCTP-PLAN-05):

- **A key passed as an argument** (receiver `_staged_refs(ctx, param)` reads `ctx.params[param]`,
  `train_receiver_model.py:406-410`): the gate follows the helper's CALL SITES in the stage module and takes the
  literal argument at each (`_staged_refs(ctx, "dataset")`, `_staged_refs(ctx, "pool_dataset")`). A call site whose
  argument is not a literal fails the gate, naming the site.
- **A dataset parameter** (a name that is some stage's `dataset_param`, e.g. `dataset`, `pool_dataset`,
  `probe_dataset`) is keyed through the dataset identity, not `params=`. The token test passes
  `dataset_identity={"name": value}` and `dataset_identities={name: {"name": value}}` built from the parameter values,
  so changing the parameter changes the identity and the token, exactly as `_dataset_identity` does in production.

Gate tests in `test_stage_modules.py`: a helper with a literal call site resolves; a non-literal call site fails; a
dataset parameter moves the token through its identity.

- [ ] **Step 2:** Run the three files. Expected: FAIL.

- [ ] **Step 3:** (`env_contract.py` exists from Task 12b.)

- [ ] **Step 4: Implement `stage_checks.py`**

```python
"""Fail-closed runtime checks for a stage process (ADR-129, S5.9): modules, tracked-file opens, environment."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


from scripts.hf_jobs.closures import ClosureViolationError  # noqa: F401 -- re-export (one definition)

_STATE: dict = {}
_HARNESS = ("scripts/hf_jobs/executor.py", "scripts/hf_jobs/stage_checks.py", "scripts/hf_jobs/env_contract.py",
            "scripts/hf_jobs/closures.py", "scripts/hf_jobs/phases.py", "scripts/_thread_pin.py", "scripts/_mem_cap.py",
            "scripts/_parallel_launch.py", "scripts/_dataset_reads.py", "scripts/__init__.py", "scripts/hf_jobs/__init__.py")


def _tracked(root: pathlib.Path) -> frozenset[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True, cwd=root)  # noqa: S603,S607
    return frozenset(out.stdout.splitlines())


def install_stage_checks(stage_key: str, *, root: pathlib.Path, entry=None, tracked=None) -> None:
    from scripts.hf_jobs import env_contract
    from scripts.hf_jobs.closures import load_lock

    entry = entry if entry is not None else load_lock()[stage_key]
    env_contract.check_job_env(os.environ)
    if os.environ.get("HF_HUB_OFFLINE") != "1":  # set by the PROCESS ENVIRONMENT, before any import (r1 B PLAN-08)
        raise ClosureViolationError(f"{stage_key}: a stage process must start with HF_HUB_OFFLINE=1")
    allowed_open = set(entry["modules"]) | set(entry["data"]) | {"uv.lock"}
    _STATE["root"] = root.resolve()  # `_rel` below needs it
    _STATE.update(root=root.resolve(), key=stage_key, modules=set(entry["modules"]) | set(_HARNESS),
                  allowed=allowed_open, tracked=frozenset(tracked) if tracked is not None else _tracked(root),
                  # repo-relative, like every path the hook compares (an absolute ctx.data_root normalizes here;
                  # a data root outside the repo has no tracked files to guard; r3 B R3-02)
                  data_roots=tuple(r for d in (os.environ.get("SILLY_KICKS_DATA_ROOT"),) if d
                                   for r in (_rel(d),) if r and r != "."))
    sys.addaudithook(_audit)
    check_modules(stage_key)


def _rel(path) -> str | None:
    try:
        return pathlib.Path(path).resolve().relative_to(_STATE["root"]).as_posix()
    except (ValueError, OSError, TypeError):
        return None


def _audit(event: str, args) -> None:
    if event != "open" or not _STATE:
        return
    rel = _rel(args[0]) if args and isinstance(args[0], (str, bytes, os.PathLike)) else None
    if rel is None or rel not in _STATE["tracked"] or rel in _STATE["allowed"]:
        return
    if any(rel.startswith(r.rstrip("/") + "/") for r in _STATE["data_roots"]):
        return
    raise ClosureViolationError(f"{_STATE['key']}: opened tracked repo file {rel!r} outside its closure and data members")


def _module_source(path: str) -> str | None:
    """The repo-relative SOURCE path of a loaded module, for files under ``silly_kicks/`` or ``scripts/`` only (a
    venv inside the repo root is never a repo module, r1 B CCTP-PLAN-04). A sourceless ``.pyc`` maps to the source
    it was compiled from."""
    import importlib.util

    if path.endswith(".pyc"):
        try:
            path = importlib.util.source_from_cache(path)
        except ValueError:  # a sourceless .pyc outside __pycache__: map by name
            path = path[:-1]
    rel = _rel(path)
    if rel is None or not rel.endswith(".py") or rel.split("/", 1)[0] not in ("silly_kicks", "scripts"):
        return None
    return rel


def check_modules(stage_key: str) -> None:
    bad = sorted(
        rel for m in list(sys.modules.values())
        if (rel := _module_source(getattr(m, "__file__", None) or "")) and rel not in _STATE["modules"]
    )
    if bad:
        raise ClosureViolationError(f"{stage_key}: loaded repo module(s) outside the closure: {bad[:5]}")
```

The toy corpus root: `SILLY_KICKS_DATA_ROOT` is set by the worker from `ctx.data_root` (register it `neutral`);
a toy root under `tests/` is then a dataset root.

**`HF_HUB_OFFLINE=1` comes from the process environment** (r1 B CCTP-PLAN-08): `huggingface_hub` reads it at import.
`executor._dispatch` and `executor._subprocess` start every worker and reduce process with `{**deterministic_env(),
"HF_HUB_OFFLINE": "1"}`; spawned inner-pool children inherit it. `install_stage_checks` refuses a process without it.
**Hub prefetch:** `_stages.prefetch_hub_files(stages, *, fetch=_hf_hub_download)` downloads every declared
`(repo, revision, path)` into the HF cache; `_stages.main` gains `--prefetch-hub` (fetch the listed `--stages`'
`hub_files`, then exit), and `launch.job_command` runs it, with network, BEFORE the run line. The workers then load
from the cache offline. In `executor.worker_main` and `reduce_main`: build the context with
`stage_context_from_json`, call `install_stage_checks(f"{ctx.pipeline.name}/{st.name}", root=REPO_ROOT)` when
`ctx.prov.get("tree_state") == "clean"` OR `SILLY_KICKS_IN_HF_JOB` is set OR the CI toy flag
`SILLY_KICKS_STAGE_CHECKS=1` is set (register it `neutral`; CI's toy runs set it in `conftest.py`); call
`check_modules` after every unit; the `InnerPool` passes `initializer=_inner_init_checked` that calls
`install_stage_checks` with the parent's stage key and then the existing `_inner_init`.

- [ ] **Step 5: Implement `stage_context_from_json`**

```python
@dataclasses.dataclass(frozen=True)
class PipelineView:
    """What a stage process needs of a pipeline, without importing the trainer (S5.9): the name (store paths),
    the module (recorded only), the params (types restored) and ONE stage."""

    name: str
    module: str
    params: tuple[Param, ...]
    stages: tuple[AnyStage, ...]
    output: str | None = None

    def stage(self, name: str) -> AnyStage:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(name)


def stage_context_from_json(text: str, stage: str) -> tuple[RunContext, AnyStage]:
    d = json.loads(text)
    mod = importlib.import_module(d["stage_modules"][stage])
    st = mod.STAGES[stage] if hasattr(mod, "STAGES") else getattr(mod, "PIPELINE").stage(stage)
    params = dict(d["params"])
    for name, tag in d["param_types"].items():
        if params.get(name) is not None and tag == "path":
            params[name] = pathlib.Path(params[name])
    view = PipelineView(d["pipeline"], d["module"], (), (st,))
    data_root = None if d["data_root"] is None else pathlib.Path(d["data_root"])
    ctx = RunContext(view, params, d["prov"], LocalStore(pathlib.Path(d["store_root"])), data_root,
                     d["token_inputs"], d["tokens"])
    return ctx, st
```

`context_to_json` adds `"pipeline": ctx.pipeline.name`, `"param_types": {p.name: "path" if p.type is pathlib.Path
else "plain" for p in ctx.pipeline.params}`, and `"stage_modules": {st.name: st.work.__module__ for st in
ctx.pipeline.stages}`. A fit-side stage's module IS the trainer, so its worker imports the trainer — correct,
because its closure contains the trainer. `RunContext.pipeline` is typed `Pipeline | PipelineView`.

- [ ] **Step 5b: The R6 measurement.** Run the win_probability toy `--run-all` twice, with and without
  `SILLY_KICKS_STAGE_CHECKS=1`, and compare the unit walls (the median of the `.ledger.json` `wall_s`). Rule: if the
  hook adds more than 2 %, filter by path prefix first in `_audit` (return before `_rel` for paths outside the repo
  root) and measure again. Record both numbers in the task note.
- [ ] **Step 6:** Run the three test files plus `test_executor.py`, `test_inner_pool.py`, `test_stage_contract.py`
  (background). Expected: PASS. Then run each pipeline's toy `--run-all` with `SILLY_KICKS_STAGE_CHECKS=1` (one
  background job per module; `scripts/_toy_corpora.TOY_PARAMS`). Every violation found = a missing `data_files=`
  declaration or a stage-module leak: fix it at its source and re-run `closures record`.

### Task 15: Content-addressed run keys, replica/canary stores, the controller write allowlist (S5.9 "Stores", S5.13)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`run_key`, `instances`), `scripts/hf_jobs/store.py`. Tests:
`test_campaign.py`, `tests/scripts/hf_jobs/test_bucket_store.py`.

**Interfaces:**
- Produces: `run_key(c, commit, job, tier, *, replica=None, canary=False) -> str` = `{job}-{tier}` (main),
  `{campaign}-{commit7}-{job}-{tier}-r{N}` (replica), `{campaign}-{commit7}-{job}-{tier}-canary` (canary);
  `Instance.role: str` (`main` | `replica` | `canary`);
  `store.CONTROLLER_PREFIXES = ("_campaign/", "_partitions/", "_actions/", "_operator/", "_attest/")`;
  `BucketStore.quarantine_generation(pipeline, stage, token, record: Mapping) -> list[str]`.

- [ ] **Step 1: Failing tests**

```python
def test_main_stores_are_content_addressed_and_replica_and_canary_stay_fresh():
    c = _campaign()
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "ghost_gk-owner"
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=1) == "cycle1-abcdef1-ghost_gk-owner-r1"
    assert C.run_key(c, SHA, "ghost_gk", "owner", canary=True) == "cycle1-abcdef1-ghost_gk-owner-canary"
```

```python
# tests/scripts/hf_jobs/test_bucket_store.py — add
def test_the_controller_handle_refuses_unit_results_and_accepts_its_prefixes():
    import pytest

    from scripts.hf_jobs.store import BucketStore

    calls = []
    api = types.SimpleNamespace(batch_bucket_files=lambda b, add=(), delete=(): calls.append((add, delete)))
    st = BucketStore("ns/b", "rk", api=api)
    for rel in ("ghost_gk/extract/tok/0001.parquet", "ghost_gk/extract/tok/_output.parquet", "_ledger/x.json"):
        with pytest.raises(PermissionError):
            st.upload_bytes(rel, b"x")
    for rel in ("_actions/p/s/t/a.json", "_operator/p/s/t/u.json", "_attest/s/t/c.json"):
        st.upload_bytes(rel, b"x")
    assert len(calls) == 3


class _MutApi(_Api):
    """A dict-backed bucket with recursive listing and `batch_bucket_files(add=, copy=, delete=)`."""

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        pre = prefix or ""
        hits = [p for p in self.files if p.startswith(pre)]
        return [_File(p) for p in hits if recursive or "/" not in p[len(pre):]]

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
        for data, dest in add or ():
            self.files[dest] = data
        for src, dest in copy or ():
            self.files[dest] = self.files[src]
        for path in delete or ():
            self.files.pop(path, None)


def test_quarantine_generation_moves_the_generation_and_its_attestations():
    files = {"rk/p/s/t/0001.parquet": b"a", "rk/p/s/t/0001.sha256": b"h", "rk/p/s/t/_output.json": b"o",
             "rk/_attest/s/t/c1.json": b"{}", "rk/p/s/other/0001.parquet": b"keep"}
    api = _MutApi(dict(files))
    moved = BucketStore("b/x", "rk", api=api).quarantine_generation("p", "s", "t", {"reason": "canary mismatch"})
    assert sorted(moved) == sorted(k for k in files if "/t/" in k)
    assert not any(k.startswith(("rk/p/s/t/", "rk/_attest/s/t/")) for k in api.files)
    q = [k for k in api.files if k.startswith("rk/_quarantine/s/t/")]
    assert len(q) == 5 and any(k.endswith("record.json") for k in q)  # 4 moved files + the record
    assert api.files["rk/p/s/other/0001.parquet"] == b"keep"
```

`batch_bucket_files` in `huggingface_hub` 2.2 takes `add=`, `copy=` and `delete=` (the existing `_UpApi` fake mirrors
that signature), so the move is server-side: one call with `copy=` + the record in `add=`, then one call with
`delete=`.

```python
# tests/scripts/hf_jobs/test_bucket_store.py — add (R4, owner ruling 9)
def test_store_size_sums_every_file_under_the_run_key():
    class _Sized(_MutApi):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
            return [types.SimpleNamespace(type="file", path=p, size=len(v)) for p, v in self.files.items()
                    if p.startswith(prefix or "")]

    api = _Sized({"rk/a": b"12345", "rk/b/c": b"123", "other/x": b"9" * 50})
    assert BucketStore("b/x", "rk", api=api).size_bytes() == 8
```

`BucketStore.size_bytes() -> int` (one recursive listing, the sum of `size`), and `campaign status` prints it per
instance store (`cmd_status` row key `store_bytes`).

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.** `upload_bytes` checks `rel.startswith(CONTROLLER_PREFIXES)`. `quarantine_generation`
  lists `{pipeline}/{stage}/{token}/` and `_attest/{stage}/{token}/` recursively, copies each file server-side to
  `_quarantine/{stage}/{token}/<utc>/<original rel>` and adds `_quarantine/{stage}/{token}/<utc>/record.json` in one
  `batch_bucket_files(copy=..., add=...)` call, then deletes the originals in one `batch_bucket_files(delete=...)`
  call, and returns the moved paths. It is the ONLY method that deletes. Update every
  `run_key` caller (`instances`, `read_plan`, `retrieve`, `cmd_summary`, the tests).
- [ ] **Step 4:** Run. Expected: PASS.

---

## Phase 5 — provenance (ADR-129)

### Task 16: Commit-free stage outputs and stamping at `adopt` (S5.10)

**Files:** Modify `scripts/train_ghost_gk.py` (`fit`/`assemble` writers at `:646-648`, `:749`, `:760`, `:772-777`), any
other writer the sweep finds, `scripts/hf_jobs/adopt.py` (new `stamp_provenance`), `scripts/hf_jobs/identity.py`
(export the writer used for `SHA256SUMS`). Create `tests/scripts/test_stage_outputs_commit_free.py`. Test:
`tests/scripts/hf_jobs/test_adopt.py`.

- [ ] **Step 1: Sweep (S6.7).** Run `git grep -nE "ctx\.prov|training_commit|run_commit|run_tree_(state|dirty)|reproducibility_note|SHA256SUMS" -- scripts`
  and list every writer inside a stage callable, and every reader of `training_commit` (`manifest.py:168`,
  `:282-295`, the cards, `identity.py`). Put the list in the task note. Any reader that treats `training_commit` as
  "the adopting HEAD" stops the task for an owner ruling (R5).

- [ ] **Step 2: Failing tests**

```python
# tests/scripts/test_stage_outputs_commit_free.py
"""S5.10: no stage module reads ctx.prov; no toy output holds a run-dependent provenance key or SHA256SUMS."""

from __future__ import annotations

import ast
import importlib
import json

import pytest

from scripts.hf_jobs.identity import PROVENANCE_KEYS

RUN_DEPENDENT = tuple(k for k in PROVENANCE_KEYS if k != "generation")


def _stage_modules():
    from scripts.hf_jobs.closures import _population, stage_module

    for mod in _population():
        for st in importlib.import_module(mod).PIPELINE.stages:
            yield stage_module(st)


@pytest.mark.parametrize("module", sorted(set(_stage_modules())))
def test_no_stage_module_reads_ctx_prov(module):
    src = importlib.import_module(module).__file__
    tree = ast.parse(open(src, encoding="utf-8").read())
    hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "prov"]
    assert not hits, f"{module}: ctx.prov read at lines {hits}"


def test_the_ghost_gk_toy_output_is_commit_free(tmp_path):
    from tests.scripts._stage_harness import run_toy  # read the harness; add `run_toy(module, tmp)` if absent

    out = run_toy("scripts.train_ghost_gk", tmp_path)  # -> {filename: bytes} of the output FileSet
    assert "SHA256SUMS" not in out
    for name in ("metadata.json", "metrics.json"):
        doc = json.loads(out[name])
        assert not (set(RUN_DEPENDENT) & set(doc)), name
```

```python
# tests/scripts/hf_jobs/test_adopt.py — add
def test_stamping_writes_every_key_from_its_source_and_regenerates_the_sums(tmp_path):
    from scripts.hf_jobs import adopt, identity

    d = tmp_path / "art"
    d.mkdir()
    (d / "metadata.json").write_text('{"variant":"default"}')
    (d / "metrics.json").write_text('{"cv":1}')
    att = {"commit": "a" * 40, "main": {"env": {"platform": "Linux-x"}, "tree_state": "clean"}}
    adopt.stamp_provenance(d, attestation=att, params={"training_platform": None}, generation="tok")
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["training_commit"] == "a" * 40 and meta["run_tree_dirty"] is False
    assert meta["generation"] == "tok"
    assert identity.verify_sums(d)  # SHA256SUMS lists the final bytes
```

- [ ] **Step 3:** Run both. Expected: FAIL.
- [ ] **Step 4: Implement.** Remove the provenance writes and the `SHA256SUMS` write from the ghost_gk stage
  callables (and any other the sweep found). `adopt.stamp_provenance(directory, *, attestation, params, generation)`
  writes the S5.10 table's keys (`training_commit`/`run_commit` = the attestation commit; `run_tree_state` /
  `run_tree_dirty` from the attesting job's provenance; `training_platform =
  resolve_training_platform(params["training_platform"], attestation["main"])` — import it from the trainer's stage
  module or move `resolve_training_platform` into `scripts/_provenance.py` and re-export it; `reproducibility_note`
  from the same `reproducibility(...)` the trainer used, called with the stamped commit; `generation` only where the
  output already carries it), then rewrites `SHA256SUMS` with the trainer's writer (move it to `identity.write_sums`
  and use it from both sides). `identity.verify_sums(d) -> bool` checks it.
- [ ] **Step 5:** Run both, plus `test_publish_verify.py`, `test_identity.py`, the ghost_gk tests. Expected: PASS.

### Task 17: Attestation, the attestation-aware plan, the canary and quarantine (S5.10)

**Files:** Modify `scripts/hf_jobs/adopt.py` (`verify` writes attestations; `adopt` checks them), `scripts/hf_jobs/campaign.py`
(the plan at HEAD, canary instances, `--with-replica`). Create `tests/scripts/hf_jobs/test_reuse_plan.py`. Test:
`test_adopt.py`, `test_campaign.py`.

**Interfaces:**
- Produces: `attest_rel(stage, token, commit) -> str` = `_attest/{stage}/{token}/{commit}.json`;
  `attestations(store, stage, token) -> list[dict]`; `is_ancestor(commit, head) -> bool` (git
  `merge-base --is-ancestor`, controller-side only); `canary_key(units, token, head) -> str` =
  `min(units, key=lambda k: sha256(token + head + k))`;
  `plan_for_role(plan, role, attested: set[tuple[stage, token]], head) -> tuple[dict, dict[str, list[str]]]` (the
  plan view for `main`/`replica`/`canary` and, for `canary`, the unit keys per stage);
  `canary_verdict(*, main_bytes, canary_bytes, quarantine: Callable[[], object]) -> bool`;
  `CanaryReadThroughStore(local: Store, main: Store, allowed: Callable[[str], bool])` (reads a generation from
  `main` when `allowed(generation)` and the local store lacks it; writes only to `local`);
  `adoption_refusals(*, head_tokens, adopted, attestations, quarantined, canary_ok, is_ancestor) -> list[str]`.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/hf_jobs/test_reuse_plan.py
"""S5.10 on toys: an attested token is reused (no replica, one canary unit); an unattested token runs main + replica;
the canary reads upstream from the main store; a mismatch quarantines and re-plans."""

def test_canary_key_is_deterministic_per_head_and_moves_with_head():
    from scripts.hf_jobs import campaign as C

    units = [f"u{i}" for i in range(50)]
    a = C.canary_key(units, "tok", "a" * 40)
    assert a == C.canary_key(units, "tok", "a" * 40) and a in units
    assert any(C.canary_key(units, "tok", h * 40) != a for h in "bcdef")


HEAD = "f" * 40


def _two_stage(reduced: bool):
    from tests.scripts.hf_jobs.test_campaign import _plan, _st

    return _plan([_st("a", n=50, reduced=reduced, done=50 if reduced else 0),
                  _st("b", n=1, needs=("a",), reduced=reduced, done=1 if reduced else 0, ready=reduced)])


def test_an_attested_stage_skips_the_replica_and_runs_one_canary_unit():
    from scripts.hf_jobs import campaign as C

    att = {("a", "t_a"), ("b", "t_b")}
    replica, _ = C.plan_for_role(_two_stage(False), "replica", att, HEAD)
    assert all(s["reduced"] for s in replica["stages"])  # nothing to replicate
    _, canary = C.plan_for_role(_two_stage(False), "canary", att, HEAD)
    units_a = [f"u{i}" for i in range(50)]
    assert canary == {"a": [C.canary_key(units_a, "t_a", HEAD)], "b": ["u0"]}  # a single-unit stage: the whole stage


def test_an_unattested_stage_runs_in_main_and_replica_and_gets_no_canary():
    from scripts.hf_jobs import campaign as C

    replica, _ = C.plan_for_role(_two_stage(False), "replica", set(), HEAD)
    assert not any(s["reduced"] for s in replica["stages"])
    _, canary = C.plan_for_role(_two_stage(False), "canary", set(), HEAD)
    assert canary == {}


def test_the_canary_runs_over_the_real_census_graph(tmp_path):
    import importlib

    from scripts._stages import ItemStage, LocalStore, make_context, read_result, run_unit, sorted_units
    from scripts._toy_corpora import TOY_PARAMS, toy_session
    from scripts.hf_jobs.campaign import CanaryReadThroughStore, canary_key
    from tests.scripts._stage_harness import PROV, params_from_args, run_in_process

    mod = "scripts.build_territory_ranking_census"
    pipe = importlib.import_module(mod).PIPELINE
    with toy_session(mod, str(tmp_path / "d")) as (data, extra_env):
        import os

        os.environ.update(extra_env)
        main = run_in_process(pipe, TOY_PARAMS[mod], store=tmp_path / "main", data_root=data)
        canary_store = CanaryReadThroughStore(LocalStore(tmp_path / "canary"), main.store, allowed=lambda gen: True)
        ctx = make_context(pipe, params_from_args(pipe, TOY_PARAMS[mod]), PROV, store=canary_store, data_root=data)
        for name in ("fit_prepass", "prefit", "score"):
            st = pipe.stage(name)
            units = sorted_units(ctx, st)  # reads `refs` through the main store
            key = canary_key(units, ctx.tokens[name], HEAD)
            refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
            run_unit(ctx, st, key, worker="canary", ref=refs.get(key))
            gen = ctx.generation(name)
            assert read_result(LocalStore(tmp_path / "canary"), f"{gen}/{key}") is not None
            sfx = next(x for x in (".parquet", ".json", ".fileset.json") if (tmp_path / "main" / gen / f"{key}{x}").is_file())
            assert (tmp_path / "canary" / gen / f"{key}{sfx}").read_bytes() == (tmp_path / "main" / gen / f"{key}{sfx}").read_bytes()


def test_a_canary_mismatch_quarantines_and_the_next_plan_recomputes():
    from scripts.hf_jobs import campaign as C

    calls = []
    quarantined = C.canary_verdict(main_bytes=b"same", canary_bytes=b"diff", quarantine=lambda: calls.append(1))
    assert quarantined is False and calls == [1]
    replica, _ = C.plan_for_role(_two_stage(False), "replica", {("b", "t_b")}, HEAD)  # "a" lost its attestation
    assert [s["reduced"] for s in replica["stages"]] == [False, True]
```

The census test reads `refs` and `prefit`/`fit_prepass` outputs from the main store through the read-through, and
writes only into the canary store; the census toy is open data, so it runs inside `toy_session`. If `run_in_process`
is too slow for the census toy (over 2 minutes), mark the test `@pytest.mark.slow` (ADR-023; the `slow` job runs it).

```python
# tests/scripts/hf_jobs/test_adopt.py — add
def test_adoption_refusals_cover_every_rule_from_both_sides():
    from scripts.hf_jobs import adopt

    anc = {"old", "mid"}.__contains__
    base = dict(head_tokens={"fit": "T"}, adopted={"fit": "T"}, attestations={"fit": ["old", "mid"]},
                quarantined=set(), canary_ok={"fit": True}, is_ancestor=anc)
    assert adopt.adoption_refusals(**base) == []
    assert "token" in adopt.adoption_refusals(**{**base, "adopted": {"fit": "OTHER"}})[0]
    assert "attestation" in adopt.adoption_refusals(**{**base, "attestations": {"fit": []}})[0]
    assert "ancestor" in adopt.adoption_refusals(**{**base, "attestations": {"fit": ["branch"]}})[0]
    assert "quarantine" in adopt.adoption_refusals(**{**base, "quarantined": {"T"}})[0]
    assert "canary" in adopt.adoption_refusals(**{**base, "canary_ok": {"fit": False}})[0]
    assert adopt.oldest_ancestor_attester(["mid", "old"], order=["new", "mid", "old"]) == "old"
```

(`oldest_ancestor_attester(commits, *, order)` takes the attesting commits and HEAD's `git rev-list --topo-order`
order, newest first, and returns the ancestor attester that comes LAST in that order.)

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `verify` (after the existing checks): for each instance and stage, compare the main generation with the replica
    generation of the same token file by file (names and bytes). Excluded from the compare (run history, never
    results; r1 B CCTP-PLAN-14): `*.ledger.json`, `*.failed.json`, `_profile.json`, the `.corrupt/` tree and `_ledger/`.
    Every result, sidecar, FileSet member, `*.excluded.json`, `*.counters.json`, `_summary.json`, `_inputs.json` and
    the output IS compared. Equal ->
    `upload_bytes(attest_rel(...), {"commit", "main": <main job env_provenance>, "replica": <replica job
    env_provenance>, "job_ids", "time"})`. The jobs' recorded provenance comes from a stage's `_inputs.json` and the
    job phase records (add `env` to the job phase record in `phases.publish_first_unit`: `env_provenance()`).
  - The plan at HEAD (`step`): for each main-instance stage token, `atts = attestations(...)`;
    `attested = any(is_ancestor(a["commit"], head) for a in atts) and not quarantined`. Attested: the main run takes
    no action for it (a done generation already exists; if not, it runs normally), the replica instance SKIPS the
    stage (mark it reduced in the replica's plan view), and the canary instance gets one action:
    `--shard-keys [canary_key(units, token, head)]` for a unit stage, or the whole stage for a single-unit stage.
  - The canary instance's store: the whole bucket is already mounted at `/scratch` (`launch.py:237`), so the main
    store is `/scratch/{job}-{tier}` and the canary store is `/scratch/{campaign}-{commit7}-{job}-{tier}-canary`; no
    second mount (r1 B CCTP-PLAN-13). `CanaryReadThroughStore` serves `read_bytes`/`names` for a generation listed in
    the ALLOW FILE from the main store; every write goes to the canary store. The controller writes the allow file
    with the canary action, `_campaign/{campaign}/{commit7}/canary/{action_id}.allow.json` = the upstream generations
    whose tokens are attested on HEAD or an ancestor and not quarantined; the job reads it at
    `/scratch/{canary run key}/...`. Wire it with `--read-through /scratch/{job}-{tier} --read-through-allow <file>` in
    `_stages.main` (accepted only together with `--shard-keys` or a single-unit stage). `allowed(gen)` =
    `gen in allow_file`.
  - The canary compare (in `verify`): the canary unit's result bytes vs the main store's bytes for that unit (or the
    stage output for a single-unit stage); a mismatch calls `quarantine_generation` (Task 15) and adds an
    `unsound_reuse` finding to `verify.json` (the summary reports it to the owner).
  - `adopt`: replace the `HEAD == campaign commit` check (`adopt.py:570-571`) with: clean tree; recompute tokens at
    HEAD (`read_plan`); every adopted generation's token == HEAD's token; an attestation on HEAD or an ancestor
    (`is_ancestor`); its canary passed or HEAD attested it this campaign; not quarantined. Then `stamp_provenance`
    with the OLDEST ancestor attester (order the attesting commits by `git rev-list --topo-order HEAD`, last wins).
    The manifest adds `adopted_at_commit` and `attested_by` (`manifest.Artifact` fields; `manifest.py:168`'s check
    stays on `training_commit`).
- [ ] **Step 4:** Run the two files plus `test_campaign.py`. Expected: PASS.

---

## Phase 6 — the job image

### Task 18: Dockerfile, image lock, in-job check, `check_image`, the startup and registry probes (S5.12, E-1, E-6)

**Files:** Create `scripts/hf_jobs/image/Dockerfile`, `scripts/hf_jobs/image.lock.json`, `scripts/hf_jobs/image.py`,
`.github/workflows/job-image.yml`, `tests/scripts/hf_jobs/test_image_lock.py`. Modify `scripts/hf_jobs/launch.py`
(`IMAGE` -> `image_ref()`, `job_command`), `scripts/hf_jobs/preflight.py` (`check_lock`, new `check_image`),
`scripts/hf_jobs/__main__.py` (`image`).

**Interfaces:**
- Produces: `image.EXTRAS` (the union of the registry's `extras`, computed); `image.derived_tag(root) -> str`;
  `image.load_lock() -> dict` (`tag`, `inputs_sha256`, `registry_ref`, `digest`, `built_at`, `startup_s`,
  `startup_samples`, `startup_jobs`); `image.image_ref() -> str` (`<registry_ref>@<digest>`);
  `image.CHECK_COMMAND: str`; CLI `python -m scripts.hf_jobs image {tag,probe-registry,probe-startup}`.

- [ ] **Step 1: Failing tests**

```python
# tests/scripts/hf_jobs/test_image_lock.py
"""S5.12: the image lock matches its inputs; the job command checks, never installs."""

from __future__ import annotations

from scripts.hf_jobs import image, launch


def test_the_derived_tag_equals_the_locked_tag():
    assert image.derived_tag() == image.load_lock()["tag"]


def test_the_job_command_checks_the_environment_once_and_installs_nothing():
    from tests.scripts.hf_jobs.test_launch_typed import _spec

    cmd = launch.job_command(_spec())[2]
    assert cmd.count(image.CHECK_COMMAND) == 1
    assert "pip install" not in cmd
    assert all("--check" in line for line in cmd.splitlines() if "uv sync" in line)


def test_the_job_command_marks_the_environment_check():
    from tests.scripts.hf_jobs.test_launch_typed import _spec

    assert "phases mark after_env_check" in launch.job_command(_spec())[2]


def test_one_image_serves_a_spec_with_the_statsbombpy_extra_and_one_without():
    import dataclasses

    from tests.scripts.hf_jobs.test_launch_typed import _spec

    for extras in ((), ("statsbombpy",)):
        cmd = launch.job_command(dataclasses.replace(_spec(), extras=extras))[2]
        assert cmd.count(image.CHECK_COMMAND) == 1  # the same check, holding every registry extra


def test_the_image_holds_every_registry_extra():
    from scripts.hf_jobs.registry import REGISTRY  # dict[str, TrainerSpec], registry.py:102 (r2 A R2-01)

    assert {e for spec in REGISTRY.values() for e in spec.extras} <= set(image.EXTRAS)


def test_a_lock_without_startup_s_refuses_preflight(monkeypatch):
    from scripts.hf_jobs import preflight

    monkeypatch.setattr(image, "load_lock", lambda: {**image.load_lock(), "startup_s": None})
    assert not preflight.check_image(resolve=lambda ref: True).ok
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement.** The Dockerfile:

```dockerfile
# ADR-128 Cycle 1b (S5.12): the locked third-party environment only. No repo code, no data.
FROM python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0
ARG UV_REQUIREMENT
ARG EXTRAS
RUN printf '%s\n' "${UV_REQUIREMENT}" > /tmp/uv-req.txt \
 && /usr/local/bin/python3.12 -m pip install -q --disable-pip-version-check --require-hashes -r /tmp/uv-req.txt
WORKDIR /env
COPY pyproject.toml uv.lock /env/
RUN uv sync --frozen --no-install-project --python /usr/local/bin/python3.12 --extra train ${EXTRAS} --group hfjobs
ENV VIRTUAL_ENV=/env/.venv PATH=/env/.venv/bin:$PATH
```

(The project's `pyproject.toml` is needed for `uv sync` to read the extras and groups; `--no-install-project`
keeps repo code out.) `image.derived_tag()` = `"env-" + sha256(uv.lock bytes + Dockerfile bytes + base digest +
",".join(EXTRAS) + UV_REQUIREMENT)[:12]`. `CHECK_COMMAND` = `uv sync --frozen <the Task 0.4 check flag>
--no-install-project --python /env/.venv/bin/python --extra train <extras> --group hfjobs`, run from `/work` against
the fetched `uv.lock` with `UV_PROJECT_ENVIRONMENT=/env/.venv`.

`job_command` becomes: the attempt-id and `script_start` lines; the git lines; `phases mark after_fetch`;
`CHECK_COMMAND`; `phases mark after_env_check`; the dataset fetch lines each followed by `phases mark
after_dataset_fetch`; the run line with `/env/.venv/bin/python`. Remove the `pip install` and the installing
`uv sync`. `launch.launch` uses `image.image_ref()`; `IMAGE_ENV` gets `image.image_ref()`; `_stages._image_identity()`'s
fallback becomes `image.image_ref()` (Task 13); the old `IMAGE` constant
stays as `BASE_IMAGE` (the Dockerfile's `FROM`, checked equal by a test).

The workflow (E-1):

```yaml
# .github/workflows/job-image.yml — ADR-128 Cycle 1b (S5.12, owner ruling 11a): build, push and attest the locked job
# environment. Triggered by a `job-image/env-<12 hex>` tag (ruleset: owner only); publishes only after the owner
# approves the `job-image` environment (required reviewer), the same pattern as publish.yml's `pypi` environment.
# Every action is pinned by full commit SHA with its version in a comment, as publish.yml does (`publish.yml:17`).
name: job-image
on:
  push:
    tags: ["job-image/env-*"]
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-24.04
    environment: job-image
    permissions:
      contents: read
      packages: write
      id-token: write
      attestations: write
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1 (the publish.yml pin)
      - id: meta
        name: Derive the image tag and the build arguments; refuse a tag that does not match the inputs
        run: |
          tag="${GITHUB_REF_NAME#job-image/}"
          want="$(python3 -m scripts.hf_jobs image tag)"
          test "${tag}" = "${want}" || { echo "tag ${tag} != derived ${want}"; exit 1; }
          echo "tag=${tag}" >> "$GITHUB_OUTPUT"
          echo "image=ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/silly-kicks-job" >> "$GITHUB_OUTPUT"
          python3 -m scripts.hf_jobs image build-args >> "$GITHUB_ENV"   # writes UV_REQ=... and EXTRAS=...
      - uses: docker/login-action@dbcb813823bdd20940b903addbd779551569679f  # v4.6.0
        with: {registry: ghcr.io, username: "${{ github.actor }}", password: "${{ secrets.GITHUB_TOKEN }}"}
      - name: Refuse an existing image tag (after login, so a private package is visible)
        run: |
          if docker manifest inspect "${{ steps.meta.outputs.image }}:${{ steps.meta.outputs.tag }}" >/dev/null 2>&1; then
            echo "image tag ${{ steps.meta.outputs.tag }} exists; never rebuilt"; exit 1; fi
      - id: push
        uses: docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc  # v7.4.0
        with:
          context: .
          file: scripts/hf_jobs/image/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.image }}:${{ steps.meta.outputs.tag }}
          build-args: |
            UV_REQUIREMENT=${{ env.UV_REQ }}
            EXTRAS=${{ env.EXTRAS }}
      - uses: actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8  # v4.2.2
        with:
          subject-name: ${{ steps.meta.outputs.image }}
          subject-digest: ${{ steps.push.outputs.digest }}
          push-to-registry: true
      - name: Record the digest for the image lock
        run: echo "digest=${{ steps.push.outputs.digest }}" >> "$GITHUB_STEP_SUMMARY"
```

The pins are the latest releases on 2026-10-21 (`gh api repos/<action>/releases/latest`, then the tag's commit SHA),
in `publish.yml`'s style (r2 B R2-08). At implementation, confirm each action's inputs against its README at that
version (`subject-name`, `subject-digest`, `push-to-registry` for attest-build-provenance; `context`, `file`, `push`,
`tags`, `build-args` for build-push-action); a renamed input is fixed in the YAML and its test, not by a downgrade.
`image build-args` prints `UV_REQ=<UV_REQUIREMENT>` and `EXTRAS=<"--extra x" per registry extra>`, one per line.
The `test_image_lock.py` YAML case also asserts: no `uses:` without a 40-hex SHA; the login step comes before the
`manifest inspect` step; `tags:` uses `steps.meta.outputs.tag`, never `github.ref_name`.

A test in `test_image_lock.py` parses the workflow YAML and asserts: the only trigger is
`push.tags: ["job-image/env-*"]`; the job runs in `environment: job-image`; `attest-build-provenance` follows the
push; every `uses:` is pinned by a full commit SHA.

Pin every action by full commit SHA, as `ci.yml` and `publish.yml` do (read them; copy their pin style), and set
`UV_REQ`/`EXTRAS` from `python3 -m scripts.hf_jobs image build-args`. The `image tag` CLI must run with the system
`python3` and stdlib only (it hashes files; it imports no third-party package).

`preflight.check_lock` keeps only the `uv lock --check` half. `check_image(resolve=_registry_resolves,
attested=_attestation_verifies)`: the lock's tag equals `derived_tag()`; `startup_s` is a positive number;
`resolve(image_ref())` is True (an anonymous GHCR manifest HEAD:
`https://ghcr.io/token?scope=repository:<owner>/silly-kicks-job:pull` then
`HEAD https://ghcr.io/v2/<owner>/silly-kicks-job/manifests/<digest>`); and `attested(image_ref())` is True
(`gh attestation verify oci://<image_ref> --owner <owner>` exits 0; preflight already shells out to `gh`,
`preflight.py:43-45`). Add it to `cmd_preflight`'s list, and add a `test_image_lock.py` case where `attested`
returns False and the check fails.

`image probe-registry` launches one `cpu-upgrade` job per candidate reference (GHCR; then Docker Hub and an HF Space
only if GHCR does not pull) running `python -c "print(1)"`; prints which pulled. `image probe-startup` launches the 5
startup jobs of S5.12 (the win_probability `refs,extract --profile 1` job command, stores
`image-probe-<tag>-<i>-win_probability-public`), waits, reads their job phase records, and writes `startup_s`,
`startup_samples`, `startup_jobs` into `image.lock.json`. It refuses a dirty tree except for `image.lock.json`
itself.

- [ ] **Step 4:** Run the test file plus `test_launch_typed.py`, `test_preflight.py`. Expected: PASS (the lock holds
  `tag` and `inputs_sha256` now; `digest`, `registry_ref` and `startup_s` are filled in Task 21 — until then mark
  `test_a_lock_without_startup_s_refuses_preflight` as the only test that reads them, and keep the lock's
  `startup_s: null`).

### Task 19: The importance split decision (S5.6, owner ruling 6)

**Files:** Read `scripts/train_ghost_gk.py` `_importance_work` and its helpers. Possibly modify them; create
`tests/scripts/test_train_ghost_gk_importance_split.py` if split.

- [ ] **Step 1: Decide by reading.** The stage may split only if, per permutation group, the result depends on
  nothing but that group's inputs: no shared RNG stream across groups (each group's seed must be derivable from the
  group key alone), no evaluation-order dependence, no whole-set quantity (a normalization over all groups is
  allowed only in the reduce, over sorted keys). Record the evidence (file:line) for each of the three conditions.
- [ ] **Step 2 (split):** write the failing byte-identity test: the toy's one-unit output at `74d57ba`'s code path
  (`git show 74d57ba:scripts/train_ghost_gk.py` is the reference; run it from an exported copy, never by checking
  out) equals the split output, byte for byte. Then split: `units` = the sorted group keys, `work` = one group,
  `reduce` = the sorted-key combine. Run the test. Add the real-HF parity run to Task 22 (S7.1 item 4).
- [ ] **Step 2 (no split):** write the reason (the failing condition with file:line) into ADR-129 (Task 20) and the
  task note. Change nothing.

---

## Phase 7 — docs, gates and the code commit

### Task 20: Bars, predictions and summary records; docs (S7.2, S9)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`cmd_summary` computes the bars; `step` appends predictions),
`scripts/hf_jobs/ledger.py` (`summarize` takes `bars`). Create `docs/superpowers/adrs/ADR-129-content-addressed-reuse-attestation.md`.
Modify ADR-128 (amendment section: P9, P18, Cycle 1b), ADR-037 (amendment note), `docs/howto/compute-pipelines.md`,
the `AGENTS.md` compute bullet, `CHANGELOG.md` (`[Unreleased]`), `docs/c4/architecture.dsl` + regenerated
`architecture.html`. Test: `test_campaign.py`.

- [ ] **Step 1: Failing test**

```python
def _bar_inputs(scale: float = 1.0):
    import datetime

    iso = lambda t: datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()  # noqa: E731
    jobs = [  # one unit job (action A, 1000 s) and one separate reduce job (action R, 100 s), main role
        {"id": "j1", "labels": {"action": "A", "run_key": "g-o"}, "role": "main", "created_at": iso(0),
         "started_at": iso(10), "finished_at": iso(1010 * scale), "wall_s": 1000.0 * scale},
        {"id": "j2", "labels": {"action": "R", "run_key": "g-o"}, "role": "main", "created_at": iso(1010 * scale),
         "started_at": iso(1020 * scale), "finished_at": iso(1120 * scale), "wall_s": 100.0},
    ]
    preds = [{"kind": "campaign", "makespan_s": 1120.0}, {"action_id": "A", "predicted_wall_s": 1000.0},
             {"action_id": "R", "predicted_wall_s": 100.0, "predicted_reduce_wall_s": 90.0}]
    stages = [{"instance": "g-o/default", "stage": "x", "needs": [], "per_unit": {"u0": 1000.0},
               "unit_cost": {"p50_s": 1000.0, "p95_s": 1000.0, "p99_s": 1000.0, "peak_rss_bytes": 1, "n_samples": 1}}]
    reduces = [{"stage": "x", "token": "t", "wall_s": 90.0, "n_results": 2000, "in_job": False, "predicted_wall_s": 90.0}]
    phases = [{"phases": {"script_start": 0.0, "first_unit": 60.0}}]
    return dict(jobs=jobs, predictions=preds, stages=stages, reduce_records=reduces, job_phases=phases,
                cycle1_main_job_hours=1100.0 / 3600, cap=20, declared_startup_s=60.0, comparisons=[True, True])


def test_the_summary_computes_each_bar_from_its_records():
    bars = C.compute_bars(**_bar_inputs())
    assert set(bars) == {"schedule_efficiency", "planner_calibration", "reduce", "startup", "cost", "determinism"}
    assert all(b["pass"] for b in bars.values()), bars
    assert bars["startup"]["value"] == pytest.approx(1.0)


def test_each_bar_fails_from_the_other_side():
    slow = C.compute_bars(**_bar_inputs(scale=2.0))  # the unit job ran twice as long as planned
    assert not slow["planner_calibration"]["pass"] and not slow["cost"]["pass"]
    bad = _bar_inputs()
    bad["comparisons"] = [True, False]
    assert not C.compute_bars(**bad)["determinism"]["pass"]
    bad = _bar_inputs()
    bad["job_phases"] = [{"phases": {"script_start": 0.0, "install": 5.0, "first_unit": 200.0}}]
    assert not C.compute_bars(**bad)["startup"]["pass"]
```

`compute_bars(*, jobs, predictions, stages, reduce_records, job_phases, cycle1_main_job_hours, cap,
declared_startup_s, comparisons) -> dict[str, dict]`: each value is `{"value", "threshold", "pass"}` (bar 2 and bar
3 also carry `"per_action"` / `"per_reduce"` rows). `cmd_summary` assembles the inputs from the ledger, the HF job
records, `predictions.jsonl`, the job phase records and the verify report.

- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3: Implement `compute_bars`** with the S7.2 formulas exactly: bar 1 makespan / LB with LB =
  max(critical path of ideal walls via `planner.ideal_stage_wall` + the shortest successful reduce record per token
  + a startup only for a separate reduce job, total job-hours / cap); bar 2 per campaign and per action over 10
  minutes (0.75–1.25); bar 3 every reduce over 1000 results vs its `plan_reduce` prediction (≤ 1.5); bar 4 no
  `install` phase and median startup (first unit − script start − dataset fetch) / `image.lock.json startup_s`
  (≤ 1.5); bar 5 main-run job-hours (no replica, canary, profile jobs; Cycle 1 excludes `6ad89c3b` and `6ad8ccde`;
  reused units excluded by key) / Cycle 1 (≤ 1.1); bar 6 all replica and canary comparisons equal. The thresholds
  are module constants named `BAR_THRESHOLDS`, declared here before the run (INV-9). `step` appends to
  `_campaign/{campaign}/{commit7}/predictions.jsonl` (download, append, upload: the controller is its one writer) —
  the first record `{"kind": "campaign", "makespan_s": <remaining_critical_path max over instances>}`, then one per
  action `{"action_id", "stage", "predicted_wall_s", "predicted_reduce_wall_s"}`.
- [ ] **Step 4: Docs.** ADR-129 from `docs/superpowers/adrs/ADR-TEMPLATE.md`: context (S2), decision (S5.9–S5.10, S5.13),
  the attestation model, the canary, INV-10/INV-11, the importance-split outcome (Task 19), consequences, the E-1/E-2
  rulings. ADR-128 amendment: P9 (closure identity replaces the commit and `cross_commit_version`), P18 (attestation
  replaces the same-commit replica), the controller changes. ADR-037: "adoption by token equality at a clean HEAD
  (ADR-129) satisfies the clean-tree split". Howto: the closure lock workflow (`closures record`), stage modules,
  `data_files=`/`hub_files=`, the env registry, `--clear-halt`, `--with-replica`, the image tag push. AGENTS.md: edit
  the compute bullet (≤ 600 chars, `tests/test_agents_md_budget.py`): add "content-addressed stores; per-stage closure
  identity (`closures.lock.json`); ADR-129 attestation". CHANGELOG `[Unreleased]`: a Cycle 1b entry. C4: run the
  `mad-scientist-skills:c4` regeneration for the new `scripts/hf_jobs` components.
- [ ] **Step 5:** Run the test, `tests/test_agents_md_budget.py`, the docs tests (`git grep -l "CHANGELOG\|ADR-" tests | head`). Expected: PASS.

### Task 21: Local gate, A/B implementation review, the code commit (owner gate G2)

- [ ] **Step 1:** Run ruff (check + format --check) at CI scope and pyright on every changed file. Fix all.
- [ ] **Step 2:** Run the targeted tests of every task above (one background command). Expected: PASS.
- [ ] **Step 3:** Hand off to reviewers A and B (`/review-impl`), tree frozen. Fix every finding; re-review until both
  APPROVE.
- [ ] **Step 4: OWNER GATE.** Show the file list and diff stat; get an explicit yes; commit
  (`feat(hf_jobs): campaign throughput, linear reduce and content-addressed reuse (ADR-129)`, trailer per Global
  Constraints); push (separate yes); open or update the PR (separate yes). Watch CI with Monitor.

### Task 22: HF probes that need the pushed code; the amend (E-1, E-5, S5.12)

- [ ] **Step 1: OWNER SETUP (once, owner acts).** Create the `job-image` environment with the owner as required
  reviewer; create a ruleset that restricts `job-image/*` tag creation to the owner. The session prints the exact
  settings; it changes no repository setting itself.
- [ ] **Step 2: OWNER GATE — image publish.** The owner pushes the tag `job-image/$(python -m scripts.hf_jobs image
  tag)` and approves the `job-image` environment run. Monitor the workflow. Then the owner sets the GHCR package
  public (outward act; owner). Run `gh attestation verify oci://<image ref> --owner <owner>` and record the result.
- [ ] **Step 2b:** `python -m scripts.hf_jobs image probe-registry` (cpu-upgrade, pre-approved). If no registry pulls:
  STOP and report to the owner (S5.12 rule).
- [ ] **Step 3:** FIRST, on the clean tree at the code commit (before any lock edit, so `require_clean_tree` passes):
  `campaign scripts/hf_jobs/campaigns/cycle1.toml history-export --commit 74d57ba23d2a79de7cb18c6d85b6218d2d820a69`
  (E-5, read-only over the Cycle 1 stores) writes `docs/research/hf_campaigns/cycle1.per_unit.json` (r1 B
  CCTP-PLAN-07). Owner ruling 2026-10-21 (the E-5 extension): Cycle 1 committed no summary, so the export ALSO
  writes the stage cost rows (`unit_cost`, `efficiency`, `startup_s`, `history_key` per stage, as `cmd_summary`
  computes them) into the same file, and `committed_history` reads both. The first Cycle 1b campaign then plans from
  measured costs and runs no profile job. The `--commit` flag names the Cycle 1 run commit, because HEAD is the
  Cycle 1b code commit.
- [ ] **Step 4:** Record `registry_ref` and `digest` in `image.lock.json`. Run `image probe-startup` (5 jobs); it writes
  `startup_s` (it tolerates exactly two dirty paths: `image.lock.json` and the new `cycle1.per_unit.json`).
- [ ] **Step 5:** If the R8 probe (Task 0.2) refused 9 labels and Task 4 used the fallback, confirm one launch with the
  fallback labels succeeds (a 1-minute `cpu-upgrade` job).
- [ ] **Step 6: OWNER GATE — amend.** Show the diff (`image.lock.json`, `cycle1.per_unit.json`, the `datasets.toml`
  `[store]` values if Task 0.3 ran after the commit). Amend with an explicit yes; force-push with a separate yes. CI
  green.

---

## Phase 8 — the Cycle 1b campaign and the artifact commit

### Task 23: Run `cycle1b` with replica and canary (S7.1 items 2, 4)

- [ ] **Step 1:** `campaign scripts/hf_jobs/campaigns/cycle1b.toml preflight` (all checks incl. `check_image`,
  `check_concurrency`). Expected: ok.
- [ ] **Step 2:** `campaign ... launch --with-replica 1 --watch` in the background; Monitor the step reports. Do not
  touch the tree while it runs. A halt: diagnose; the fix path is `--clear-halt` with a reason, never a scratchpad
  launcher (INV-7).
- [ ] **Step 3 (if Task 19 split importance):** the importance parity run (S7.1 item 4): the split output vs a
  one-unit output on the real corpus, byte-identical.
- [ ] **Step 4:** `retrieve`, then `verify`. `verify` writes the attestations. `cycle1b` reuses nothing: the Cycle 1
  stores are `cycle1-74d57ba-*`, not `{job}-{tier}`, so every token is new, runs in main and replica, and is
  attested on the `cycle1b` commit. There is therefore no canary compare in this first campaign.
- [ ] **Step 5:** `campaign ... summary` (clean tree): writes `docs/research/hf_campaigns/cycle1b.json` with the six bars.
  Report every bar to the owner with its numbers; a missed bar is a finding (owner ruling 10).

### Task 24: Adopt, sandbox publish, the reuse demonstration, the artifact commit (owner gates)

- [ ] **Step 1: OWNER GATE — adopt** (clean tree; by token + attestation). Then the sandbox publish (each repo an owner
  yes).
- [ ] **Step 2:** Write the artifact commit's files into the tree (adopted weights, `observed.json`, cards,
  `cycle1b.json`, `manifest.toml`, CHANGELOG, TODO.md).
- [ ] **Step 3: OWNER GATE — the artifact commit.** Show the file list; commit with an explicit yes. Do not push yet.
- [ ] **Step 4: The E-2 plan table (S7.1 item 3, owner ruling 11b step 1).** At that HEAD (clean), run `campaign
  scripts/hf_jobs/campaigns/cycle1b.toml plan --with-replica 1`. For each stage: if the lock lists a file the commit
  changed, it must show a main-run action; otherwise it must show zero main-run units, no replica action and exactly
  one canary unit. Record the table and the predicted cost of every re-run; report both to the owner. A miss is a
  finding before the push.
- [ ] **Step 5: OWNER GATE — push** (a separate yes). CI green.
- [ ] **Step 6: The E-2 real run (owner ruling 11b step 2).** `campaign ... preflight`, then `campaign ... launch
  --with-replica 1 --watch` at the artifact HEAD, in the background. Before the launch, if Step 4 showed any re-run,
  get the owner's yes for its cost (or launch with those stages excluded by the plan view). Expected: zero main-run
  units; no replica action; one canary unit per attested stage, all byte-identical (`verify` reports the canary
  compares). Report the result to the owner. A canary mismatch quarantines the token (Task 17) and is an owner
  finding before any merge.

---

## Self-review (done at plan time)

- **Spec coverage:** S5.1 -> Tasks 2, 3; S5.2 -> 4, 7; S5.3 -> 4, 6 (wait-while-owned, remainder, same-n id,
  round-robin, action label, two campaigns); S5.4 -> 7; S5.5 -> 8; S5.6 -> 7 (cv), 19;
  S5.7 -> 9; S5.8 -> 10; S5.9 -> 11, 12, 12b, 13, 14 (incl. .pyc, preload, toy root, Hub offline + prefetch), 15; S5.10 -> 16, 17; S5.11 -> 0.1, 5; S5.12 -> 0.4, 18, 22;
  S5.13 -> 15; S7.1 -> 0.3, 22, 23, 24 (item 3 = Task 24 Steps 4 and 6); S7.2 -> 4, 20, 23; S6 sweeps -> 2, 11, 13, 16; R1 -> 12; R2 -> 0.3; R3 -> 8;
  R4 -> 15 (`size_bytes`, `cmd_status`); R5 -> 16; R6 -> 14 Step 5b; R7 -> 12; R8 -> 0.2, 4; S5.12 statsbombpy launch
  test -> 18.
- **Placeholders:** none. Every test of Tasks 2–20 is written out; the workflow pins are real SHAs; every symbol the
  plan imports exists at `74d57ba` or is created by an earlier task (r2 A R2-01, B R2-08). Steps that depend on a measurement (Task 0.3 into
  Task 3's `[store]` values, Task 0.4 into Task 18's check flag, Task 0.2 into Task 4's label form) name the task that
  produces the value.
- **Commits:** one code commit (Task 21; amended once in Task 22 for the measured image lock and the per-unit history,
  each step behind an owner yes) and one artifact commit (Task 24). No other commit exists in this plan (owner rule:
  minimal commits dictated by clean provenance).
- **Type consistency:** `UnitStates`, `read_results`, `Ownership`, `make_action_id`, `plan_reduce`/`ReducePlan`,
  `stage_code_identity`, `install_stage_checks`, `stage_context_from_json`, `quarantine_generation`, `canary_key`,
  `stamp_provenance`, `compute_bars` are each defined once and used with the same signature.

---

## Change log

- r1 (2026-10-21): first plan from the approved spec r10 and owner ruling 11.
- r2 (2026-10-21): plan review r1 — A APPROVE (0 findings); B REQUEST CHANGES. Every B finding is fixed:
  - PLAN-01 (BLOCKING): a stage whose pending units are all owned WAITS; a reduce only when none is pending; test.
  - PLAN-02: a partial grant fills k jobs under the job length limit and leaves a remainder (`StagePlan.remainder`).
  - PLAN-03: a stage absent from the lock keeps the commit identity; in a job a compliant stage absent refuses.
  - PLAN-04: the module check considers only `silly_kicks/` and `scripts/` sources (a repo-root venv passes).
  - PLAN-05: helper keys resolve through literal call sites; dataset parameters move the token through the identity.
  - PLAN-06: the store probe measures a RESULT read (data + sidecar + digest + decode); `per_result_read_s` = wall at
    the safe cap ÷ results.
  - PLAN-07: Task 4 asserts only its own marks; the trainer-closure test moves to Task 12; the env registry becomes
    Task 12b (before Task 13); `_image_identity()` uses `launch.IMAGE` until Task 18's `image.image_ref()`; Task 22
    exports the history before any lock edit.
  - PLAN-08: steps and tests for the Hub prefetch and `HF_HUB_OFFLINE` from the process environment, the `.pyc`,
    preload and toy-root checks, the four S5.3 tests, the statsbombpy launch test, R4 (`size_bytes`) and R6 (Step 5b).
  - PLAN-09: the R1 measurement reports the harness members of every closure (Task 12 Step 5); a boundary is an
    owner ruling. PLAN-10: the workflow YAML written as it must run. PLAN-11: the population moves to
    `scripts/_pipeline_population.py`. PLAN-12: one writer per attempt (`O_EXCL` marker). PLAN-13: no second mount;
    the canary allow file. PLAN-14: the attestation compare excludes run history only. PLAN-15: `tests/_scale_guarded.py`.
- r3 (2026-10-21): plan re-review r2 — A APPROVE (1 CONSIDER); B APPROVE WITH FINDINGS (5 SHOULD FIX, 4 CONSIDER).
  - A R2-01: the Task 18 test imports `registry.REGISTRY` (`registry.py:102`).
  - B R2-01: the partial-grant test moves to Task 7 (it needs the grant and the remainder) with numbers that leave
    a remainder (40 x 2000 s, W = 8 -> 26 taken, 14 left).
  - B R2-02: the worker/reduce and inner-pool tests start with `HF_HUB_OFFLINE=1`; the inner test also asserts no
    `BrokenProcessPool`.
  - B R2-03: `closures.py` uses a stdlib AST resolver (namespace-safe; parent `__init__.py` files included) instead of
    `modulefinder`, which crashes on the `scripts` namespace package; tests on the real tree.
  - B R2-04: `all_script_trees` and `call_name` move to `scripts/_pipeline_population.py` with the population; a
    fresh-interpreter test proves no `tests` import.
  - B R2-05: the toy-root test opens a real tracked fixture and fails without the root.
  - B R2-06..09: `monkeypatch` in the Hub test; `stage_code_identity -> str | None`; real SHA pins in the YAML; a
    stage-level offline Hub test under `install_stage_checks`. The `check_concurrency` step states the API fact
    instead of a search.
- r4 (2026-10-21): plan re-review r3 — A APPROVE (0 findings); B APPROVE WITH FINDINGS (1 SHOULD FIX, 2 CONSIDER).
  - B R3-01: the unit-level Hub test installs the checks with the REAL runner closure (`module_closure("scripts._stages")`),
    so it fails in the unit, for the right reason.
  - B R3-02: the data root normalizes to a repo-relative path at install (an absolute `ctx.data_root` now matches).
  - B R3-03: `test_compute_contract`'s `_calls` moves with the population (`calls()`).
- r5 (2026-10-21): owner ruling 12 (B r3, outside the round): the job wall is the LPT makespan of a job's units on its
  W workers (`planner.job_makespan`), not a fluid `work / W`; partitions keep LPT order and the executor runs a
  `--shard-keys` list in that order. The Task 7 partial-grant test now takes 8 units per job (16 taken, 24 left).
- r6 (2026-10-21): plan re-review r4 — A APPROVE (0 findings); B APPROVE (1 CONSIDER). B R4-01: the job-length fill keeps
  each bin's worker loads incrementally (exactly `job_makespan` of the bin, at O(log W) per check); a property test and a
  timing bound (3961 units under 1 s).
- r7 (2026-10-21): owner ruling — the Cycle 1 artifact work is cut and PR #288 merges without a version bump. Task 1
  branches off `main` after that merge; Task 22 Step 3's export also writes the stage cost rows (no Cycle 1 summary
  exists) and names the Cycle 1 commit.

- (2026-10-21) Owner ruling: this cycle is named **Cycle 1b** (`cycle1b.toml`, `cycle1b.json`), so ADR-128 section 11's
  roadmap keeps its numbers (Cycle 2 = the three public trainers; Cycle 3+ = the corpus drivers). Every "Cycle 2" /
  `cycle2` of earlier revisions now reads "Cycle 1b" / `cycle1b`.
