# HF Jobs Parallel Training Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan **INLINE**, task-by-task. **Subagents are banned on this project (owner mandate) — do NOT use superpowers:subagent-driven-development.** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the five heavy silly-kicks trainers into a uniform three-phase map/reduce so extraction and CV/HPO work fan out across small `cpu-upgrade` HF Jobs (with a serial, byte-identical reduce), fix the owner-tier re-fit defects (floor feature-vintage, xcross TF-19 probe, repackage), and validate the whole pipeline e2e on the DGX before any HF spend.

**Architecture:** Each heavy trainer exposes `extract` (fan-out by `--shard-index I/N`) → `fold`/`study` (fan-out) → `assemble` (serial reduce; final fit + probe + importance). `launch.py` orchestrates the stages as a local DAG over an injected runner — an HF-Jobs runner (`hf jobs run … -- bash -lc`) or a local-subprocess runner (the DGX e2e + fallback executor). The head-to-head no-regression floor scores each variant at its own feature vintage. Storage is a read-write HF Storage bucket mounted at `/scratch`; every phase resumes via `for_each` (ADR-052) + `--attempts`.

**Tech Stack:** Python 3.12, pandas/numpy/scikit-learn/xgboost, `scripts/_driver.py` (`for_each`/`generation_dir`/`shard_path`), the `hf` CLI ≥ v2.2.0 (`--volume`/`--attempts`/`inspect --format json`), HF Storage buckets. No new `silly_kicks/` runtime dependency.

> **Amendment A (2026-10-14, owner ruling — see the spec's Amendment A).** The code that runs on HF is
> DATASET-AGNOSTIC: NO pining info to HF (token/manifest/`SILLY_KICKS_PINING_MANIFEST_DIR`/pining match-ids),
> NO `_loader_pining`/`--providers` in-job. Every owner trainer reads a staged neutral parquet dataset via a
> data path; staging (pining on the DGX, or anyone's loader) is off-HF. Task 9 is therefore **xcross
> `--probe-data-dir`** (staged per-game GS probe slice; `TrainerSpec.probe`; launch `probe_repo`→`/tmp/probe`),
> NOT `owner_pining`. **Receiver** gains a `--data-dir` agnostic source (`_data_dir_source`, flat shards+`_actions`);
> `owner_sb360` reads `corpus_repo` via `--data-dir`+`--shard-root`, no manifest; the `sb360_repo` launch param
> is removed. Task 12 staging produces ONLY neutral parquet. Ghost adds a race-free **`--build-cache`** phase
> between extract and fold. All Status DONE+green: Tasks 1–9 + receiver (ruff+pyright clean, 83 pass/3 @slow-skip).

**Spec:** `docs/superpowers/specs/2026-10-14-hf-jobs-parallel-harness-design.md` (read it alongside this plan; the plan argues from it).

## Global Constraints

- **Flavor:** every `REGISTRY` spec's `TrainerSpec.flavor` is `cpu-upgrade` (8 vCPU / 32 GB / $0.03/hr); the BLOCKED `gk_retention` spec (`registry.py:261`, never launched) is not re-flavored. The ONLY possible exception is the serial reduce phase, and only on an explicit owner ruling after the DGX e2e measures its peak RSS (spec §6/§8/§11) — never a default.
- **Determinism contract:** for every heavy trainer, the fanned pipeline (`extract` + `fold`/`study` + `assemble`) produces **byte-identical** shipped weights + `metrics.json` to a single-box serial run. Every determinism test uses a toy corpus spanning **≥ 2 shards and ≥ 2 CV folds** (an N=1 fan-out passes vacuously).
- **Final fit is serial and faithful:** only extraction, CV folds, and HPO studies fan out. The single final model fit and the permutation importance stay in the serial reduce. No model/feature/metric change — this is a re-materialization harness.
- **Faithfulness:** owner-tier re-fits reproduce the committed corpus, the per-variant feature vintage, and (xcross only) the TF-19 probe. xshot ships no probe; only xcross threads a probe config.
- **DGX e2e gate:** no HF spend until each trainer's full pipeline is green on the DGX against `~/tc3-kinematics`, including every phase's peak RSS under 32 GB (or an owner ruling on any phase that exceeds it).
- **Commit discipline (OVERRIDES the skill's per-task "Commit" steps):** tasks end at green tests. There are **NO per-task commits**. The entire plan lands as **one** owner-approved harness-tune commit on `feat/hf-jobs-training-harness` (not amending `6f6508a`), after all tasks are green AND the DGX e2e passes AND both independent reviews clear. Weights land in a subsequent owner-approved commit; PR #284 merges non-squash. No commit / push / merge / tag / HF spend without explicit per-action owner approval.
- **Lint/type:** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright` scoped to changed files locally (CI is authoritative). The hour-long suite is not run locally.

---

### Task 1: Shard-selection partition helper

A pure helper that slices an ordered item sequence into a disjoint cover for `--shard-index I/N`. Shared by every trainer's extraction phase so the partition logic (and its CI guard) lives in one place.

**Files:**
- Modify: `scripts/_driver.py` (add `shard_select` near `shard_path`)
- Test: `tests/scripts/test_driver_shard_select.py`

**Interfaces:**
- Produces: `shard_select(seq: Sequence[T], index: int, n: int) -> list[T]` — returns `list(seq)[index::n]`; validates `0 <= index < n` and `n >= 1` (raises `ValueError` otherwise). The union over `index in range(n)` is exactly `seq` with no overlap.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_driver_shard_select.py
import pytest
from scripts._driver import shard_select

def test_disjoint_cover_over_two_or_more_shards():
    seq = [f"m{i}" for i in range(7)]
    parts = [shard_select(seq, i, 3) for i in range(3)]
    # disjoint
    seen = [x for p in parts for x in p]
    assert len(seen) == len(set(seen))
    # complete cover, order-preserving union
    assert sorted(seen) == sorted(seq)
    # every shard non-trivial for N<=len (the >=2-shards determinism precondition)
    assert all(len(p) >= 1 for p in parts)

def test_n_one_is_identity():
    seq = [1, 2, 3]
    assert shard_select(seq, 0, 1) == [1, 2, 3]

@pytest.mark.parametrize("index,n", [(-1, 3), (3, 3), (0, 0)])
def test_rejects_bad_bounds(index, n):
    with pytest.raises(ValueError):
        shard_select([1, 2, 3], index, n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_driver_shard_select.py -v`
Expected: FAIL with `ImportError: cannot import name 'shard_select'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/_driver.py (add near shard_path)
def shard_select(seq, index, n):
    """Disjoint-cover slice for --shard-index I/N: list(seq)[index::n].

    The union over index in range(n) is exactly seq, with no overlap -- the invariant
    every fan-out extraction relies on (and the determinism tests assert with >=2 shards).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0 <= index < n):
        raise ValueError(f"index must be in [0, {n}), got {index}")
    return list(seq)[index::n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/test_driver_shard_select.py -v`
Expected: PASS.

- [ ] **Step 5: No commit** (see Commit Discipline). Run `python -m ruff check scripts/ tests/` and fix any lint on the touched files.

---

### Task 2: Flavor policy → cpu-upgrade + re-tuned timeouts

**Files:**
- Modify: `scripts/hf_jobs/registry.py:128-253` (the five owner-tier specs)
- Test: `tests/scripts/hf_jobs/test_registry_flavor.py`

**Interfaces:**
- Consumes: `registry.REGISTRY` (Task 0 state).
- Produces: every `TrainerSpec.flavor == "cpu-upgrade"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/hf_jobs/test_registry_flavor.py
from scripts.hf_jobs import registry as reg

def test_every_trainer_runs_on_cpu_upgrade():
    # Owner directive 2026-10-14: cpu-upgrade only until further notice (spec Global Constraints).
    for name, spec in reg.REGISTRY.items():
        assert spec.flavor == "cpu-upgrade", f"{name} is {spec.flavor}"

def test_blocked_trainer_is_untouched():
    assert "gk_retention" in reg.BLOCKED  # still blocked; not re-flavored into REGISTRY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_registry_flavor.py -v`
Expected: FAIL — `ghost_gk is cpu-performance`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/hf_jobs/registry.py`, set `flavor="cpu-upgrade"` on `ghost_gk` (`:132`), `ghost_outfield` (`:170`), `xshot_occurrence` (`:194`), `xcross_attempt` (`:218`), `receiver_model` (`:242`). Leave the BLOCKED `gk_retention` spec (`:261`) untouched. Re-tune `timeout` for the new per-phase jobs (each fanned unit is far shorter than the old monolith): set the per-phase ceilings the orchestrator passes per phase (Task 10 owns per-phase timeouts); keep `TrainerSpec.timeout` as the assemble-phase ceiling — `ghost_gk`/`ghost_outfield` `"3h"`, `xshot_occurrence`/`xcross_attempt` `"2h"`, `receiver_model` `"2h"`. Leave the four public specs unchanged (already `cpu-upgrade`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_registry_flavor.py -v`
Expected: PASS.

- [ ] **Step 5: No commit.** Ruff the touched files.

---

### Task 3: repackage emits actions.parquet; drop the launch symlink workaround

**Files:**
- Modify: `scripts/hf_jobs/repackage.py:57-85` (`flat_to_pergame`)
- Modify: `scripts/hf_jobs/launch.py:66-80` (`_corpus_steps` owner_pergame branch — remove the symlink step)
- Test: `tests/scripts/hf_jobs/test_repackage.py` (extend), `tests/scripts/hf_jobs/test_launch.py:105-113` (update `test_owner_pergame_repackages_flat_to_per_game_in_job`)

**Interfaces:**
- Produces: `flat_to_pergame` writes `{game}/frames.parquet` (symlink) + `{game}/shots.parquet` + `{game}/actions.parquet` (symlink to the staged `_actions/<provider>__<game>.parquet`).

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/hf_jobs/test_repackage.py (add)
import pandas as pd
from scripts.hf_jobs.repackage import flat_to_pergame

def test_emits_actions_parquet_per_game(tmp_path):
    flat = tmp_path / "flat"; (flat / "shards" / "g").mkdir(parents=True); (flat / "_actions").mkdir()
    frames = pd.DataFrame({"game_id": ["idsse__M1"], "x": [1.0]})
    frames.to_parquet(flat / "shards" / "g" / "idsse__M1.parquet")
    acts = pd.DataFrame({"type_id": [21, 11], "type_name": ["shot", "pass"], "game_id": ["idsse__M1"] * 2})
    acts.to_parquet(flat / "_actions" / "idsse__M1.parquet")
    out = tmp_path / "pergame"
    flat_to_pergame(flat, out)
    gdir = out / "idsse-M1"
    assert (gdir / "frames.parquet").exists()
    assert (gdir / "shots.parquet").exists()
    assert (gdir / "actions.parquet").exists()                       # NEW
    # actions.parquet is the FULL SPADL actions (xcross reads it), not just shots
    assert len(pd.read_parquet(gdir / "actions.parquet")) == 2
```

```python
# tests/scripts/hf_jobs/test_launch.py — replace the symlink assertion in test_owner_pergame_*
    assert "python -m scripts.hf_jobs.repackage /tmp/corpus /tmp/pergame" in inner
    assert "symlink_to" not in inner                                  # workaround GONE
    assert "--data-dir /tmp/pergame --output-dir" in inner
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/hf_jobs/test_repackage.py -v tests/scripts/hf_jobs/test_launch.py -k owner_pergame -v`
Expected: FAIL — `actions.parquet` missing; `symlink_to` still present in the inner command.

- [ ] **Step 3: Write minimal implementation**

In `flat_to_pergame` (after the `shots.parquet` write, ~`:83`), add the actions symlink (mirror the `frames.parquet` symlink-or-copy fallback for unprivileged Windows CI):

```python
    if apath.exists():
        alink = gdir / "actions.parquet"
        if alink.exists() or alink.is_symlink():
            alink.unlink()
        try:
            os.symlink(apath.resolve(), alink)
        except (OSError, NotImplementedError):
            shutil.copy(apath, alink)
```

In `launch._corpus_steps` owner_pergame branch (`:66-80`), delete the `steps.append("python -c \"import pathlib; ad=…symlink_to…\"")` workaround block and its comment; keep the `repackage` step.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/hf_jobs/test_repackage.py tests/scripts/hf_jobs/test_launch.py -v`
Expected: PASS.

- [ ] **Step 5: No commit.** Ruff the touched files.

---

### Task 4: Floor — feature-vintage-correct eval per variant

The head-to-head eval must extract features at the variant's `feature_set`, so a 26-feature `position_only` booster is scored on a 26-feature frame (xgboost `predict_proba` is strict). Fixes the `ValueError: feature_names mismatch` that crashed `xshot_occurrence/position_only`.

**Files:**
- Modify: `scripts/hf_jobs/registry.py` (add `feature_set: str = "faithful"` to `VariantBuild`; set it per variant)
- Modify: `scripts/hf_jobs/headtohead.py:411-426` (`build_eval` gains `feature_set`), `:597-632` (`_owner_eval_classifier`/`_owner_eval_xshot`/`_owner_eval_xcross` pass it to `prepare_*`), `:705-719` (`main` adds `--feature-set`)
- Modify: `scripts/hf_jobs/launch.py:127-128` (thread `--feature-set {build.feature_set}` into the headtohead call)
- Test: `tests/scripts/hf_jobs/test_headtohead_vintage.py`

**Interfaces:**
- Consumes: `VariantBuild.feature_set`.
- Produces: `build_eval(name, corpus_dir, *, feature_set="faithful")`; `headtohead` CLI `--feature-set`.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/hf_jobs/test_headtohead_vintage.py
from scripts.hf_jobs import headtohead as h2h
from scripts.hf_jobs import registry as reg

def test_build_eval_accepts_feature_set_kw():
    # fail-loud when corpus absent, but the SIGNATURE must accept feature_set
    import pytest
    with pytest.raises(h2h.EvalDataUnavailableError):
        h2h.build_eval("xshot_occurrence", corpus_dir=None, feature_set="position_only")

def test_position_only_variant_carries_its_feature_set():
    xshot = reg.REGISTRY["xshot_occurrence"]
    po = next(b for b in xshot.builds if b.variant == "position_only")
    assert po.feature_set == "position_only"
    df = next(b for b in xshot.builds if b.variant == "default")
    assert df.feature_set == "faithful"
```

Add a scoring test with a tiny 26-column position_only frame proving no mismatch (mirror `tests/scripts/hf_jobs/test_headtohead.py::synthetic_bundle` for `xshot_occurrence`, but build the frame at `feature_set="position_only"` column count and load a position_only committed stub).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_headtohead_vintage.py -v`
Expected: FAIL — `VariantBuild` has no `feature_set`; `build_eval` has no `feature_set` kw.

- [ ] **Step 3: Write minimal implementation**

In `registry.py`, add `feature_set: str = "faithful"` to the `VariantBuild` dataclass and set it on the xshot/xcross/ghost variants (`position_only` builds → `"position_only"`). In `headtohead.build_eval`, add `feature_set="faithful"` and forward it to `_build_owner_eval(name, corpus_dir, feature_set)`; `_owner_eval_xshot`/`_owner_eval_xcross` pass `feature_set=` through `_owner_eval_classifier` into `prepare_xshot_training_data`/`prepare_xcross_training_data` (both already accept `feature_set`). `main` adds `ap.add_argument("--feature-set", default="faithful")` and passes it to `build_eval`. In `launch.build_inner_command`, append `--feature-set {build.feature_set}` to the headtohead command.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_headtohead_vintage.py tests/scripts/hf_jobs/test_launch.py -v`
Expected: PASS.

- [ ] **Step 5: No commit.** Ruff the touched files.

---

### Task 5: ghost_gk map/reduce seam (largest change)

Add `--shard-index I/N` (extract a disjoint file subset), `--fold K` (run one CV fold from the built cache), and `--assemble` (reduce: build cache from shards → aggregate folds → final fit → parity gate → serial importance → weights + metrics). Byte-identical to the serial monolith.

**Files:**
- Modify: `scripts/train_ghost_gk.py` — args (`:220-312`), the parquet list (`:413`-ish) + `_items()` (`:607`), the CV loop (`:844-917`), the final-fit/importance block (`:919-1000`)
- Test: `tests/tracking/test_train_ghost_gk_mapreduce.py`

**Interfaces:**
- Consumes: `scripts._driver.shard_select` (Task 1).
- Produces: ghost_gk CLI flags `--shard-index I/N`, `--fold K`, `--assemble`; a fold result file `{shard_root}/fold_{K}.json`; the reduce reads `fold_*.json` + the feature cache and ships identically to serial.

- [ ] **Step 1: Write the failing determinism test**

```python
# tests/tracking/test_train_ghost_gk_mapreduce.py
# Toy corpus: >=2 parquet files (>=2 shards) covering >=2 games so >=2 CV folds exist.
# Run A (serial): python -m scripts.train_ghost_gk --data-dir toy --home-teams ht.json --output-dir A
# Run B (fanned): for i in 0,1: --shard-index i/2 --output-dir B ; then --fold 0..k --output-dir B ;
#                 then --assemble --output-dir B
# Assert: GhostGkModel.load(A).npz arrays == load(B); A/metrics.json CV aggregate == B (allclose 0).
def test_fanned_equals_serial_byte_identical(tmp_path, toy_ghost_corpus):
    a = _run_serial(tmp_path / "A", toy_ghost_corpus)
    b = _run_fanned(tmp_path / "B", toy_ghost_corpus, n=2, folds=2)
    assert _weights_bytes(a) == _weights_bytes(b)
    assert _cv_metrics(a) == _cv_metrics(b)
```

(`toy_ghost_corpus` fixture: 2 parquet frame files, 2+ games, a `home_teams.json`; small enough for a real fit. `_run_serial`/`_run_fanned` shell out to the module; `_weights_bytes` reads the npz + metadata bytes.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_train_ghost_gk_mapreduce.py -v`
Expected: FAIL — unrecognized args `--shard-index`/`--fold`/`--assemble`.

- [ ] **Step 3: Write the implementation**

Add the args:
```python
    parser.add_argument("--shard-index", default=None,
        help="I/N: extract only this disjoint file subset (fan-out worker); shards go to the shared shard_root.")
    parser.add_argument("--fold", type=int, default=None,
        help="Run ONLY CV fold K (reads the built feature cache) and write {shard_root}/fold_K.json, then exit.")
    parser.add_argument("--assemble", action="store_true",
        help="Reduce: build the cache from shards, aggregate fold_*.json, final fit + importance, ship. Then exit.")
```

Partition the extraction (after `parquets` is built, ~`:413`):
```python
    if args.shard_index is not None:
        from scripts._driver import shard_select
        i, n = (int(x) for x in args.shard_index.split("/"))
        parquets = shard_select(parquets, i, n)
    # A --shard-index worker extracts its subset to the shared shard_root and EXITS before CV
    # (the feature cache is whole-corpus; only --assemble writes it).
    if args.shard_index is not None and not args.assemble:
        _extract_only(parquets, ...)   # runs the for_each at :724 over this subset, then return
        return
```

Refactor the existing CV loop body (`:847-908`) into `run_one_fold(features, labels, groups, provider_labels, fold, folds, grid_spec, cp, args) -> dict` (the per-fold dict already assembled at `:896-907`). **Both CV regimes must be preserved** — ghost has two (`:811-842`): the default `StratifiedGroupKFold(random_state=42)` AND `--keeper-grouped` (`keeper_folds` over the common domain, `:823/:833`). The `--fold` worker and `--assemble` compute `folds` by the SAME regime branch as the serial loop (switch on `args.keeper_grouped`), both deterministic. `--fold K`: build/load the cache (the existing cache-hit branch), compute `folds` via that branch, run `run_one_fold(..., K, folds, ...)`, write `{shard_root}/fold_{K}.json`, exit. `--assemble`: build the cache from shards (the existing fresh-extract branch resumes every shard), load `fold_*.json` into `fold_metrics` (instead of looping), then the unchanged aggregate + final fit + parity gate + importance + publish. A plain serial run (no flags) keeps the current inline fold loop. The fold split and final fit are seed-fixed, so fanned == serial. **Committed ghost weights use the DEFAULT regime** (`--keeper-grouped` OFF — the shipped headline metrics are match-grouped, arg help `:288-289`), so the faithful re-fit runs stratified; the determinism test (Step 1) exercises the shipped (stratified) regime.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_train_ghost_gk_mapreduce.py -v`
Expected: PASS — fanned weights + CV metrics byte-identical to serial.

- [ ] **Step 5: No commit.** Ruff + `pyright scripts/train_ghost_gk.py`.

---

### Task 6: ghost_outfield map/reduce seam

Mirror Task 5 in `scripts/train_ghost_outfield.py` (it already mirrors ghost_gk's `for_each` extraction at `:362`). Same `--shard-index`/`--fold`/`--assemble` surface, same `run_one_fold` refactor of its CV loop, same determinism test (`tests/tracking/test_train_ghost_outfield_mapreduce.py`) with a ≥2-shard/≥2-fold toy corpus.

**Files:**
- Modify: `scripts/train_ghost_outfield.py` (args, `_items`/for_each `:336-362`, CV loop, final fit)
- Test: `tests/tracking/test_train_ghost_outfield_mapreduce.py`

**Interfaces:**
- Consumes: `scripts._driver.shard_select`.
- Produces: same flag surface as Task 5 for ghost_outfield.

- [ ] **Step 1:** Write the fanned==serial determinism test (mirror Task 5 Step 1 for ghost_outfield; `GhostOutfieldModel` weights bytes).
- [ ] **Step 2:** Run → FAIL (unknown args).
- [ ] **Step 3:** Add `--shard-index`/`--fold`/`--assemble`; `shard_select(parquets)`; extract `run_one_fold`; `--assemble` reads `fold_*.json`. Final fit serial. Preserve ghost_outfield's CV regime branch exactly as its serial loop has it (the `--fold` worker computes the same deterministic split), mirroring Task 5's two-regime handling.
- [ ] **Step 4:** Run → PASS (byte-identical).
- [ ] **Step 5:** No commit. Ruff + pyright the file.

---

### Task 7: receiver_model extraction fan-out

`receiver_model` has a `for_each` extraction (`:387/:421`, `--shard-root` required) then a single `ReceiverModel.fit` (`:545`). Add `--shard-index I/N` to the per-provider extraction so N workers write disjoint shards to the shared `shard_root`; the serial reduce (no `--shard-index`) runs the fit from the resumed shards. No fold fan-out (the fit is a single model; receiver has no CV-fold seam).

**Files:**
- Modify: `scripts/train_receiver_model.py` (args + `_extract_provider_rows` `:415-421`)
- Test: `tests/tracking/test_train_receiver_mapreduce.py`

**Interfaces:**
- Consumes: `scripts._driver.shard_select`.
- Produces: receiver CLI `--shard-index I/N` (extraction only).

- [ ] **Step 1:** Write the test: two `--shard-index 0/2` and `1/2` extraction runs to one `shard_root`, then a serial reduce; assert the fitted `ReceiverModel` weights == a single serial run (byte-identical).
- [ ] **Step 2:** Run → FAIL (unknown `--shard-index`).
- [ ] **Step 3:** Add `--shard-index`; in `_extract_provider_rows`, `refs = shard_select(refs, i, n)` before `for_each`; a `--shard-index` worker extracts and exits before the fit.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** No commit. Ruff + pyright.

---

### Task 8: xshot/xcross extraction fan-out (seam otherwise present)

`xshot_occurrence`/`xcross_attempt` already have the study seam (`--list-studies`/`--study`/`--study-list`/`--assemble`/`--shard-root`, byte-identical per their docstrings). Add `--shard-index I/N` to their extraction prep so extraction fans out too (today the serial prep extracts all matches before studies).

**Files:**
- Modify: `scripts/train_xshot_occurrence.py` (`_extract`/for_each `:135`, args), `scripts/train_xcross_attempt.py` (`_extract`/for_each `:165`, args `:1060`)
- Test: `tests/tracking/test_xshot_xcross_shard_index.py`

**Interfaces:**
- Consumes: `scripts._driver.shard_select`.
- Produces: `--shard-index I/N` on both trainers' extraction; the existing `--assemble` reduces the fanned shards byte-identically.

- [ ] **Step 1:** Write the test: `--shard-index 0/2` + `1/2` extraction into one `shard_root`, then `--assemble`; assert identical shipped weights to a serial `--shard-root` run; and that `shard_select` over the match refs is a disjoint cover.
- [ ] **Step 2:** Run → FAIL (unknown `--shard-index`).
- [ ] **Step 3:** Add `--shard-index` to both; slice the match refs (`refs = shard_select(refs, i, n)`) before the extraction `for_each`; a `--shard-index` worker extracts its shards and exits (the existing `--study`/`--assemble` path reads the shared `shard_root`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** No commit. Ruff + pyright both files.

---

### Task 9: xcross faithful TF-19 probe threading

The committed xcross used a separate held-out gradientsports probe pass (`probe_sample_matches=[['gradientsports','10502'],['gradientsports','10503']]`, via `--probe-match-ids-json`). Record that config in the registry and thread it through launch so `assemble_studies` reproduces the probe instead of refusing. xshot ships no probe.

**Files:**
- Modify: `scripts/hf_jobs/registry.py` (add `probe_match_ids: dict | None = None` to the xcross spec; `{"gradientsports": ["10502", "10503"]}`)
- Modify: `scripts/hf_jobs/launch.py` (`_trainer_cmd`/`build_inner_command` append `--probe-match-ids-json <path>` + `--providers idsse,skillcorner` for xcross when `probe_match_ids` is set; write the JSON to a temp path in the inner command)
- Test: `tests/scripts/hf_jobs/test_launch_probe.py`

**Interfaces:**
- Consumes: `TrainerSpec.probe_match_ids`.
- Produces: xcross inner command carries `--probe-match-ids-json` + `--providers idsse,skillcorner`; no probe flags for any other trainer.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/hf_jobs/test_launch_probe.py
from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import build_inner_command

def test_xcross_threads_probe_config():
    spec = reg.REGISTRY["xcross_attempt"]; b = spec.builds[0]
    inner = build_inner_command(spec, b, tag="r", run_key="r1", corpus_repo="ns/xshot")
    assert "--probe-match-ids-json" in inner
    assert '"gradientsports"' in inner and "10502" in inner and "10503" in inner
    assert "--providers idsse,skillcorner" in inner

def test_xshot_has_no_probe():
    spec = reg.REGISTRY["xshot_occurrence"]; b = spec.builds[0]
    inner = build_inner_command(spec, b, tag="r", run_key="r1", corpus_repo="ns/xshot")
    assert "probe-match-ids" not in inner
```

- [ ] **Step 2:** Run → FAIL (no probe flags emitted).
- [ ] **Step 3:** Add `probe_match_ids` to the xcross spec; in `build_inner_command`, when `spec.probe_match_ids` is set, add a step writing the JSON to `/tmp/probe.json` and append `--probe-match-ids-json /tmp/probe.json --providers idsse,skillcorner` to the trainer command. (The trainer's existing disjoint-probe guard refuses a probe id that overlaps training — honored because `10502/10503` are gradientsports, not in idsse/skillcorner.)
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** No commit. Ruff. (Staging the GS probe matches is Task 12.)

---

### Task 10: launch.py staged orchestrator (map/reduce DAG + local runner)

Sequence the phases as a DAG over an injected runner: HF-Jobs runner (`hf jobs run`) or local-subprocess runner (DGX). Each stage launches its jobs, waits for all terminal, fails loud on any ERROR.

**Files:**
- Modify: `scripts/hf_jobs/launch.py` (add `run_phase`, `run_stage`, `wait_all`, `run_pipeline`, `local_runner`)
- Test: `tests/scripts/hf_jobs/test_launch_orchestrator.py`

**Interfaces:**
- Consumes: `build_inner_command`, `build_argv`, `poll_until_terminal`, `shard_select`.
- Produces:
  - `run_stage(phase_cmds: list[list[str]], *, runner) -> list[str]` — launches each argv, returns job ids.
  - `wait_all(job_ids, *, runner, poll=poll_until_terminal) -> dict[str, str]` — polls each to terminal; raises `StagePhaseError` if any is `ERROR`/`CANCELED`.
  - `run_pipeline(spec, *, run_key, tag, corpus_repo, sb360_repo, bucket, attempts, n_extract, runner) -> dict` — extract stage (`n_extract` `--shard-index` jobs) → wait → fold/study stage → wait → assemble+floor job → wait; returns the final stage ids.
  - `local_runner(argv) -> tuple[str, int]` — runs the inner `bash -lc` payload as a local subprocess (DGX), parsing a synthetic job id; `poll_until_terminal` short-circuits for local runs (already terminal).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/hf_jobs/test_launch_orchestrator.py
from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import run_stage, wait_all, run_pipeline, StagePhaseError

def test_wait_all_raises_on_terminal_error():
    import json
    import pytest
    def runner(argv):  # the inspect call reports one ERROR -> poll_until_terminal returns "ERROR"
        return json.dumps([{"id": argv[3], "status": {"stage": "ERROR"}}]), 0
    with pytest.raises(StagePhaseError):
        wait_all(["ns/j1"], runner=runner)

def test_run_pipeline_sequences_extract_then_fold_then_assemble():
    seen = []
    def runner(argv):
        seen.append(argv)
        if argv[:3] == ["hf", "jobs", "inspect"]:               # poll -> always COMPLETED
            import json
            return json.dumps([{"id": argv[3], "status": {"stage": "COMPLETED"}}]), 0
        h = f"6abcde{len(seen)}"                                 # launch -> emit an id
        return f"id={h} url=https://huggingface.co/jobs/ns/{h}", 0
    run_pipeline(reg.REGISTRY["ghost_gk"], run_key="r1", tag="r", corpus_repo="ns/tc3",
                 sb360_repo=None, bucket="ns/b", attempts=1, n_extract=2, runner=runner)
    kinds = [" ".join(a) for a in seen]
    assert any("--shard-index 0/2" in k for k in kinds) and any("--shard-index 1/2" in k for k in kinds)
    assert any("--fold 0" in k for k in kinds)
    assert any("--assemble" in k for k in kinds)
    # order: every extract job launched before any fold; every fold before assemble
```

- [ ] **Step 2:** Run → FAIL (`run_stage`/`wait_all`/`run_pipeline`/`StagePhaseError` undefined).
- [ ] **Step 3:** Implement `StagePhaseError`; `run_stage` (loop `launch`-style over argvs via the runner, collect ids); `wait_all` (poll each id, raise on ERROR/CANCELED); `run_pipeline` building the per-phase inner commands (`--shard-index i/n` for extract, `--fold k` for ghost / `--study-list` for xshot/xcross, `--assemble` + headtohead for the reduce) and sequencing extract→fold/study→assemble with `wait_all` between; `local_runner` (subprocess `bash -lc` of the inner, synthetic id, exit code → terminal). Per-phase timeouts set here (extract/fold short, assemble = `spec.timeout`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** No commit. Ruff + pyright `scripts/hf_jobs/launch.py`.

---

### Task 11: DGX e2e driver with per-phase peak-RSS measurement

Runs each trainer's full pipeline locally on the DGX via `run_pipeline(..., runner=local_runner)`, wrapping every phase subprocess to capture peak RSS, and asserts each phase's peak RSS < 32 GB (the cpu-upgrade ceiling). Blocks the HF wave on any breach pending an owner ruling (spec §8/§11).

**Files:**
- Create: `scripts/hf_jobs/dgx_e2e.py`
- Test: `tests/scripts/hf_jobs/test_dgx_e2e.py`

**Interfaces:**
- Consumes: `run_pipeline`, `local_runner`.
- Produces:
  - `measure_rss(argv) -> tuple[str, int, int]` — runs a phase subprocess, returns `(stdout+stderr, exit_code, peak_rss_bytes)` via `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` delta (or `/usr/bin/time -v` parse).
  - `e2e(trainer_name, corpus_dir, *, folds, n_extract, rss_ceiling=32*2**30) -> dict` — runs the pipeline with `measure_rss`, returns `{phase: {peak_rss, wall, ok}}`; a phase over the ceiling sets `ok=False` and the summary `blocked=True`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/hf_jobs/test_dgx_e2e.py
from scripts.hf_jobs.dgx_e2e import measure_rss, e2e

def test_measure_rss_reports_a_positive_peak():
    out, code, rss = measure_rss(["python", "-c", "x=bytearray(50_000_000); print('ok')"])
    assert code == 0 and "ok" in out and rss > 10_000_000

def test_e2e_flags_a_phase_over_the_ceiling():
    # inject a runner whose 'assemble' phase reports a huge RSS -> blocked=True
    summary = e2e("ghost_gk", corpus_dir="toy", folds=2, n_extract=2,
                  rss_ceiling=1, _runner=_fake_runner_reporting_big_rss)
    assert summary["blocked"] is True
```

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `measure_rss` (subprocess + `resource` peak RSS; on non-Linux the test skips), and `e2e` (drive `run_pipeline` with a measuring local runner; collect per-phase peak RSS + wall; compare to ceiling; return summary with `blocked`). The real DGX run is a manual gate (documented in the howto), not CI.
- [ ] **Step 4:** Run → PASS (`test_measure_rss` skips off-Linux).
- [ ] **Step 5:** No commit. Ruff + pyright.

---

### Task 12: Faithful staging scripts (DGX-side) incl xcross GS probe slice

Stage the owner corpora from the DGX to private HF datasets, preserving faithfulness: full TC3 (ghost), the 17 training ids + a separate gradientsports `10502/10503` probe slice (xcross), SB360 30 (receiver). Reuses the halted campaign's `~/hf_jobs_stage` recipe; adds the probe slice.

**Files:**
- Modify/Create: `scripts/hf_jobs/stage.py` (the staging entry points; add `stage_probe_slice`)
- Test: `tests/scripts/hf_jobs/test_stage.py` (the manifest/layout writer is unit-testable; the actual upload is a DGX run)

**Interfaces:**
- Produces: `stage_probe_slice(provider, match_ids, cache_dir, out_dir)` — writes the held-out probe matches into their own slice so the training glob does not pick them up; dataset names confirmed with the owner before any push.

- [ ] **Step 1:** Write a unit test that `stage_probe_slice` writes the probe matches under a distinct `probe/` prefix and does NOT place them under the training `shards/` glob (held-out by construction).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the probe-slice writer (mirror the existing `stage_corpus` layout; probe matches to `probe/`). The actual DGX staging run + `hf upload` is manual (documented), gated on owner-confirmed dataset names.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** No commit. Ruff + pyright.

---

### Task 13: Docs — ADR-123, howto, C4

**Files:**
- Create: `docs/superpowers/adrs/ADR-123-hf-jobs-parallel-harness.md` (re-confirm 123 is free at write time)
- Modify: `docs/howto/hf-jobs-training.md` (the map/reduce phases, `--shard-index`/`--fold`/`--assemble`, the DGX e2e gate + RSS measurement, the reduce-RAM owner-hold, cpu-upgrade-only, the probe-slice staging)
- Modify: `docs/c4/architecture.{dsl,html}` (regen via the pinned Graphviz `dot`; the orchestrator + DGX e2e + staging datasets)
- Spec §11 cleanup already applied.

**Interfaces:** none (docs travel WITH the single harness-tune commit).

- [ ] **Step 1:** Write ADR-123 (context = the halted campaign's four failures + the owner directives; decision = uniform three-phase map/reduce, cpu-upgrade-only with the measured reduce-RAM hold, HF-primary/DGX-validate-and-fallback; consequences incl the determinism contract + the faithful-probe staging). Reference ADR-122.
- [ ] **Step 2:** Update the howto with the operator runbook for the staged DAG + the DGX e2e gate.
- [ ] **Step 3:** Regen C4 with the pinned `dot` (`-graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe"`); verify the assemble ran (no 0-entity placeholder). Confirm the `tracking` aggregator count is unaffected.
- [ ] **Step 4:** Run `python -m pytest tests/test_agents_md_budget.py tests/test_howto_links_wired.py -v` (and any ADR/C4 wiring tests) → PASS.
- [ ] **Step 5:** No commit — this plan + spec + ADR + howto + C4 all travel with the single owner-gated harness-tune commit.

---

## Final gate (not a task — the handoff)

After all tasks are green locally (ruff + pyright scoped + the relevant unit tests) and the determinism tests pass:

1. **DGX e2e** — run `dgx_e2e.e2e` for every trainer against `~/tc3-kinematics` (+ SB360 for receiver). Record per-phase peak RSS + wall. Any phase over 32 GB BLOCKS pending an owner ruling (reduce-on-DGX vs an HF flavor exception).
2. **Freeze the tree** → hand to independent review (author does not run own review).
3. On review clear + **owner go** → the single harness-tune commit (docs travel with it).
4. Re-fit waves on cpu-upgrade (HF) or DGX per the owner's executor ruling → floor PASS every variant → **weights commit** (owner-gated) → merge PR #284 non-squash.
