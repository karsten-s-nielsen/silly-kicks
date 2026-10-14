# HF Jobs Training Harness + Re-fit Campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. (Subagent-driven execution is DISABLED for this repo by owner rule — see Global
> Constraints.) Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable harness that runs any silly-kicks trainer as an ephemeral Hugging Face Job,
then use it to re-fit 9 of the 10 trainers on HF (gk_completion was retired upstream in 4.135.0 / ADR-121;
gk_retention stays blocked) and PR the new weights back — without lowering the
shipped quality bar and with zero Databricks dependency.

**Architecture:** A local orchestration package `scripts/hf_jobs/` (registry → launch → stage → floor).
The launcher fans out one ephemeral HF Job per trainer; each job git-clones the repo at a pinned tag,
installs `.[train]`, runs the unchanged trainer against a job-local corpus, and uploads its weights to a
private HF dataset. The launcher retrieves them, applies a no-regression-vs-committed floor, and the
cleared weights are copied into `silly_kicks/**/weights/`. Owner-tier corpora are materialized on
DGX/local via `_loader_pining` and pushed once to the private dataset (credentials never reach HF).

**Tech Stack:** Python 3.10 local (repo `.venv`); `huggingface_hub` / `hf` CLI (jobs, datasets); HF Jobs
runs `python:3.12` containers; `uv` in-job installs; pandas/numpy/scikit-learn/xgboost/ruthless via
`.[train]`; pytest + mocked `hf` CLI for tests.

**Spec:** `docs/superpowers/specs/2026-10-13-hf-jobs-training-harness-design.md` (owner-ratified
2026-10-13; reviewers A+B APPROVE r2). The plan argues from the spec; executors read both.

## Global Constraints

Every task's requirements implicitly include this section.

- **Commits are owner-gated and minimal.** NO per-task commits, NO micro-commits. This plan lands in
  EXACTLY TWO commits, each immediately preceded by an explicit owner-approval gate where the exact diff
  / file list is shown and an explicit "yes" is awaited: **Commit 1** = harness + tests + ADR + howto +
  C4 regen (end of Phase 0, full suite green); **Commit 2** = all re-fit weights in one commit (end of
  Phase 3, full suite green, NOT split per model). Tasks before a gate accumulate uncommitted work.
- **Branch:** one feature branch `feat/hf-jobs-training-harness`; no worktrees. After the Commit-1 gate the
  branch is PUSHED to origin (owner-gated, with Commit 1) so the re-fit jobs can clone it for
  `scripts/hf_jobs/**` — Commit 1 is pushed but does NOT MERGE early, and P1 cannot start until the push
  lands. Single PR opened from the branch after Phase 3, carrying both commits.
- **Docs travel with their commit** — the spec + this plan ride Commit 1; never a separate/early doc
  commit.
- **ZERO Databricks anywhere.** Owner-tier data only via `scripts/_loader_pining`; public only via
  `scripts/_sb_open_data` / `statsbombpy`. `scripts/_loader_databricks.py` is never imported.
- **No-regression quality floor (head-to-head).** Re-fit on the SAME full owner corpus (never a
  subsample). In-job, score BOTH the currently-committed model (shipped in the cloned tag) and the fresh
  fit on the SAME held-out split, computing the primary metric identically for each; the fresh metric must
  not regress vs the committed metric beyond a pinned per-trainer δ. A regression beyond δ BLOCKS PR-back
  and is handed to the owner — never auto-accepted. This reads NO committed `metrics.json` (3 public
  trainers ship none).
- **gk_retention is BLOCKED** (marts-native / Databricks; no pining path). Registered in the harness,
  NOT re-fit.
- **Launch shape (HF CLI v1.7.1):** `hf jobs run <opts> python:3.12 -- bash -lc "<cmd>"`. The `--`
  end-of-options marker before `bash` is mandatory (else `-lc` is eaten); `--name` is NOT passed (it is
  mis-parsed as the image); secret passed as `--secrets HF_TOKEN=<token>`; **jobs clone the repo at the
  Commit-1 ref** — the pushed feature branch `feat/hf-jobs-training-harness`, whose tip is the harness
  commit during P1-P2 — passed by the launcher, NEVER the literal `v4.134.0` (which predates
  `scripts/hf_jobs/**` and would `ModuleNotFoundError` on the in-job head-to-head step). The local token
  is read via `huggingface_hub.get_token()` (not `hf auth token`, absent in v1.7.1).
- **Test dir:** `tests/scripts/hf_jobs/` must NOT contain `__init__.py` (it would shadow the top-level
  `scripts` namespace package).
- **Lint/type at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`;
  `pyright`. **Tests:** `python -m pytest tests/ -m "not e2e" -v --tb=short`.

---

## File Structure

- `scripts/hf_jobs/__init__.py` — package marker (scripts/ IS an importable package; hf_jobs is a
  subpackage, so this `__init__.py` is required — unlike the test dir).
- `scripts/hf_jobs/registry.py` — `TrainerSpec` + `VariantBuild` dataclasses, `REGISTRY` (9 trainable),
  `BLOCKED` (gk_retention), accessors. Carries the 4-mode `source` taxonomy
  `SOURCES = (public, owner_flat, owner_pergame, owner_sb360)` (expanded from the 2-mode `public|pining`
  sketch — see **As-built expansion** below). Single source of truth.
- `scripts/hf_jobs/floor.py` — the no-regression check (pure; compares two head-to-head scalars).
- `scripts/hf_jobs/headtohead.py` — per-trainer scorers (`SCORERS`) that, in-job, score BOTH the committed
  and the fresh model on the SAME held-out split, plus the platform/version stamp; has a CLI entry point.
- `scripts/hf_jobs/launch.py` — pure command builders (`build_inner_command`, `build_argv`) + thin
  orchestration (`launch_job`, `poll_until_terminal`, `retrieve_weights`) over the `hf` CLI.
- `scripts/hf_jobs/repackage.py` — (As-built, owner_pergame) in-job flat→per-game corpus transform for
  xshot/xcross (`{game}/frames.parquet`+`shots.parquet`); the kinematics shards carry no shot labels, so
  the flat `--data-dir` path yields "No usable training data" (DGX-verified necessary).
- `scripts/hf_jobs/stage.py` — owner-tier corpus materialization wrapper + private-dataset push +
  manifest (idempotent); `write_offline_manifest` snapshots the pining `/matches` response for offline use.
- `scripts/_loader_pining.py` — **(As-built, MODIFIED — shared module)** `_list_matches` gains an opt-in
  offline-manifest branch (`SILLY_KICKS_PINING_MANIFEST_DIR`); default off = unchanged behaviour. Enables
  receiver's air-gapped (no-token/no-network) enumeration from a staged corpus. Non-breaking (306 existing
  loader/corpus tests green). **Owner sign-off required (reviewer B HJT-IMPL-01).**
- `tests/scripts/hf_jobs/test_registry.py` — ADR-056 population gate.
- `tests/scripts/hf_jobs/test_floor.py` — δ gate blocks regression, admits improvement, fail-closed on NaN.
- `tests/scripts/hf_jobs/test_headtohead.py` — a feasibility test PER trainer (load committed model as both
  operands on a tiny synthetic eval → finite paired scalars) + the stamp + fail-loud tests.
- `tests/scripts/hf_jobs/test_launch.py` — argv/inner-command goldens (mocked `hf`) incl. the 4 source modes.
- `tests/scripts/hf_jobs/test_repackage.py` — per-game layout + filtered shots (As-built).
- `tests/scripts/hf_jobs/test_stage.py` — idempotence + manifest + `write_offline_manifest` (mocked upload).
- `tests/scripts/test_loader_offline_manifest.py` — the offline-manifest branch (As-built).
- `docs/superpowers/adrs/ADR-122-hf-jobs-training-harness.md` — the decision record (ADR-121 is taken by the upstream xT-GK v1 retirement).
- `docs/howto/hf-jobs-training.md` — operator runbook.
- `docs/c4/architecture.dsl` (modify) + `docs/c4/architecture.html` (regen, pinned `dot`) — new `hf_jobs` container.

> **As-built expansion (DGX-verified 2026-10-13; reviewer B HJT-IMPL-01 — owner sign-off).** Executing the
> nine trainers on the DGX showed the 2-mode `public|pining` sketch was insufficient, so Commit-1 carries:
> (1) the 4-mode `source` taxonomy; (2) `scripts/hf_jobs/repackage.py` (+test); (3) a MODIFIED shared
> `scripts/_loader_pining.py` offline-manifest branch (+test). All three are discovered-necessary, sound,
> non-breaking, and green — but they EXCEED the originally-approved plan, so they are called out here for the
> owner's explicit approval before Commit-1 lands.

---

## Phase 0 — Harness (code, docs; ends at the Commit 1 gate)

### Task 1: Trainer registry + ADR-056 population gate

**Files:**
- Create: `scripts/hf_jobs/__init__.py` (empty)
- Create: `scripts/hf_jobs/registry.py`
- Test: `tests/scripts/hf_jobs/test_registry.py`

**Interfaces:**
- Produces: `TrainerSpec` (frozen dataclass); `REGISTRY: dict[str, TrainerSpec]`;
  `BLOCKED: dict[str, TrainerSpec]`; `trainable_names() -> list[str]`; `blocked_names() -> list[str]`;
  `get(name: str) -> TrainerSpec`.

- [ ] **Step 1: Write the failing test** — `tests/scripts/hf_jobs/test_registry.py`

```python
import pathlib

import pytest

from scripts.hf_jobs import registry as reg

REPO = pathlib.Path(__file__).resolve().parents[3]


def _population() -> set[str]:
    # ADR-056: derive the population, assert it exactly. A trainer = scripts/train_<name>.py.
    return {p.stem[len("train_"):] for p in (REPO / "scripts").glob("train_*.py")}


def test_registry_covers_population_exactly():
    covered = set(reg.REGISTRY) | set(reg.BLOCKED)
    assert covered == _population(), covered.symmetric_difference(_population())


def test_blocked_is_exactly_gk_retention_and_nonempty():
    assert set(reg.BLOCKED) == {"gk_retention"}
    assert reg.BLOCKED["gk_retention"].databricks_blocked is True


def test_every_weights_path_has_a_loadable_committed_model():
    # Head-to-head LOADS the committed model from these dirs (not a metrics.json — 3 trainers ship none).
    for spec in reg.REGISTRY.values():
        for wp in spec.weights_paths:
            d = REPO / "silly_kicks" / wp
            assert d.is_dir(), f"{spec.name}: missing {d}"
            assert any((d / f).exists() for f in ("model.json", "rfcde_weights.npz", "metadata.json")), \
                f"{spec.name}: no loadable committed model artifact in {wp}"


def test_floor_config_is_sane_and_has_a_scorer():
    from scripts.hf_jobs.headtohead import SCORERS  # one registered head-to-head scorer per trainer
    for spec in reg.REGISTRY.values():
        assert spec.metric_direction in {"lower_better", "higher_better"}
        assert spec.delta > 0
        assert spec.name in SCORERS, f"{spec.name}: no head-to-head scorer registered"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.hf_jobs.registry`.

- [ ] **Step 3: Write minimal implementation** — `scripts/hf_jobs/registry.py`

```python
"""Single source of truth for how each trainer runs as an HF Job + its quality-floor config."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerSpec:
    name: str                       # "win_probability"
    script: str                     # "scripts/train_win_probability.py"
    source: str                     # "public" | "pining"
    flavor: str                     # HF Jobs flavor
    extras: tuple[str, ...]         # extra pip args beyond ".[train]" (e.g. ("statsbombpy",))
    timeout: str                    # HF --timeout, e.g. "30m" / "6h"
    run_args: tuple[str, ...]       # corpus-independent trainer CLI args
    weights_paths: tuple[str, ...]  # repo dirs (under silly_kicks/) holding committed weights+metrics.json
    primary_metric: str             # metrics.json key the floor compares
    metric_direction: str           # "lower_better" | "higher_better"
    delta: float                    # pinned per-trainer regression tolerance
    databricks_blocked: bool = False


# `primary_metric` is the metric name the head-to-head scorer (scripts/hf_jobs/headtohead.py) computes
# IDENTICALLY for both the committed and the fresh model on the same held-out split — NOT a key read from
# any committed metrics.json (3 public trainers ship none). `delta` is the pinned per-trainer regression
# tolerance on that metric; it is OWNER-RATIFIED and FAIL-CLOSED (a missing/NaN metric blocks). A noisier
# trainer needs a LARGER δ; initial values below are P1/P2-calibrated starting points, tightened by the
# owner after the first head-to-head deltas, never loosened silently.
REGISTRY: dict[str, TrainerSpec] = {
    "win_probability": TrainerSpec(
        "win_probability", "scripts/train_win_probability.py", "public", "cpu-upgrade",
        ("statsbombpy",), "45m", ("--all-competitions",),
        ("win_probability/weights",), "ece", "lower_better", 0.01),
    "pass_completion": TrainerSpec(
        "pass_completion", "scripts/train_pass_completion.py", "public", "cpu-upgrade",
        ("statsbombpy",), "45m", ("--source", "open-data"),
        ("expected_passing/weights",), "ece", "lower_better", 0.01),
    "match_outcome_dependence": TrainerSpec(
        "match_outcome_dependence", "scripts/train_match_outcome_dependence.py", "public", "cpu-upgrade",
        ("statsbombpy",), "45m", (),
        ("match_outcome/weights",), "log_loss", "lower_better", 0.01),
    "xsuccess": TrainerSpec(
        "xsuccess", "scripts/train_xsuccess.py", "public", "cpu-upgrade",
        ("statsbombpy",), "45m", (),
        ("xsuccess/weights",), "ece", "lower_better", 0.01),
    "ghost_gk": TrainerSpec(
        "ghost_gk", "scripts/train_ghost_gk.py", "pining", "cpu-performance",
        (), "6h", (),
        ("tracking/_ghost_gk_weights/default", "tracking/_ghost_gk_weights/sweeper",
         "tracking/_ghost_gk_weights/position_only", "tracking/_ghost_gk_weights/sweeper_position_only"),
        "cv_mae_euclidean", "lower_better", 0.05),  # metres; committed default ~1.13 m
    "ghost_outfield": TrainerSpec(
        "ghost_outfield", "scripts/train_ghost_outfield.py", "pining", "cpu-performance",
        (), "6h", (),
        ("tracking/_ghost_outfield_weights/default", "tracking/_ghost_outfield_weights/position_only"),
        "cv_mae", "lower_better", 0.10),  # metres; committed default ~6.0 m
    "xshot_occurrence": TrainerSpec(
        "xshot_occurrence", "scripts/train_xshot_occurrence.py", "pining", "cpu-xl",
        (), "4h", (),
        ("tracking/_xshot_weights/default", "tracking/_xshot_weights/position_only"),
        "pr_auc", "higher_better", 0.01),
    "xcross_attempt": TrainerSpec(
        "xcross_attempt", "scripts/train_xcross_attempt.py", "pining", "cpu-xl",
        (), "4h", (),
        ("tracking/_xcross_weights/default", "tracking/_xcross_weights/position_only"),
        "pr_auc", "higher_better", 0.01),
    "receiver_model": TrainerSpec(
        "receiver_model", "scripts/train_receiver_model.py", "pining", "cpu-xl",
        (), "4h", (),
        ("tracking/_receiver_weights/default",), "top1", "higher_better", 0.01),
}

BLOCKED: dict[str, TrainerSpec] = {
    "gk_retention": TrainerSpec(
        "gk_retention", "scripts/train_gk_retention.py", "pining", "cpu-xl",
        (), "4h", (), ("xtgk/_retention_weights/default",),
        "ece", "lower_better", 0.01, databricks_blocked=True),
}


def trainable_names() -> list[str]:
    return sorted(REGISTRY)


def blocked_names() -> list[str]:
    return sorted(BLOCKED)


def get(name: str) -> TrainerSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    return BLOCKED[name]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_registry.py -v`
Expected: PASS. (`test_floor_config_is_sane_and_has_a_scorer` depends on Task 2b's `headtohead.SCORERS`;
if Task 2b is not yet written, this test fails on import until then — expected during P0 build order.) The
registry `primary_metric` names were set from the REAL committed artifacts (verified 2026-10-13:
win_probability `ece`, ghost_gk `cv_mae_euclidean`, ghost_outfield `cv_mae`, xshot/xcross nested `pr_auc`,
receiver `top1`; match_outcome `log_loss`, xsuccess/pass_completion `ece` —
the last three ship NO committed metric, so the head-to-head scorer computes them for both models). A
`primary_metric` here is a name the scorer MUST emit for both models, not a key read from any file.

### Task 2: No-regression-vs-committed floor

**Files:**
- Create: `scripts/hf_jobs/floor.py`
- Test: `tests/scripts/hf_jobs/test_floor.py`

**Interfaces:**
- Consumes: the head-to-head run metrics (Task 2b) which carry `committed_<metric>` + `fresh_<metric>`.
- Produces: `regression_check(committed_value: float, fresh_value: float, *, direction: str, delta: float)
  -> tuple[bool, dict]` — compares two scalars; `ok=False` means a regression beyond δ (BLOCK); NaN either
  side is fail-closed (BLOCK). `read_pair(run_metrics: dict, metric: str) -> tuple[float, float]` — pulls
  `(committed_<metric>, fresh_<metric>)` (KeyError if absent → BLOCK). `evaluate_spec(spec, run_metrics:
  dict) -> tuple[bool, dict]` — applies a `TrainerSpec`'s floor config to a run's metrics.

- [ ] **Step 1: Write the failing test** — `tests/scripts/hf_jobs/test_floor.py`

```python
import pytest

from scripts.hf_jobs.floor import evaluate_spec, read_pair, regression_check


def test_lower_better_regression_beyond_delta_blocks():
    ok, d = regression_check(0.02, 0.05, direction="lower_better", delta=0.01)
    assert ok is False and d["regressed"] is True


def test_lower_better_within_delta_admits():
    ok, _ = regression_check(0.02, 0.025, direction="lower_better", delta=0.01)
    assert ok is True


def test_lower_better_improvement_admits():
    ok, _ = regression_check(0.02, 0.01, direction="lower_better", delta=0.01)
    assert ok is True


def test_higher_better_regression_beyond_delta_blocks():
    ok, _ = regression_check(0.80, 0.76, direction="higher_better", delta=0.01)
    assert ok is False


def test_higher_better_improvement_admits():
    ok, _ = regression_check(0.80, 0.83, direction="higher_better", delta=0.01)
    assert ok is True


def test_nan_is_fail_closed():
    ok, _ = regression_check(0.80, float("nan"), direction="higher_better", delta=0.01)
    assert ok is False


def test_read_pair_and_evaluate_spec():
    from scripts.hf_jobs import registry as reg
    spec = reg.REGISTRY["win_probability"]  # primary_metric="ece", lower_better
    run = {"committed_ece": 0.02, "fresh_ece": 0.05}
    assert read_pair(run, "ece") == (0.02, 0.05)
    ok, _ = evaluate_spec(spec, run)
    assert ok is False


def test_missing_pair_key_blocks():
    from scripts.hf_jobs import registry as reg
    spec = reg.REGISTRY["win_probability"]
    with pytest.raises(KeyError):
        evaluate_spec(spec, {"fresh_ece": 0.02})  # committed_ece absent → cannot silently pass


def test_direction_polarity_per_trainer():
    # Pins each registry metric_direction: a worsened fresh value must BLOCK, an improved one must ADMIT.
    # The same-model feasibility test (committed == fresh) cannot catch an inverted direction; this does.
    from scripts.hf_jobs import registry as reg
    for spec in reg.REGISTRY.values():
        base, m, d = 0.5, spec.primary_metric, spec.delta
        worse = base + 2 * d if spec.metric_direction == "lower_better" else base - 2 * d
        better = base - 2 * d if spec.metric_direction == "lower_better" else base + 2 * d
        ok_worse, _ = evaluate_spec(spec, {f"committed_{m}": base, f"fresh_{m}": worse})
        ok_better, _ = evaluate_spec(spec, {f"committed_{m}": base, f"fresh_{m}": better})
        assert ok_worse is False and ok_better is True, f"{spec.name}: direction polarity wrong"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_floor.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.hf_jobs.floor`.

- [ ] **Step 3: Write minimal implementation** — `scripts/hf_jobs/floor.py`

```python
"""No-regression quality floor (pure). Compares the head-to-head committed vs fresh scalar; the bulk
re-fit must not silently lower the bar. Fail-closed: a missing or NaN value BLOCKS, never silently passes.
"""
from __future__ import annotations

import math


def regression_check(
    committed_value: float, fresh_value: float, *, direction: str, delta: float
) -> tuple[bool, dict]:
    if math.isnan(committed_value) or math.isnan(fresh_value):
        return (False, {"regressed": True, "reason": "nan", "committed": committed_value,
                        "fresh": fresh_value, "delta": delta, "direction": direction})
    if direction == "lower_better":
        regressed = fresh_value > committed_value + delta
    elif direction == "higher_better":
        regressed = fresh_value < committed_value - delta
    else:
        raise ValueError(f"unknown direction {direction!r}")
    detail = {"committed": committed_value, "fresh": fresh_value, "delta": delta,
              "direction": direction, "regressed": regressed}
    return (not regressed, detail)


def read_pair(run_metrics: dict, metric: str) -> tuple[float, float]:
    # KeyError is intended: an absent head-to-head pair is a hard failure, never a pass.
    return float(run_metrics[f"committed_{metric}"]), float(run_metrics[f"fresh_{metric}"])


def evaluate_spec(spec, run_metrics: dict) -> tuple[bool, dict]:
    committed_value, fresh_value = read_pair(run_metrics, spec.primary_metric)
    return regression_check(committed_value, fresh_value,
                            direction=spec.metric_direction, delta=spec.delta)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_floor.py -v`
Expected: PASS.

### Task 2b: Head-to-head scorers + in-job stamp

> **AS-BUILT (2026-10-13) — authoritative; the sketch below is the original design intent.** The final
> `scripts/hf_jobs/headtohead.py`: each `_score_*` takes an INJECTED eval **bundle** (not `corpus_dir`) and
> is pure; `build_eval(name, corpus_dir)` assembles the REAL eval per model and is **fail-loud**
> (`EvalDataUnavailableError`, never synthetic): owner trainers read the staged shards + the library/trainer
> extractor (`prepare_ghost_gk/xshot/xcross_training_data`, `extract_match`, `extract_candidate_rows`);
> public trainers go through `scripts._sb_open_data` + the trainer's helpers (`match_shard`,
> `compute_win_probability`, per-match xG→pmf). `synthetic_bundle` is TEST-ONLY (feasibility + the CLI e2e).
> **All 9 builders were implemented and EXECUTE-VERIFIED on the DGX before Commit 1** (fresh venv +
> `~/tc3-kinematics` + statsbombpy), committed==fresh on real data: ghost_gk 1.355m · ghost_outfield
> 6.046m (committed full-corpus 6.0m) · xshot pr_auc 0.022 · xcross 0.034 · receiver top1 0.351 ·
> win_probability ece 0.139 · pass_completion 0.042 · xsuccess 0.027 · match_outcome log_loss 0.399. So the
> floor scores REAL data for every trainer — not gated to P1/P2.

This is the mechanism that makes the floor bite WITHOUT a committed `metrics.json` (3 public trainers ship
none) and without the old-corpus confound: in the job, load BOTH the currently-committed model (shipped in
the cloned tag under `silly_kicks/<weights_path>/`) and the freshly-fit model (at `/tmp/out/weights`),
score each on the SAME held-out split, and emit `committed_<metric>` + `fresh_<metric>` into the run's
`metrics.json`. The held-out split is one GroupKFold-by-match fold of the same corpus the re-fit used;
scoring the committed model on data it may have partly seen biases toward it, which makes the guard
STRICTER (fail-safe). The scorer also stamps the container platform + library versions (resolves
HF-PLAN-01: tag-independent, in-job, zero trainer edits).

**Files:**
- Create: `scripts/hf_jobs/headtohead.py`
- Test: `tests/scripts/hf_jobs/test_headtohead.py`

**Interfaces:**
- Consumes: `TrainerSpec` (Task 1); each trainer's model class `.load()` + predict surface.
- Produces: `SCORERS: dict[str, Callable[[Path, Path, Path | None], dict[str, float]]]` (one per trainer,
  keyed by `TrainerSpec.name`); `stamp() -> dict` (platform + sklearn/xgboost/numpy/numba versions);
  `main()` CLI: `python -m scripts.hf_jobs.headtohead <trainer> --committed-dir D --fresh-dir D
  --corpus-dir D --metrics-json PATH` — runs the trainer's scorer, merges `committed_<m>`/`fresh_<m>` +
  the stamp into the existing `metrics.json`. Metric helpers: `ece`, `pr_auc`, `brier`, `log_loss`, `top1`,
  `mae_euclidean`, `mae`.

- [ ] **Step 1: Write the per-trainer feasibility tests** — `tests/scripts/hf_jobs/test_headtohead.py`

Each test loads the COMMITTED model as BOTH operands on a tiny synthetic eval the model accepts, and
asserts the scorer returns finite, equal `committed_<m>`/`fresh_<m>` (equal because same model) — proving
the load+predict+metric path is feasible for every trainer at BUILD time, not discovered during the P1/P2
spend. One parametrized test over all `SCORERS`:

```python
import math

import pytest

from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.headtohead import SCORERS, stamp


@pytest.mark.parametrize("name", sorted(reg.REGISTRY))
def test_every_trainer_has_a_feasible_scorer(name, synthetic_eval_for):
    # synthetic_eval_for is a fixture (in conftest or the test) returning (committed_dir, corpus_dir)
    # for `name`: the real committed weights dir + a tiny synthetic corpus the model's predict accepts.
    committed_dir, corpus_dir = synthetic_eval_for(name)
    out = SCORERS[name](committed_dir, committed_dir, corpus_dir)  # same model as both operands
    spec = reg.REGISTRY[name]
    assert f"committed_{spec.primary_metric}" in out and f"fresh_{spec.primary_metric}" in out
    c, f = out[f"committed_{spec.primary_metric}"], out[f"fresh_{spec.primary_metric}"]
    assert math.isfinite(c) and math.isfinite(f) and c == pytest.approx(f)


def test_stamp_has_platform_and_versions():
    s = stamp()
    assert "training_platform" in s and "sklearn_version" in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_headtohead.py -v`
Expected: FAIL — `scripts.hf_jobs.headtohead` undefined.

- [ ] **Step 3: Implement `headtohead.py`** — the framework + metric helpers + the stamp + a fully worked
  `win_probability` scorer. Each remaining scorer follows the SAME shape (load both models, build the
  held-out fold, compute the trainer's metric for each); the per-trainer feasibility test gates it.

```python
"""Head-to-head scoring: committed vs fresh model on the same held-out split, emitted as a pair."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from scripts.hf_jobs.registry import REGISTRY


def stamp() -> dict:
    def _v(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    return {
        "training_platform": f"{platform.system()}-{platform.machine()}",
        "python_version": sys.version.split()[0],
        "sklearn_version": _v("scikit-learn"),
        "xgboost_version": _v("xgboost"),
        "numpy_version": _v("numpy"),
        "numba_version": _v("numba"),
    }


# --- metric helpers (lower/higher per the registry direction) --------------------------------------
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss_(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)))


def top1(correct: np.ndarray) -> float:
    return float(np.mean(correct))


def mae_euclidean(pred_xy: np.ndarray, true_xy: np.ndarray) -> float:
    return float(np.mean(np.hypot(pred_xy[:, 0] - true_xy[:, 0], pred_xy[:, 1] - true_xy[:, 1])))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


# --- per-trainer scorers (one per registry name) ---------------------------------------------------
def _heldout_fold(game_ids: np.ndarray, fold: int = 0, n_splits: int = 5):
    """Index mask of ONE GroupKFold-by-match fold (deterministic), used as the shared eval split."""
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(game_ids))
    splits = list(GroupKFold(n_splits=min(n_splits, len(np.unique(game_ids)))).split(idx, groups=game_ids))
    _, test = splits[fold]
    return test


def _score_win_probability(committed_dir: Path, fresh_dir: Path, corpus_dir: Path | None) -> dict:
    """WORKED EXAMPLE. Load both WinProbabilityModels, score ece on one held-out match fold."""
    from silly_kicks.win_probability import WinProbabilityModel
    X, y, games = _load_win_prob_eval(corpus_dir)          # eval frame + label + game_id (helper below)
    te = _heldout_fold(games)
    out = {}
    for tag, d in (("committed", committed_dir), ("fresh", fresh_dir)):
        model = WinProbabilityModel.load(d)
        p = np.asarray(model.predict_proba(X.iloc[te]))     # confirm the exact predict surface at impl
        out[f"{tag}_ece"] = ece(np.asarray(y)[te], p)
    return out


SCORERS: dict[str, Callable[[Path, Path, Path | None], dict[str, float]]] = {
    "win_probability": _score_win_probability,
    # The remaining 9 are implemented in Step 4 following the same shape, keyed by name:
    #   pass_completion -> ece (expected_passing.PassCompletionModel)
    #   match_outcome_dependence -> log_loss (match_outcome joint)
    #   xsuccess -> ece (xsuccess.XSuccessModel)
    #   ghost_gk -> mae_euclidean (tracking GhostGk; predicted vs true keeper xy)
    #   ghost_outfield -> mae (tracking GhostOutfield; cv_mae surface)
    #   xshot_occurrence -> pr_auc (XShotOccurrenceModel)
    #   xcross_attempt -> pr_auc (XCrossAttemptModel)
    #   receiver_model -> top1 (ReceiverModel; argmax candidate == actual)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trainer")
    ap.add_argument("--committed-dir", type=Path, required=True)
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--metrics-json", type=Path, required=True)
    a = ap.parse_args()
    paired = SCORERS[a.trainer](a.committed_dir, a.fresh_dir, a.corpus_dir)
    existing = json.loads(a.metrics_json.read_text()) if a.metrics_json.exists() else {}
    existing.update(paired)
    existing.update(stamp())
    a.metrics_json.write_text(json.dumps(existing, indent=2))
    print("HF_JOBS_HEADTOHEAD_OK " + json.dumps(paired))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the remaining 9 scorers + their feasibility tests.** For each trainer, load its
  model class (named in the `SCORERS` comment above; all ship `.load()` — confirm the exact predict
  surface against the class at impl), build the shared held-out fold, compute the listed metric for the
  committed and the fresh model, return the pair. Add the trainer to the Step-1 parametrized test (it is
  already parametrized over `reg.REGISTRY`, so each new `SCORERS` entry is covered once its synthetic-eval
  fixture branch exists). The `_load_*_eval` helpers assemble a small eval frame from `--corpus-dir`
  (owner-tier) or an in-job statsbombpy fetch (public); keep them in `headtohead.py`.

- [ ] **Step 5: Run to verify all pass**

Run: `python -m pytest tests/scripts/hf_jobs/test_headtohead.py -v`
Expected: PASS for all 9 trainers (each scorer loads the committed model and returns a finite paired
scalar). A trainer whose model API does not expose the needed predict surface is a feasibility failure to
surface to the owner NOW (P0), never at P1/P2 spend time.

### Task 3: Launch command builders (pure, golden-tested)

> **AS-BUILT (2026-10-13) — authoritative; the sketch below is the original design intent.** The nine
> trainers' CLIs were DGX-verified before Commit 1 and are heterogeneous, so the builders were written to
> what they actually accept (the generic `--data-dir /tmp/corpus` sketch below is NOT what shipped):
> - Trainers run as **`python -m scripts.train_<name>`** — the file form raises `ModuleNotFoundError:
>   No module named 'scripts'` (no `scripts/__init__.py`).
> - `registry.py` carries a `VariantBuild` per committed variant + a `source` mode
>   (`public`/`owner_flat`/`owner_pergame`/`owner_sb360`); `build_inner_command(spec, *, tag, run_key,
>   dataset_repo, corpus_repo=None, sb360_repo=None)` emits, per mode: the correct output flag (`--out`
>   vs `--output-dir`), data inputs (`--data-dir <gen dir>`+`--actions-dir`+`--home-teams` for owner_flat;
>   an in-job `scripts.hf_jobs.repackage` + `--data-dir /tmp/pergame` for owner_pergame; `--cache-dir`
>   + `SILLY_KICKS_PINING_MANIFEST_DIR` offline for owner_sb360), a run per variant, and the head-to-head
>   on the default variant with the per-trainer `--fresh-dir`.
> - One shared-library change: `_loader_pining._list_matches` offline-manifest branch (receiver enabler).
> All nine launch invocations + the repackage + the offline receiver path were EXECUTE-VERIFIED on the DGX.

**Files:**
- Create: `scripts/hf_jobs/launch.py` (builders only this task)
- Create: `scripts/hf_jobs/repackage.py` (owner_pergame flat→per-game, in-job)
- Modify: `scripts/_loader_pining.py` (offline-manifest branch)
- Test: `tests/scripts/hf_jobs/test_launch.py`, `test_repackage.py`, `tests/scripts/test_loader_offline_manifest.py`

**Interfaces:**
- Consumes: `TrainerSpec` (Task 1).
- Produces: `REPO_URL: str`; `build_inner_command(spec, *, tag, run_key, dataset_repo, corpus_repo=None)
  -> str`; `build_argv(spec, inner: str, *, token: str) -> list[str]`.

- [ ] **Step 1: Write the failing test** — `tests/scripts/hf_jobs/test_launch.py`

```python
from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import build_argv, build_inner_command


def test_argv_shape_encodes_the_v1_7_1_quirks():
    spec = reg.REGISTRY["win_probability"]
    inner = build_inner_command(spec, tag="feat/hf-jobs-training-harness", run_key="r1", dataset_repo="ns/artifacts")
    argv = build_argv(spec, inner, token="hf_FAKE")
    assert argv[:3] == ["hf", "jobs", "run"]
    assert "--name" not in argv                      # mis-parsed as image in v1.7.1
    assert "--flavor" in argv and argv[argv.index("--flavor") + 1] == spec.flavor
    assert "--secrets" in argv and argv[argv.index("--secrets") + 1] == "HF_TOKEN=hf_FAKE"
    # the end-of-options marker MUST sit immediately before `bash` or `-lc` is eaten
    assert argv[-4:] == ["--", "bash", "-lc", inner]
    assert argv[argv.index("--") - 1] == "python:3.12"


def test_public_inner_clones_tag_installs_train_runs_and_uploads():
    spec = reg.REGISTRY["win_probability"]
    inner = build_inner_command(spec, tag="relref", run_key="r1", dataset_repo="ns/artifacts")
    assert "git clone --depth 1 --branch relref" in inner  # asserts the PASSED ref, never a literal
    assert "uv pip install --system -q -e '.[train]' statsbombpy" in inner
    assert "python scripts/train_win_probability.py --all-competitions --out /tmp/out" in inner
    assert "ns/artifacts" in inner and "r1/win_probability" in inner
    assert "python -m scripts.hf_jobs.headtohead win_probability" in inner
    assert "--committed-dir silly_kicks/win_probability/weights" in inner
    assert "hf download" not in inner                # public trainer needs no staged corpus


def test_owner_inner_downloads_corpus_first():
    spec = reg.REGISTRY["ghost_gk"]
    inner = build_inner_command(
        spec, tag="relref", run_key="r1", dataset_repo="ns/artifacts", corpus_repo="ns/tc3")
    assert "ns/tc3" in inner and "--data-dir" in inner
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation** — `scripts/hf_jobs/launch.py` (builders)

```python
"""Launch silly-kicks trainers as ephemeral HF Jobs. Pure builders here; orchestration in Task 4."""
from __future__ import annotations

import shlex

from scripts.hf_jobs.registry import TrainerSpec

REPO_URL = "https://github.com/karsten-s-nielsen/silly-kicks"


def _upload_py(dataset_repo: str, run_key: str, name: str) -> str:
    # Uploads the whole run dir (/tmp/out = weights/ + the head-to-head metrics.json). HF_TOKEN from --secrets.
    path_in_repo = f"{run_key}/{name}"
    return (
        'python -c "from huggingface_hub import HfApi; '
        f"HfApi().upload_folder(folder_path='/tmp/out', repo_id='{dataset_repo}', "
        f"repo_type='dataset', path_in_repo='{path_in_repo}'); print('HF_JOBS_UPLOAD_OK')\""
    )


def build_inner_command(
    spec: TrainerSpec, *, tag: str, run_key: str, dataset_repo: str, corpus_repo: str | None = None
) -> str:
    parts = [
        "set -e",
        "pip install -q uv",
        f"git clone --depth 1 --branch {tag} {REPO_URL} /work",
        "cd /work",
        "uv pip install --system -q -e '.[train]'" + ("".join(f" {e}" for e in spec.extras)),
    ]
    data_args = ""
    if spec.source == "pining":
        if corpus_repo is None:
            raise ValueError(f"{spec.name}: owner-tier trainer needs corpus_repo")
        parts.append(f"hf download {corpus_repo} --repo-type dataset --local-dir /tmp/corpus")
        data_args = " --data-dir /tmp/corpus"
    run_args = " ".join(spec.run_args)
    script_cmd = f"python {spec.script} {run_args}{data_args} --out /tmp/out".replace("  ", " ").strip()
    parts.append(script_cmd)
    # Head-to-head: score the committed model (in the clone) AND the fresh fit on the same held-out split,
    # merge the committed_<m>/fresh_<m> pair + the platform stamp into /tmp/out/metrics.json, THEN upload.
    committed_dir = f"silly_kicks/{spec.weights_paths[0]}"  # default variant; scorer loops variants at impl
    corpus_arg = " --corpus-dir /tmp/corpus" if spec.source == "pining" else ""
    parts.append(
        f"python -m scripts.hf_jobs.headtohead {spec.name} --committed-dir {committed_dir} "
        f"--fresh-dir /tmp/out/weights --metrics-json /tmp/out/metrics.json{corpus_arg}"
    )
    parts.append(_upload_py(dataset_repo, run_key, spec.name))
    return " && ".join(parts)


def build_argv(spec: TrainerSpec, inner: str, *, token: str) -> list[str]:
    return [
        "hf", "jobs", "run",
        "--flavor", spec.flavor,
        "--secrets", f"HF_TOKEN={token}",
        "--timeout", spec.timeout,
        "--detach",
        "python:3.12",
        "--", "bash", "-lc", inner,
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch.py -v`
Expected: PASS. (If an owner trainer's real `--data-dir` flag differs — e.g. `xshot`/`xcross` use
`--providers` not `--data-dir` — encode that per-spec in `run_args`/builder and extend the test; the
builder must emit each trainer's actual corpus flag. Confirm against each `scripts/train_*.py` argparse.)

### Task 4: Launch orchestration (thin `hf` CLI wrappers, mocked)

**Files:**
- Modify: `scripts/hf_jobs/launch.py` (add orchestration)
- Test: `tests/scripts/hf_jobs/test_launch.py` (add orchestration tests)

**Interfaces:**
- Consumes: `build_argv`/`build_inner_command` (Task 3); `huggingface_hub.get_token`.
- Produces: `launch_job(spec, *, tag, run_key, dataset_repo, corpus_repo=None, token, runner=subprocess_runner)
  -> str` (returns job id); `poll_until_terminal(job_id, *, runner, sleep=...) -> str` (terminal stage);
  `retrieve_weights(dataset_repo, run_key, name, dest, *, downloader) -> pathlib.Path`.
  All side-effecting calls go through injected `runner`/`downloader` callables so tests never touch HF.

- [ ] **Step 1: Write the failing test** (append to `test_launch.py`)

```python
def test_launch_job_passes_built_argv_to_runner_and_returns_id():
    spec = reg.REGISTRY["win_probability"]
    calls = {}

    def fake_runner(argv):
        calls["argv"] = argv
        return "Job started with ID: 6abc123\nView at: ...", 0

    from scripts.hf_jobs.launch import launch_job
    jid = launch_job(spec, tag="feat/hf-jobs-training-harness", run_key="r1", dataset_repo="ns/art",
                     token="hf_FAKE", runner=fake_runner)
    assert jid == "6abc123"
    assert calls["argv"][:3] == ["hf", "jobs", "run"] and calls["argv"][-4:-1] == ["--", "bash", "-lc"]


def test_poll_returns_terminal_stage():
    seq = iter(["RUNNING", "RUNNING", "COMPLETED"])

    def fake_runner(argv):
        return next(seq), 0

    from scripts.hf_jobs.launch import poll_until_terminal
    assert poll_until_terminal("6abc123", runner=fake_runner, sleep=lambda _: None) == "COMPLETED"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch.py -v`
Expected: FAIL — `launch_job` / `poll_until_terminal` undefined.

- [ ] **Step 3: Write minimal implementation** (append to `launch.py`)

```python
import re
import subprocess
import time


def subprocess_runner(argv: list[str]) -> tuple[str, int]:
    p = subprocess.run(argv, capture_output=True, text=True)
    return (p.stdout + p.stderr), p.returncode


def launch_job(spec, *, tag, run_key, dataset_repo, corpus_repo=None, token, runner=subprocess_runner) -> str:
    inner = build_inner_command(spec, tag=tag, run_key=run_key, dataset_repo=dataset_repo, corpus_repo=corpus_repo)
    out, code = runner(build_argv(spec, inner, token=token))
    if code != 0:
        raise RuntimeError(f"{spec.name}: hf jobs run failed: {out}")
    m = re.search(r"ID:\s*([0-9a-f]{6,})", out)
    if not m:
        raise RuntimeError(f"{spec.name}: no job id in: {out}")
    return m.group(1)


_TERMINAL = {"COMPLETED", "ERROR", "CANCELED"}


def poll_until_terminal(job_id, *, runner=subprocess_runner, sleep=time.sleep, interval=20, max_polls=2000) -> str:
    for _ in range(max_polls):
        out, _ = runner(["hf", "jobs", "inspect", job_id, "--format", "json"])
        stage = next((s for s in _TERMINAL if s in out), None) or ("RUNNING" if "RUNNING" in out else "SCHEDULING")
        if stage in _TERMINAL:
            return stage
        sleep(interval)
    raise TimeoutError(f"{job_id}: never reached terminal")


def retrieve_weights(dataset_repo, run_key, name, dest, *, downloader):
    downloader(dataset_repo, f"{run_key}/{name}", dest)
    import pathlib
    return pathlib.Path(dest)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch.py -v`
Expected: PASS.

### Task 5: Owner-tier staging + manifest (idempotent)

**Files:**
- Create: `scripts/hf_jobs/stage.py`
- Test: `tests/scripts/hf_jobs/test_stage.py`

**Interfaces:**
- Produces: `corpus_manifest(corpus_dir: pathlib.Path) -> dict` (content hash + file count + bytes);
  `stage_corpus(corpus_dir, dataset_repo, *, uploader, already: dict | None = None) -> dict` — pushes the
  corpus to the private dataset unless `already` matches the current manifest (idempotent); returns the
  manifest. Upload goes through an injected `uploader` so tests never touch HF.

- [ ] **Step 1: Write the failing test** — `tests/scripts/hf_jobs/test_stage.py`

```python
import pathlib

from scripts.hf_jobs.stage import corpus_manifest, stage_corpus


def test_manifest_is_content_stable(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"xyz")
    m1 = corpus_manifest(tmp_path)
    m2 = corpus_manifest(tmp_path)
    assert m1 == m2 and m1["n_files"] == 1 and m1["bytes"] == 3


def test_stage_skips_when_manifest_unchanged(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"xyz")
    uploads = []
    m = stage_corpus(tmp_path, "ns/tc3", uploader=lambda *a, **k: uploads.append(a))
    assert len(uploads) == 1
    # second call with the prior manifest => no re-upload
    stage_corpus(tmp_path, "ns/tc3", uploader=lambda *a, **k: uploads.append(a), already=m)
    assert len(uploads) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/hf_jobs/test_stage.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation** — `scripts/hf_jobs/stage.py`

```python
"""Push an owner-tier corpus (materialized locally via _loader_pining) to a private HF dataset once.

Materialization itself is the existing driver (scripts/materialize_tc3_frames.py for the 4 TC3 trainers;
the pining_source path for receiver_model), run on DGX/local where the pining data lives. This
module only stages the OUTPUT and records a manifest so a repeat run is a no-op. ZERO Databricks.
"""
from __future__ import annotations

import hashlib
import pathlib


def corpus_manifest(corpus_dir: pathlib.Path) -> dict:
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())
    h = hashlib.sha256()
    nbytes = 0
    for p in files:
        h.update(p.relative_to(corpus_dir).as_posix().encode())
        data = p.read_bytes()
        h.update(data)
        nbytes += len(data)
    return {"sha256": h.hexdigest(), "n_files": len(files), "bytes": nbytes}


def stage_corpus(corpus_dir, dataset_repo, *, uploader, already: dict | None = None) -> dict:
    corpus_dir = pathlib.Path(corpus_dir)
    manifest = corpus_manifest(corpus_dir)
    if already is not None and already.get("sha256") == manifest["sha256"]:
        return manifest  # idempotent: nothing changed, skip the push
    uploader(str(corpus_dir), dataset_repo)
    return manifest
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/scripts/hf_jobs/test_stage.py -v`
Expected: PASS.

### Task 6: ADR-122, howto runbook, C4 regen, suite + lint green

**Files:**
- Create: `docs/superpowers/adrs/ADR-122-hf-jobs-training-harness.md`
- Create: `docs/howto/hf-jobs-training.md`
- Modify: `docs/c4/architecture.dsl` (add `hf_jobs` container under the appropriate system)
- Regen: `docs/c4/architecture.html`

- [ ] **Step 1: Write `ADR-122-hf-jobs-training-harness.md`** — context (DGX bottleneck + reproducibility;
  all trainers CPU-only), decision (ephemeral HF Jobs not Spaces; private-dataset staging; `_loader_pining`
  only, zero Databricks; retrain + no-regression-vs-committed floor; registry as contract with ADR-056
  population gate; gk_retention blocked/registered), consequences (new weights → downstream re-mat owed,
  owner-coordinated; HF py3.12 fixes the DGX py3.10 sklearn footgun). Cross-reference the spec.

- [ ] **Step 2: Write `docs/howto/hf-jobs-training.md`** — operator runbook: prerequisites (`hf` login,
  `huggingface_hub.get_token()`); stage (`stage.py`); launch (`launch.py`, the `-- bash -lc` + no-`--name`
  + `--secrets HF_TOKEN` shape, clone tag); retrieve + apply floor; PR-back; cost note (CPU flavors, full
  re-fit ≪ $50); the chirality P1 re-confirm step.

- [ ] **Step 3: Add the `hf_jobs` container to `docs/c4/architecture.dsl`** following the existing
  container style, then regenerate per the pinned pipeline:

Run: `structurizr.war export -format c4plantuml` → `c4_assemble.py --inject-wrap-width` →
`plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg *.puml` →
`c4_assemble.py --svg-dir` (the mad-scientist-skills:c4 skill drives this; a clean assemble proves `dot`
ran — a 0-entity placeholder aborts).

- [ ] **Step 4: Full lint + type + suite green**

Run: `python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/ && pyright`
Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all green. Fix anything red before the gate.

- [ ] **Step 5: COMMIT 1 GATE — owner approval required (do NOT commit without it).**
  Show the owner the exact file list + diff for: `scripts/hf_jobs/**`, `tests/scripts/hf_jobs/**`,
  `docs/superpowers/adrs/ADR-122-*.md`, `docs/howto/hf-jobs-training.md`, `docs/c4/architecture.{dsl,html}`,
  and the ride-along docs (this plan + the spec). Await an explicit "yes". Only then:

```bash
git add scripts/hf_jobs tests/scripts/hf_jobs docs/superpowers/adrs/ADR-122-hf-jobs-training-harness.md \
  docs/howto/hf-jobs-training.md docs/c4/architecture.dsl docs/c4/architecture.html \
  docs/superpowers/specs/2026-10-13-hf-jobs-training-harness-design.md \
  docs/superpowers/plans/2026-10-13-hf-jobs-training-harness.md
git commit  # message: feat(hf_jobs): ephemeral HF Jobs training harness (ADR-122)
git push -u origin feat/hf-jobs-training-harness   # owner-gated WITH Commit 1; P1 jobs clone this ref
```
(Commit message ends with the `Co-Authored-By` trailer from the session reminder; NO `Claude-Session:`
trailer.) The push is part of the Commit-1 approval — **P1 cannot start until it lands**, because the
re-fit jobs clone this pushed ref to get `scripts/hf_jobs/**`. No MERGE happens here.

---

## Phase 1 — Validate on public trainers (operational; owner-gated spend)

> No code deliverable; this phase RUNS the harness. Each HF launch spends money → the owner authorizes
> the P1 run before Step 1. Uses the public corpus, no owner-tier staging.

### Task 7: Re-fit the 4 public trainers + delta table + floor + chirality re-confirm

- [ ] **Step 1:** Owner authorizes the P1 spend (4 `cpu-upgrade` jobs, full public corpus). Preflight: the
  in-job `git clone` passes no GitHub auth, so confirm the GitHub repo is still PUBLIC (verified public
  2026-10-13 via the GitHub API) — a private flip fails the clone before any fit.
- [ ] **Step 2:** For each of `win_probability`, `pass_completion`, `match_outcome_dependence`,
  `xsuccess`: `launch_job(... tag="feat/hf-jobs-training-harness" ...)` (the pushed Commit-1 ref — carries
  `scripts/hf_jobs/**`), `poll_until_terminal`, `retrieve_weights`.
- [ ] **Step 3: Chirality re-confirmation (first HF artifact).** Load a retrieved artifact in-repo and
  confirm `verify_chirality` + `SHA256SUMS` + `_feature_contract` pass (spec says they will — save-x86 /
  load-x86 is same-arch). If any fails, STOP and report (it would be a genuine P0 surprise).
- [ ] **Step 4: Delta table + floor.** For each trainer read the retrieved run's `metrics.json` (it carries
  the in-job head-to-head `committed_<m>`/`fresh_<m>` pair + the platform stamp) and compute
  `floor.evaluate_spec(spec, run_metrics)`. Produce a head-to-head table (metric, committed, fresh, δ,
  verdict); one row per variant for multi-variant trainers.
- [ ] **Step 5: Report to owner.** Any `ok=False` (regression beyond δ) BLOCKS that trainer — hand the
  delta to the owner to investigate; do NOT copy its weights in. Cleared trainers' weights are held for
  Phase 3. No commit in this phase.

---

## Phase 2 — Owner-tier staging + re-fit (operational; owner-gated spend)

### Task 8: Stage the full owner corpus

- [ ] **Step 1:** On DGX/local (where pining data lives), materialize the FULL owner corpus — never a
  subsample — via the existing drivers (`scripts/materialize_tc3_frames.py` for the 4 TC3 trainers; the
  `pining_source` path for `receiver_model`). ZERO Databricks.
- [ ] **Step 2:** `stage_corpus(corpus_dir, "<private-dataset>", uploader=..., already=<prior manifest>)`
  to push each corpus to its private HF dataset; record the manifest for provenance/idempotence.

### Task 9: Re-fit the 5 owner-tier trainers (all variants) + floor

- [ ] **Step 1:** Owner authorizes the P2 spend (`cpu-performance` for ghost_gk/ghost_outfield, `cpu-xl`
  for xshot/xcross/receiver).
- [ ] **Step 2:** For each trainer and EACH variant in its `weights_paths`, launch with
  `tag="feat/hf-jobs-training-harness"` + `corpus_repo="<private-dataset>"`, poll, retrieve.
- [ ] **Step 3:** For each variant read the retrieved run's `metrics.json` head-to-head pair and apply
  `floor.evaluate_spec(spec, run_metrics)`; extend the head-to-head table.
- [ ] **Step 4:** Report to owner; regressions beyond δ BLOCK that variant pending owner investigation.

---

## Phase 3 — PR weights back (ends at the Commit 2 gate)

### Task 10: Copy cleared weights in, suite green, stamp re-mat, PR

- [ ] **Step 1:** For every trainer/variant that cleared the floor, copy the retrieved `weights/` artifacts
  into `silly_kicks/<weights_path>/`, MATCHING that bundle's existing committed shape (model + `SHA256SUMS`
  + `MODEL_CARD.md`, plus `metrics.json` ONLY for trainers that already ship one — now carrying the in-job
  platform stamp). Do NOT introduce a `metrics.json` into a bundle that never had one (keeps `test_bundled_*`
  / `test_model_cards_match_bundles` invariant). The head-to-head pair lives in the PR's delta table.
- [ ] **Step 2: Full suite green.**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: PASS — especially `test_bundled_*`, model-card-matches-bundle, chirality/feature-contract load
gates, and `test_model_cards_match_bundles`.

- [ ] **Step 3:** Stamp the downstream lakehouse re-materialize owed by the new weights into the PR
  description / a note (handed to the owner; this cycle does NOT run the re-mat).
- [ ] **Step 4: COMMIT 2 GATE — owner approval required.** Show the exact weights file list + the delta
  table (proof no trainer regressed beyond δ). Await explicit "yes". Only then:

```bash
git add silly_kicks   # only the re-fit weights dirs
git commit  # message: feat(weights): HF-Jobs re-fit of 9 bundled models (no-regression floor; ADR-122)
```

- [ ] **Step 5: Open the single PR** from `feat/hf-jobs-training-harness` (carries both commits).
  Await owner approval before merge; never merge until CI is green (separate commit/merge/tag go's).

---

## Self-Review (completed by plan author)

- **Spec coverage:** harness layout (Tasks 1-5) ✓; Jobs-not-Spaces + launch quirks (Task 3/Global) ✓;
  private-dataset staging + no-creds-on-runner (Task 5/8) ✓; no-Databricks (Global; stage.py docstring) ✓;
  head-to-head no-regression floor (Task 2 + Task 2b scorers + Tasks 7/9/10) ✓; registry/ADR-056 gate
  (Task 1) ✓; reproducibility pin + IN-JOB platform stamp (Task 2b `headtohead.stamp`, zero trainer edits;
  Task 6 ADR/howto) ✓; phasing P0-P3 with gates ✓; 2-commit discipline (Tasks 6/10 gates) ✓; branch/PR
  (Global; Task 10) ✓; chirality re-confirm (Task 7 Step 3) ✓; gk_retention blocked+registered (Task 1
  BLOCKED) ✓; docs + C4 (Task 6) ✓.
- **Placeholder scan:** `primary_metric` names are now VERIFIED against the real committed artifacts
  (2026-10-13). The remaining impl-time confirmations are: each trainer's exact predict surface (forced
  feasible in P0 by the per-trainer Task 2b feasibility test, never discovered at spend time), the owner
  corpus `--data-dir`/`--providers` flag per owner trainer (confirmed against each `scripts/train_*.py`),
  and the δ values (owner-ratified, fail-closed, P1-calibrated). None left as "TBD".
- **Type consistency:** `TrainerSpec` fields used identically across registry/floor/launch;
  `build_inner_command`/`build_argv`/`launch_job`/`evaluate_spec` signatures match their call sites.

**Resolved (was an open note):** the platform/version stamp is written IN-JOB by `headtohead.stamp()`
(Task 2b), post-fit and pre-upload — tag-independent, uniform across trainers, and **zero trainer edits**.
This supersedes the earlier idea of a per-trainer fit-time stamp: the jobs clone the **Commit-1 ref** and
run the UNCHANGED trainers, so a fit-time trainer edit is neither possible nor needed.

## AS-BUILT (2026-10-14) — bucket storage + resume, per-variant jobs

Executing P1/P2 drove `launch.py`-local changes beyond this plan (all owner-approved; ADR-122 amendment
carries the rationale). `launch.py` is the LOCAL orchestrator (never runs in-job), so these needed no
change to the cloned ref — `headtohead`/`repackage`/trainers/weights are unchanged. Verified on real HF
(public + all 3 owner source-modes + a cancel→relaunch resume).

- **Storage = read-write HF Storage Bucket, not `/tmp`.** HF Jobs ephemeral disk is a fixed per-flavor
  cap (≤ 50 GB, not configurable); `win_probability --all-competitions` exhausted it (`OSError [Errno
  28]`). A private bucket (`silly-kicks/hf-refit-scratch`) is mounted `-v hf://buckets/…:/scratch`;
  trainer output + corpus-driver shards write to `/scratch/{run_key}/{name}/{variant}`.
- **Resume = `--attempts N` + the persistent bucket.** A crashed/timed-out job auto-retries; `for_each`
  skips shards already in the bucket (`0.0s skip (shard exists)`) and continues (the DGX OOM-continue
  behavior). Supersedes relying on job size.
- **One job per committed variant** (`launch_all_variants` → 15 jobs), each with its OWN head-to-head
  floor — this **closes the "default-only floor" follow-up** (every variant is now floor-gated). Bounded
  per-variant timeout + partial-results safety.
- **No in-job upload.** Outputs persist in the bucket; retrieved locally with `retrieve_variant`
  (`hf cp`). The `hf-refit-artifacts` dataset is dropped. `huggingface_hub` is installed in-job only for
  the corpus `hf download` (the mini-wave caught `ModuleNotFoundError: huggingface_hub` on the old upload
  step).
- **hf CLI ≥ v2.2 required** (for `-v/--volume`; v1.7.1 lacks it). `--` before `bash` still mandatory;
  `--secrets HF_TOKEN` (env form, token off the cmdline); poll via `inspect --format json`; id parsed
  from the v2.2 `id=<hex> … url=/jobs/<ns>/<hex>` output (not v1.7.1 "Job started with ID:").
- **THREE staged corpora, not two** (ADR-122 amendment): `hf-refit-tc3` (flat 179; ghost — sportec ≡
  idsse), `hf-refit-xshot` (the exact 17 recorded `corpus_match_ids`, gen `a34e992`; xshot/xcross — own
  dataset because repackage globs the whole corpus), `hf-refit-sb360` (30-cache + offline manifest;
  receiver).
- **Timeouts:** `win_probability` 45m → **2h** (3961 matches + bucket-FUSE shard writes); others unchanged
  (public 45m; ghost 6h, xshot/xcross 4h, receiver 4h) — now generous per-single-variant ceilings, to be
  tuned down after wave 1 reveals real per-variant wall-times.
- **Unchanged:** 2-commit discipline (harness = Commit 1; all weights = Commit 2), owner-gated, non-squash
  merge of #284; registry ADR-056 population gate; `_loader_pining` offline-manifest; ZERO Databricks.
