# Combined Cycle Completion Implementation Plan (rev 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Owner overrides that bind every task:**
> - Exactly two commits (C1, C2), each only after the owner's explicit yes for that commit.
> - No per-task commits.
> - No worktrees: one feature branch.
> - Push, PR, merge and tag are each a separate explicit approval.
> - Each post-release Hub push batch (mirror republish; Hub-only cards) is a separate explicit approval too.
> - Never merge `main` into the branch.
> - Phase B amendment 3 (owner-approved 2026-10-04) supersedes the commit structure: C1 is rebased onto `main` and squash-merged as PR-1 before the wave (the run commit is that merge commit, M); C2 follows as PR-2 on a new branch.

**Goal:** Finish the combined provenance cycle and cut the release. Every artifact must be traceable to a clean commit reachable from `main`, and no owner-tier id may enter the repo. The work:
- corrected F1b re-fits;
- reused valid re-fits;
- T10;
- DAS corpus parity + performance;
- §7.3 downstream;
- the das-native leftovers.

**Architecture:**
- **C1** carries all new code, the reused weights and the re-captured golden.
- **One DGX wave at the C1 SHA** produces every new artifact. The receiver gate follows the receiver re-fit; the DAS benchmark runs alone at the end.
- **C2** adds the re-fit weights, the artifacts, the gates that read them, and the release.

**Tech Stack:**
- Python 3.10+ (DGX venvs 3.12), pandas 2/3, numpy, numba, xgboost, ruthless-efficiency `>=0.6,<0.7`, scikit-learn, pytest + pytest-benchmark, huggingface_hub;
- the repo's `scripts/_driver.py` `for_each` seam and `scripts/_partition.py`;
- the pining loader;
- DGX Spark (aarch64, 20 cores, 119 GiB).

**Spec:** `docs/superpowers/specs/2026-10-01-combined-cycle-completion-design.md` (rev 8). Read it first; this plan argues from it. Reviews: two independent review sessions, reports in `$REVIEWS` (Global Constraints); every round's verdicts and dispositions are in spec Appendix A. Rev 3 of the plan implemented the owner decisions of 2026-10-02 (spec §12) and B round-2 CCC-PLAN-20..23 + CONSIDER C1–C9. Rev 4 implements the revised D9 (owner, 2026-10-02):
- new Task 6b: the card-only Hub push seam;
- Task 7: Hub README-vs-card check, and `ghost-gk-v1` re-classified `hf_only` (the rev-3 registry still said `mirror`);
- Task 22 Steps 10–14: the post-release Hub pushes, performed by this session; each batch (mirrors, Hub-only cards) is its own explicit OWNER GATE (A r4 A-PLAN-02).

Rev 4 also applies B round-3 CCC-PLAN-24..28 (spec Appendix A): Task 10 (`nbmax` kept, token-naming refusal, `_map_generation` / `--print-generation`), Task 11 (typed `best_of`), Task 12b (generation-keyed DAS done-marker, `--das-generation`, `--prep-only` test), Tasks 17–18 (fresh shard root, generation passed), and pyright in the in-task check.

Rev 5 applies B round 4 (spec Appendix A, round-4 table): Task 2b (message-asserting test; probe ∩ training refused), Task 5 (receiver test seam offline, D7 `main()` test, sandboxed run, `pathlib.Path`), Task 6 (`require_clean_tree` in `main`; discrimination and D8-bound tests), Task 6b (seam test gaps), Task 7 (injectable `run()`, `--require-mirrors-match-wheel`, anonymous downloads), Task 8 (commit-keyed generation test), Task 12b (`--prep-only` test stubs only the registry check), Task 14 Step 1b (bundled probe ids pinned), Task 15 Step 6 (final spec status), Task 21 Steps 1b and 8 (row-anchored card test; stronger id scan), Task 22 Step 13.

Rev 6 applies B round 5 (spec Appendix A, round-5 table): the Rule C exemption for the Hub-smoke `run()` and a Global Constraint that runs the repo-wide script gates in every `scripts/` task (CCC-PLAN-41); hermetic Task 2b refusal tests (CCC-PLAN-42); the pre-registered C1 `mirrors_mismatched` (CCC-PLAN-43); the absent-meta probe guard; and the outfield card anchor.

Rev 7 applies B round 6 (CCC-PLAN-44): the cached-probe test's fixture saves `labels.npy` as a numeric array, matching how the trainer reads it.

## Global Constraints

- **Anchors:** reused artifacts at `3ca609f`; every new run at **C1** (spec §2). `run_tree_dirty == false` everywhere. No `--allow-dirty` on any production run.
- **Completeness and commits:**
  - Completeness = every listed key has a shard or an exclusion marker in the ONE generation.
  - Generations are keyed on the run commit.
  - Every worker manifest names the same commit.
  - Summed `n_attempted` is informative only (spec §2).
- **Pining token per run, never sourced globally** (spec §4):
  - Public = `~/cc/bin/public` (unsets the variable; the loader falls back to `test-token-pining-for-the-data`).
  - Owner = `~/cc/bin/owner`, which refuses an empty or unreadable token.
  - Public-arm runs use the public token; owner-tier runs use the owner token.
- **Public-arm corpus** = `BUNDLED_PUBLIC_ARM` (the original 17; gk_completion skillcorner = its 10 SkillCorner ids).
- **Every output stays outside every checkout:** `--out`, `--output-dir`, shard root and cache.
- **Private locations are variables, never paths.** No committed doc names a local path or a private host; the owner supplies the three paths at Task 0 Step 2 (the session already knows `$DGX`), and none of them enter the repo:
  - `$ARCHIVE` — the private archive of the `3ca609f` DGX outputs: `f1b_artifacts_3ca609f.tar`, `logs/`, `corpus_lists/`, `tc3_cache_f64/`, `tc3_cache_f32/`;
  - `$CC_OUT` — the private collection directory for this cycle's DGX outputs (Task 20);
  - `$REVIEWS` — the shared external reviews folder;
  - `$DGX` — the DGX ssh target (`user@host`).
- **Private data never enters the repo:** owner-tier match ids, `corpus179.json`, slices, sample lists, tc3 corpora, per-match tables. Committed artifacts carry aggregates only. Match ids are recorded only for all-public corpora; restricted corpora get a SHA-256 digest.
- **Partitioned runs:** `--providers` lists only providers with a key in the allowlist; partitioned drivers use `_partition.providers_for_slice`.
- **Code in `scripts/` is ASCII-only** (`test_driver_source_is_ascii`). Never use `§`, `Δ`, `→` or `≥` there.
- **Lint at CI scope:**
  - `python -m ruff format silly_kicks/ tests/ scripts/` (write), then `--check`;
  - `python -m ruff check silly_kicks/ tests/ scripts/`;
  - bare `pyright`.
- **Tests:** CI-faithful (no `-W`): `python -m pytest tests/ -m "not e2e" -p no:randomly --benchmark-skip --tb=short`, plus the same with `-m slow`.
- **Failing tests:** a test that pins bundled-model output is re-captured per its own convention, with the measured move recorded — never loosened. A test that stubs a seam changed here is updated in the same task.
- **Version:** the version string (D10) appears only in C2. NEXT-FREE numbers are derived READ-ONLY from `origin/main` (`git fetch && git show origin/main:silly_kicks/_version.py`, `git show origin/main:CHANGELOG.md`).
- **Owner decisions (DECIDED 2026-10-02, spec §12):**

  | Decision | Outcome |
  |---|---|
  | D1 | benchmark, gold standard |
  | D2 | four-cell parity |
  | D3 | launcher fix + validations in this cycle |
  | D3a | stratified ~30-match serial check |
  | D3b | study-worker RSS on the small paired set |
  | D4 | original 17 |
  | D5 | (c) held-out GS probe |
  | D6 | per-engine constant |
  | D7 | (a) manifest label `restricted` + receiver re-fit at C1 (licensed-derived weights in the wheel accepted) |
  | D8 | combined rule: point estimate ≥ 0 AND 95 % LB > −0.01 |
  | D9 | revised: this session republishes the 4 mirrors (weights + card) and pushes the 5 Hub-only cards after the release, via the Task 6b card-only seam, each batch behind its own explicit OWNER GATE; all relevant cards updated |
  | D10 | 4.128.0, breaking |

- **Repo-wide script gates run in every task that adds or changes a file under `scripts/`** (B r5 CCC-PLAN-41: a new loop tripped Rule C, and no task command ran it): `python -m pytest tests/scripts/test_corpus_driver_resilience.py tests/scripts/test_provenance_wiring.py -q`, plus the ASCII gate (`-k ascii`), in addition to the task's own PASS command.
- **Code blocks** are applied verbatim, then `ruff format`, `ruff check` and `pyright <the files this task touched>` run immediately in the same task, not only at Task 15 (B r3 CCC-PLAN-25: pyright errors used to surface only at Task 15). A lint or type error left after formatting (E501 on a long message, RUF046, F841, a pyright `reportOptionalSubscript` on a branch the prose describes) is fixed in place by wrapping, deleting, narrowing or annotating; logic is never changed to satisfy a checker.

---

## File structure

| File | C | Responsibility |
|---|---|---|
| `scripts/_corpus.py` | C1 | `BUNDLED_PUBLIC_ARM`; G1 helpers; `corpus_identity`; `reproducibility` |
| `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` | C1 | `--expect-variant`; ship-time check; corpus identity; reproducibility; `--prep-only` / `--list-studies` / `--study-list` (Task 12b, D3) |
| `scripts/train_xcross_attempt.py` | C1 | `--probe-match-ids-json`: probe-only extraction pass, never training (Task 2b, D5c) |
| `scripts/_parallel_launch.py` | C1 | `{worker}` token; das done-marker over list-matches items + `.excluded.json` (Task 12b, D3); refuses once, before any worker, when the memory cap cannot start (Phase B amendment F3) |
| `scripts/_mem_cap.py` | C1 | cgroup cap via the user manager for a non-root user (`systemd-run --user --scope`); `preflight` (Phase B amendment F3) |
| `scripts/train_gk_completion.py` | C1 | skillcorner pre-extraction refusal; `requested_match_ids` |
| `scripts/train_ghost_gk.py` | C1 | emits `reproducibility` (restricted, fail-closed) |
| `scripts/train_receiver_model.py` | C1 | `--match-ids-json`; `corpus_visibility` from the manifest; reproducibility (D7a) |
| `scripts/validate_receiver_widening.py` (new) | C1 | receiver gate driver (§7; D8 combined rule) |
| `scripts/validate_hub_variants.py` (new) | C1 | Hub smoke driver (§9); README-vs-card check; fail-closed load refusals recorded as `load_refused` (owner-approved 2026-10-02) |
| `scripts/_hub_publish.py`, `scripts/publish_model_card.py` (new) | C1 | `CARD_SOURCE`; card-only seam `publish_card_only`; LF card staging in both seams (Task 6b, D9) |
| ADR-088 amendment; `docs/context/trained-models.md`; `AGENTS.md` publish bullet | C1 | card-only seam documented (Task 6b) |
| `scripts/measure_f1b_feature_delta.py` | C1 | T10 sharding, commit-keyed generation; ghost_gk adapter keys via `return_meta=True` (amendment 2 F9) |
| `scripts/_partition.py`, `scripts/build_gkdv_arm_values.py`, `scripts/build_layer2_spells.py`, `scripts/build_tf60_layer3_arm_values.py`, `scripts/validate_xshot_causal.py` | C1 | `partition_keys` in every summed worker manifest, overlap refusal in `aggregate_manifests`, GKDV/spells `--reduce-only` (`require_population_shards`, `population_table`, `worker_lineage`) (amendment 2 F10) |
| `scripts/build_tf19_instrument_responsiveness.py` | C1 | `--reduce-only` over the allowlist population; worker `partition_keys` (amendment 2 F10) |
| `scripts/validate_das_native_parity.py` | C1 | schema `-4`; golden-bound counts (4 cells); finite / D-KEY / direction counts; completeness + commit checks; `providers_for_slice`; `--benchmark`; string-id player key (F1), possession-derived path-leg frames (F2), path-leg memory ceiling + speedups over finished pairs (F7) (Phase B amendment); worker `partition_keys` (amendment 2 F10); exclusions counted from the markers (F11) |
| `scripts/_das_reference_leg.py` | C1 | in-process timing; `repeat`; inferred-direction leg; string ids pass through, `_id_key` parse-back (F1); player results read by input row, result-shape guard (F5); carrier forwarded (F6) (Phase B amendment); rows frame-sorted before the library call (amendment 2 F8) |
| `scripts/_das_path_timing.py` (new) | C1 | `add_das` / `das_xfns` timing harness, run in a pandas-2 env for both paths; memory-ceiling watchdog (Phase B amendment F7) |
| `scripts/_loader_pining.py` | C1 | unique temp name per download (race fix) |
| `silly_kicks/tracking/_das.py` | C1+C2 | per-engine cost constant (D6); constants re-derived in C2 |
| `silly_kicks/tracking/_das_engine.py` | C1 | numpy default `chunk_size` 32 → 16 (`_DEFAULT_NUMPY_CHUNK`; Task 13 Step 2, ADR-107 chunk table) |
| `tests/tracking/test_das.py`, `tests/tracking/test_das_cost_guardrail.py` | C1 | cost-guardrail tests independent of the constant and the engine (D6 consumer set) |
| `tests/scripts/_corpus_load_rules.py` | C1 | Rule C exemptions: `validate_hub_variants.run` (Task 7) and `validate_das_native_parity.run_benchmark` (Task 11) |
| `tests/tracking/test_das_benchmark.py` (new) | C1 | pytest-benchmark ms/frame per engine |
| ADR-106, ADR-107, ADR-108, `docs/context/tracking-features.md`, `.gitattributes` | C1 | amendments, chunk table, reference-leg notes (incl. the Phase B F1/F5/F6 contract and the amendment 2 F8 frame order), `diff` attribute |
| `docs/context/corpus-drivers.md` | C1 | partitions must not overlap; `--reduce-only` combine (amendment 2 F10) |
| `tests/test_bundled_weights_corpus_policy.py` (new) | C1+C2 | G2 |
| `tests/test_bundled_probe_ids_are_committed_ids.py` (new) | C1 | bundle probe ids ⊆ public arm + GS 10502/10503 (Task 14 Step 1b) |
| `tests/scripts/test_corpus_guard.py`, `test_expect_variant_wiring.py`, `test_gk_completion_public_arm_guard.py`, `test_receiver_widening_driver.py`, `test_hub_variants_driver.py`, `test_publish_model_card.py`, `test_das_path_timing.py` (new) + modified driver tests (incl. `test_parallel_launch.py`, `test_hub_publish_guard.py`; Phase B amendment: `test_das_reference_leg.py`, `test_das_native_parity_driver.py`, `test_mem_cap.py`, `test_das_path_timing.py`; amendment 2: `test_das_reference_contract.py`, `test_measure_f1b_feature_delta.py`, `test_partition.py`, `test_build_gkdv_arm_values.py`, `test_build_layer2_spells.py`) | C1 | tests |
| `.github/workflows/ci.yml` (`das-reference-contract` job), `tests/scripts/test_das_reference_contract.py` (new), `tests/test_ci_das_reference_contract_wired.py` (new), `docs/context/ci.md` | C1 | the reference leg's contract against the real pinned oracle (Phase B amendment, re-review m-4) |
| reused bundle dirs (6) | C1 | from the archive |
| `tests/tracking/data/ghost_velocity_path_baseline.npz` + its test docstring; `tests/tracking/test_position_only_bundled.py` | C1 (+C2) | golden + history constants |
| re-fit bundle dirs (xshot ×2, xcross ×2, gkc skillcorner, ghost `position_only`, receiver) | C2 | weights |
| `docs/research/{das_native_parity,f1b_float32,tf19_instrument_responsiveness,tf19_signoff_power,tf24_stage2_refresh}/…` | C2 | artifacts (aggregates only) |
| `tests/tracking/test_das_parity_artifact.py`, `tests/test_research_artifacts_carry_no_ids.py` (new) | C2 | artifact gates |
| `docs/huggingface/model-cards/*-model-card.md` (9); wheel `MODEL_CARD.md` (receiver, gk_completion ×2) | C2 | cards match the bundles (D9) |
| `tests/test_model_cards_match_bundles.py` (new) | C2 | card ↔ bundle gate (D9) |
| `tests/scripts/test_artifact_provenance_output.py` | C2 | `_UNPROVENANCED`: tf24 added, tf19 removed |
| `CHANGELOG.md`, `TODO.md`, `silly_kicks/_version.py` | C2 | release |

---

## PHASE A — local, up to commit C1

### Task 0: Decisions, branch, baseline

- [ ] **Step 1: Confirm the recorded decisions** (Global Constraints table; spec §12). Every task below implements those outcomes and no alternative. If the owner changes any decision before execution, STOP and re-plan the affected tasks.
- [ ] **Step 2: Private locations + local corpora check** (already preserved; spec §0.9). Ask the owner for `$ARCHIVE`, `$CC_OUT` and `$REVIEWS` and record them in the session notes only. Then:

```bash
for d in tc3_cache_f64/tc3-cache tc3_cache_f32/tc3-cache-f32; do
  find "$ARCHIVE/$d" -path '*shards*' -name '*.parquet' | wc -l   # 179 each
done
```

- [ ] **Step 3: Branch** (no worktree):

```bash
git fetch origin && git status --short          # only the spec + plan untracked
git switch -c feat/combined-cycle-completion origin/main
git merge-base --is-ancestor 3ca609f HEAD && echo "3ca609f reachable"
```

- [ ] **Step 4: Baseline** the CI-faithful suite on the untouched branch (both `-m "not e2e"` and `-m slow`), and record the counts in the session notes.

### Task 1: `scripts/_corpus.py` — `BUNDLED_PUBLIC_ARM`, G1 helpers, corpus identity, reproducibility

**Files:** Modify `scripts/_corpus.py` (append after `assert_public_corpus`; add `import hashlib`, `import json` at the top). Test `tests/scripts/test_corpus_guard.py` (create).

**Interfaces (produced):**

| Name | Signature |
|---|---|
| `BUNDLED_PUBLIC_ARM` | `dict[str, tuple[str, ...]]` |
| `match_id_pairs` | `(providers, match_ids) -> list[list[str]]` |
| `bundled_public_arm_pairs` | `(providers=("idsse", "skillcorner")) -> list[list[str]]` |
| `requested_is_all_public` | `(pairs, visibility) -> bool` |
| `check_expected_variant` | `(expected, *, all_public) -> None` |
| `check_shipped_variant` | `(expected, shipped) -> None` |
| `corpus_identity` | `(providers, match_ids, *, all_public: bool) -> dict` |
| `reproducibility` | `(shipped: str, providers, *, training_commit: str \| None = None) -> dict` |

- [ ] **Step 1: Failing tests** — `tests/scripts/test_corpus_guard.py`:

```python
"""G1 corpus guard + corpus identity + reproducibility helpers (combined-cycle-completion spec 5, 0.11)."""

import pytest

from scripts._corpus import (
    BUNDLED_PUBLIC_ARM,
    PUBLIC_CORPUS,
    bundled_public_arm_pairs,
    check_expected_variant,
    check_shipped_variant,
    corpus_identity,
    match_id_pairs,
    reproducibility,
    requested_is_all_public,
)


def test_bundled_public_arm_is_the_original_17_inside_the_registered_public_corpus():
    assert len(BUNDLED_PUBLIC_ARM["skillcorner"]) == 10
    assert len(BUNDLED_PUBLIC_ARM["idsse"]) == 7
    for prov, ids in BUNDLED_PUBLIC_ARM.items():
        assert set(ids) <= PUBLIC_CORPUS[prov]
    # 1874553 is one of the ten SkillCorner ids added on 2026-09-10 (ce0401a) -- not in the bundled arm
    assert "1874553" in PUBLIC_CORPUS["skillcorner"]
    assert "1874553" not in BUNDLED_PUBLIC_ARM["skillcorner"]
    assert set(BUNDLED_PUBLIC_ARM["idsse"]) == PUBLIC_CORPUS["idsse"]


def test_bundled_public_arm_pairs_shape():
    pairs = bundled_public_arm_pairs()
    assert len(pairs) == 17
    assert pairs == sorted(pairs)
    assert ["skillcorner", "1886347"] in pairs
    assert ["idsse", "DFL-MAT-J03WMX"] in pairs
    assert bundled_public_arm_pairs(("skillcorner",)) == [
        ["skillcorner", m] for m in sorted(BUNDLED_PUBLIC_ARM["skillcorner"])
    ]


def test_match_id_pairs_dedups_sorts_and_stringifies():
    assert match_id_pairs(["sk", "sk", "id"], [2, 2, "a"]) == [["id", "a"], ["sk", "2"]]


@pytest.mark.parametrize(
    ("pairs", "vis", "want"),
    [
        ([], {}, False),  # an empty request is never public
        ([("sk", "1")], {("sk", "1"): "public"}, True),
        ([("sk", "1"), ("sk", "2")], {("sk", "1"): "public", ("sk", "2"): "private"}, False),
        ([("sk", "1")], {}, False),  # absent from the manifest -> restricted (fail-closed)
    ],
)
def test_requested_is_all_public(pairs, vis, want):
    assert requested_is_all_public(pairs, vis) is want


def test_check_expected_variant_public_refuses_a_restricted_request():
    with pytest.raises(SystemExit, match="non-public"):
        check_expected_variant("public", all_public=False)


@pytest.mark.parametrize(
    ("expected", "all_public"), [(None, False), ("public", True), ("sc_extended", False), ("full", False)]
)
def test_check_expected_variant_passes(expected, all_public):
    check_expected_variant(expected, all_public=all_public)


def test_check_shipped_variant():
    check_shipped_variant(None, "sc_extended")
    check_shipped_variant("public", "public")
    with pytest.raises(SystemExit, match="would ship 'sc_extended'"):
        check_shipped_variant("public", "sc_extended")


def test_corpus_identity_records_ids_only_for_an_all_public_corpus():
    pub = corpus_identity(["sk", "sk"], ["1", "2"], all_public=True)
    assert pub == {"corpus_match_ids": [["sk", "1"], ["sk", "2"]]}
    restricted = corpus_identity(["sk", "sk"], ["1", "999"], all_public=False)
    assert set(restricted) == {"corpus_match_ids_sha256", "corpus_n_matches"}
    assert restricted["corpus_n_matches"] == 2
    assert "999" not in str(restricted)  # no id leaks into a restricted artifact
    # the digest is a function of the identity, so two runs on one corpus agree and different corpora differ
    assert corpus_identity(["sk"], ["999"], all_public=False) != restricted
    assert corpus_identity(["sk", "sk"], ["999", "1"], all_public=False) == restricted


def test_reproducibility_public_and_restricted():
    assert reproducibility("public", ["idsse", "skillcorner"]) == {"reproducibility": "public"}
    r = reproducibility("sc_extended", ["skillcorner"], training_commit="abc1234")
    assert r["reproducibility"] == "restricted"
    assert "abc1234" in r["reproducibility_note"] and "skillcorner" in r["reproducibility_note"]
    assert len(r["reproducibility_note"]) > 20  # the ADR-067 M4 test's floor
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'BUNDLED_PUBLIC_ARM'`): `python -m pytest tests/scripts/test_corpus_guard.py -q`.
- [ ] **Step 3: Implement** — append to `scripts/_corpus.py`:

```python
# The 17 matches the wheel-bundled PUBLIC-arm models were trained on: xshot / xcross `default` and
# `position_only` (all 17) and gk_completion `skillcorner` (the 10 SkillCorner ids). This is
# PUBLIC_CORPUS as it stood before the 2026-09-10 SkillCorner growth (ce0401a^). Re-fits pin THIS set;
# widening to the current PUBLIC_CORPUS is a separate, gated decision (combined-cycle-completion D4).
BUNDLED_PUBLIC_ARM: dict[str, tuple[str, ...]] = {
    "skillcorner": (
        "1886347",
        "1899585",
        "1925299",
        "1953632",
        "1996435",
        "2006229",
        "2011166",
        "2013725",
        "2015213",
        "2017461",
    ),
    "idsse": (
        "DFL-MAT-J03WMX",
        "DFL-MAT-J03WN1",
        "DFL-MAT-J03WOH",
        "DFL-MAT-J03WOY",
        "DFL-MAT-J03WPY",
        "DFL-MAT-J03WQQ",
        "DFL-MAT-J03WR9",
    ),
}


def match_id_pairs(providers, match_ids) -> list[list[str]]:
    """Sorted unique ``[provider, match_id]`` pairs of a per-row corpus."""
    return [list(p) for p in sorted({(str(p), str(m)) for p, m in zip(providers, match_ids, strict=True)})]


def bundled_public_arm_pairs(providers: tuple[str, ...] = ("idsse", "skillcorner")) -> list[list[str]]:
    """``BUNDLED_PUBLIC_ARM`` restricted to ``providers``, in the :func:`match_id_pairs` shape."""
    provs = [p for p in providers for _ in BUNDLED_PUBLIC_ARM[p]]
    mids = [m for p in providers for m in BUNDLED_PUBLIC_ARM[p]]
    return match_id_pairs(provs, mids)


def requested_is_all_public(pairs, visibility: dict[tuple[str, str], str]) -> bool:
    """True iff the requested corpus is non-empty and every requested match is public (fail-closed)."""
    if not pairs:
        return False
    return bool(
        is_public_row(
            providers=np.asarray([p for p, _ in pairs]),
            match_ids=np.asarray([m for _, m in pairs]),
            visibility=visibility,
        ).all()
    )


def check_expected_variant(expected: str | None, *, all_public: bool) -> None:
    """G1 launch preflight, run BEFORE extraction: an expected ``public`` artifact needs an all-public corpus.

    A ``public`` bundle can only come from a corpus in which every requested match is public: any
    owner-tier match makes the paired gate eligible to ship a restricted variant. ``sc_extended`` and
    ``full`` expectations are enforced at ship time only (:func:`check_shipped_variant`).
    """
    if expected == "public" and not all_public:
        raise SystemExit(
            "--expect-variant public, but the requested corpus contains non-public matches (is the "
            "OWNER pining token set?). A public bundle must be trained on public matches only. "
            "Refusing before extraction."
        )


def check_shipped_variant(expected: str | None, shipped: str) -> None:
    """G1 ship-time check: never write an artifact whose shipped variant differs from the expectation."""
    if expected is not None and shipped != expected:
        raise SystemExit(
            f"--expect-variant {expected}, but this run would ship {shipped!r}. Refusing to write the artifact."
        )


def corpus_identity(providers, match_ids, *, all_public: bool) -> dict:
    """The corpus identity an artifact may record: the exact ids for an all-public corpus, else a digest.

    Hub publishes copy ``metrics.json``, so an owner/NDA id must never appear in one; the digest still
    lets two artifacts be proven to share (or not share) a corpus.
    """
    pairs = match_id_pairs(providers, match_ids)
    if all_public and pairs:
        return {"corpus_match_ids": pairs}
    digest = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"corpus_match_ids_sha256": digest, "corpus_n_matches": len(pairs)}


def reproducibility(shipped: str, providers, *, training_commit: str | None = None) -> dict:
    """The ADR-067 M4 caveat, emitted by the TRAINER -- never hand-added to a driver-stamped artifact."""
    if shipped == "public":
        return {"reproducibility": "public"}
    provs = ", ".join(sorted({str(p) for p in providers}))
    at = f" at training_commit {training_commit}" if training_commit else ""
    return {
        "reproducibility": "restricted",
        "reproducibility_note": (
            f"Restricted: trained{at} on a corpus ({provs}) that includes matches which cannot be "
            f"redistributed or carries no public-visibility proof (variant {shipped!r}), so the weights "
            "cannot be reproduced from public data (ADR-067 M4)."
        ),
    }
```

- [ ] **Step 4: Run — expect PASS** (15 tests): `python -m pytest tests/scripts/test_corpus_guard.py -q`.

### Task 2: G1 + corpus identity + reproducibility in the xshot / xcross trainers

**Files:**
- Modify `scripts/train_xshot_occurrence.py`:
  - argparse after `--allow-dirty` (~`:662-667`);
  - preflight after `from _cache import …` (`:720`);
  - `config` (`:797-807`);
  - `assemble_studies` imports `:499`; paired verdict `:537`; single branch `:564-575`; metrics `:605-617`.
- Modify `scripts/train_xcross_attempt.py`:
  - argparse after `--allow-dirty` (`:807-812`);
  - preflight after `:867`;
  - `config` `:969-980`;
  - imports `:519`; override block `:578-581`; single branch `:608-624`; metrics `:655-667`.
- Test: `tests/scripts/test_expect_variant_wiring.py` (create).

**Interfaces:**
- Consumes the Task 1 helpers.
- Produces:
  - CLI `--expect-variant {public,sc_extended,full}`;
  - `config["expect_variant"]`;
  - metrics keys: `corpus_match_ids` or (`corpus_match_ids_sha256`, `corpus_n_matches`); `reproducibility` (+ `reproducibility_note` when restricted).

- [ ] **Step 1: Failing tests** — `tests/scripts/test_expect_variant_wiring.py`:

```python
"""G1 wiring: the public-arm trainers refuse a restricted corpus BEFORE extraction, refuse to ship a
variant other than the expected one (also via the persisted config an --assemble worker reads), and
record their corpus identity -- ids only when all-public (spec section 5)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _loader_pining  # noqa: E402  -- the module object the trainers' function-local imports resolve

from scripts import train_xcross_attempt as xc  # noqa: E402
from scripts import train_xshot_occurrence as xs  # noqa: E402


def _listing(monkeypatch, vis):
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))

    def _no_extraction(*a, **k):
        raise AssertionError("extraction started -- the G1 preflight must refuse first")

    monkeypatch.setattr(_loader_pining, "pining_source", _no_extraction)


_ARGV = ["--providers", "skillcorner", "--output-dir", "{out}", "--expect-variant", "public", "--allow-dirty"]


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_expect_public_refuses_a_restricted_request_before_extraction(trainer, tmp_path, monkeypatch):
    _listing(monkeypatch, {("skillcorner", "1886347"): "public", ("skillcorner", "900"): "private"})
    with pytest.raises(SystemExit, match="non-public"):
        trainer.main([a.format(out=tmp_path) for a in _ARGV])


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_expect_public_passes_an_all_public_request(trainer, tmp_path, monkeypatch):
    """The other side of the band: an all-public request clears the preflight and reaches extraction."""
    _listing(monkeypatch, {("skillcorner", "1886347"): "public"})
    with pytest.raises(AssertionError, match="extraction started"):
        trainer.main([a.format(out=tmp_path) for a in _ARGV])


def _inputs(root, feature_names, *, public: bool, expect, extra_config, n_games=4, rows=60):
    from scripts._study_shared import persist_study_inputs

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((n_games * rows, len(feature_names))), columns=feature_names)
    y = (X.iloc[:, 0] > 0).to_numpy().astype(int)
    mids = np.repeat(["1886347", "1899585", "DFL-MAT-J03WMX", "DFL-MAT-J03WN1"][:n_games], rows)
    persist_study_inputs(
        root,
        X=X,
        y=y,
        groups=mids.copy(),
        providers=np.repeat(["skillcorner", "skillcorner", "idsse", "idsse"][:n_games], rows),
        match_ids=mids,
        is_public=np.full(n_games * rows, public),
        config={
            "n_trials": 1,
            "negative_subsample": None,
            "seed": 42,
            "feature_set": "faithful",
            "horizon_seconds": 1.0,
            "study_db_dir": str(root),
            "artifact_dir": str(root / "art"),
            "run_paired": False,
            "run_prov": {"commit": "test", "dirty": False, "tree_state": "clean"},
            "expect_variant": expect,
            **extra_config,
        },
    )


def _names(trainer):
    if trainer is xs:
        from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL as names

        return names, {}, {}
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL as names

    return names, {"ship_variant": None}, {"run_probe": False}


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_ship_time_check_refuses_from_the_persisted_config_before_any_fit(trainer, tmp_path):
    """An --assemble worker reads the expectation from the persisted config: a restricted single-candidate
    corpus (owner-tier SkillCorner only) would ship sc_extended, so expect=public refuses with no fit."""
    names, extra, kw = _names(trainer)
    _inputs(tmp_path, names, public=False, expect="public", extra_config=extra)
    with pytest.raises(SystemExit, match="would ship 'sc_extended'"):
        trainer.assemble_studies(tmp_path, study_shard_dir=tmp_path, **kw)
    assert not (tmp_path / "art" / "model.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_public_run_records_ids_and_public_reproducibility(trainer, tmp_path):
    names, extra, kw = _names(trainer)
    _inputs(tmp_path, names, public=True, expect="public", extra_config=extra, rows=80)
    metrics, _model = trainer.assemble_studies(tmp_path, study_shard_dir=tmp_path, **kw)
    assert metrics["shipped_variant"] == "public"
    assert metrics["corpus_match_ids"] == [
        ["idsse", "DFL-MAT-J03WMX"],
        ["idsse", "DFL-MAT-J03WN1"],
        ["skillcorner", "1886347"],
        ["skillcorner", "1899585"],
    ]
    assert metrics["reproducibility"] == "public"


@pytest.mark.slow
def test_restricted_run_records_a_digest_never_ids(tmp_path):
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    _inputs(tmp_path, XSHOT_FEATURE_NAMES_FAITHFUL, public=False, expect=None, extra_config={}, rows=80)
    metrics, _model = xs.assemble_studies(tmp_path, study_shard_dir=tmp_path)
    assert metrics["shipped_variant"] == "sc_extended"
    assert "corpus_match_ids" not in metrics and metrics["corpus_n_matches"] == 4
    assert metrics["reproducibility"] == "restricted"
    assert "1886347" not in (tmp_path / "art" / "metrics.json").read_text(encoding="utf-8")
```

(xcross's `_probe_sample` is only read when `run_probe=True`, so the xcross slow test runs without it.)

- [ ] **Step 2: Run — expect FAIL** (`unrecognized arguments: --expect-variant`; no SystemExit from assemble; `KeyError: 'corpus_match_ids'`): first `python -m pytest tests/scripts/test_expect_variant_wiring.py -q -m "not slow"`, then with `-m slow`.
- [ ] **Step 3: Implement in `scripts/train_xshot_occurrence.py`.**

(a) argparse, after `--allow-dirty`:

```python
    ap.add_argument(
        "--expect-variant",
        choices=["public", "sc_extended", "full"],
        default=None,
        help="G1 guard: refuse BEFORE extraction unless the requested corpus can ship this variant "
        "(public => every requested match is public), and refuse at ship time if the shipped variant "
        "differs. Default off (unchanged behaviour).",
    )
```

(b) preflight — immediately after `from _cache import cache_is_valid, write_cache_meta`:

```python
    # G1 launch preflight (combined-cycle-completion spec section 5): BEFORE any corpus work, so a run
    # that would train a public bundle on owner-tier data never starts.
    if args.expect_variant is not None:
        if not args.providers:
            ap.error("--expect-variant needs --providers (only the pining path lists a requested corpus)")
        from _corpus import check_expected_variant, requested_is_all_public
        from _loader_pining import match_visibility, select_match_ids

        _provs = args.providers.split(",")
        _allow = json.load(open(args.match_ids_json)) if args.match_ids_json else None
        _pairs = select_match_ids(providers=_provs, match_ids=_allow, max_per_provider=args.max_per_provider)
        check_expected_variant(args.expect_variant, all_public=requested_is_all_public(_pairs, match_visibility(_provs)))
```

(c) In `config`, after `"run_prov": run_prov,`, add `"expect_variant": args.expect_variant,`.

(d) In `assemble_studies`:
- Import line: `from _corpus import artifact_label, check_shipped_variant, corpus_identity, reproducibility`.
- Paired branch: after `print(f"Fixed-sequence verdict: ship {shipped} -- {why}")`, add `check_shipped_variant(cfg.get("expect_variant"), shipped)`.
- Single branch: replace it with:

```python
    else:
        ship_mask = np.ones(len(X), bool)
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
        check_shipped_variant(cfg.get("expect_variant"), shipped)  # before the study: a refusal costs no fit
        params_all = _hpo_once(
            X, y, groups, out, "single", n_trials, negative_subsample=ns, seed=seed, study_shard_dir=study_shard_dir
        )
        candidates[shipped] = {
            "params": params_all,
            "metrics": _cv_metrics(X, y, groups, params_all, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }
```

(e) In the metrics dict, after `"providers": sorted(provset),`:

```python
        # Corpus IDENTITY (spec section 5): exact ids only for an all-public corpus, else a digest
        # (Hub publishes copy metrics.json). n_rows alone could not tell 17 public matches from 27.
        **corpus_identity(providers.tolist(), match_ids.tolist(), all_public=bool(is_public.all())),
        # ADR-067 M4 caveat, emitted here -- never hand-added at bundling (spec 0.11).
        **reproducibility(shipped, candidates[shipped]["providers"], training_commit=run_prov["commit"]),
```

- [ ] **Step 4: Implement the same in `scripts/train_xcross_attempt.py`:**
- the identical argparse block (after `--allow-dirty`);
- the identical preflight (immediately after `from _cache import cache_is_valid, write_cache_meta`, before `probe_bundle = …`);
- `"expect_variant": args.expect_variant,` in `config`, after `"ship_variant": args.ship_variant,`;
- the same import change;
- paired branch: `check_shipped_variant(cfg.get("expect_variant"), shipped)` after the `if ship_variant is not None:` override block;
- the single branch replaced with:

```python
    else:
        if ship_variant is not None:
            raise SystemExit(
                "--ship-variant requires the multi-candidate (run_paired) corpus (public + owner "
                "SkillCorner + gradientsports) so the variant masks and the TF-19 probe cohort exist."
            )
        ship_mask = np.ones(len(X), bool)
        ship_provs = set(providers[ship_mask].tolist())
        shipped = artifact_label(providers=ship_provs, all_public=bool(is_public[ship_mask].all()))
        check_shipped_variant(cfg.get("expect_variant"), shipped)  # before the study: a refusal costs no fit
        params_all = _hpo_once(
            X, y, groups, out, "single", n_trials, negative_subsample=ns, seed=seed, study_shard_dir=study_shard_dir
        )
        candidates[shipped] = {
            "params": params_all,
            "metrics": _cv_metrics(X, y, groups, params_all, negative_subsample=ns, seed=seed),
            "providers": sorted(provset),
        }
```

- the same two metrics lines after `"providers": sorted(provset),`.

- [ ] **Step 5: Run — expect PASS:** `python -m pytest tests/scripts/test_expect_variant_wiring.py -q` (all, including slow).
- [ ] **Step 6: Blast radius.** These tests stub or read the changed seams, so run them:

```bash
python -m pytest tests/scripts/test_train_study_parallel_parity.py tests/tracking/test_xshot_occurrence_integration.py \
  tests/tracking/test_xcross_attempt_integration.py tests/scripts/test_provenance_wiring.py tests/scripts/test_corpus_taxonomy.py \
  tests/scripts/test_trainer_load_seam.py tests/scripts/test_trainer_cache_and_providers.py \
  tests/scripts/test_corpus_driver_resilience.py tests/causal/test_owner_run_refusals.py -q
```

Expect PASS. A failure is fixed in the code or in the stub, never by deleting the assertion.

### Task 2b: xcross held-out probe (D5c)

**Files:** Modify `scripts/train_xcross_attempt.py`:
- argparse;
- the `probe_provs` check (`:858-861`);
- the cache-hit branch (`:877-883`);
- the providers branch of Phase 1 (`:885-909`).

Test: `tests/scripts/test_expect_variant_wiring.py` (append).

**Interfaces (produced):** CLI `--probe-match-ids-json FILE` (shape `{provider: [ids]}`). The probe cohort comes from a separate pass whose rows are discarded. The training stream captures no probe cohort when the flag is set.

- [ ] **Step 1: Failing test** (append):

```python
def test_probe_only_matches_never_enter_training(tmp_path, monkeypatch):
    """D5(c): GS probe matches are extracted in a separate pass for the TF-19 probe; training sees only the
    public allowlist, and the probe sample lists exactly the probe matches (held out by construction)."""
    import json

    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL as names

    vis = {("skillcorner", "1886347"): "public", ("idsse", "DFL-MAT-J03WMX"): "public"}
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))
    monkeypatch.setattr(
        _loader_pining,
        "pining_source",
        lambda provs, match_ids=None, **kw: ([(p, m) for p in provs for m in (match_ids or {}).get(p, [])], None),
    )

    def fake_extract(source, horizon, *, shard_root, probe_providers, probe_comparison_providers, feature_set, load, **kw):
        cohort, rows = xc._new_probe_cohort(), list(source)
        for prov, mid in rows:
            if prov in probe_providers:
                cohort["frames"].append(pd.DataFrame({"game_id": [mid], "frame_id": [1]}))
                cohort["actions"].append(pd.DataFrame({"game_id": [mid]}))
                cohort["home"] = 1
                cohort["matches"].append([prov, mid])
                cohort["match_groups"][mid] = [mid]
        X = pd.DataFrame(0.0, index=range(len(rows)), columns=names)
        mids = np.array([m for _, m in rows], dtype=object)
        provs = np.array([p for p, _ in rows], dtype=object)
        return X, np.zeros(len(rows), int), mids.copy(), provs, mids, (cohort, xc._new_probe_cohort(), 0)

    monkeypatch.setattr(xc, "_extract", fake_extract)
    import scripts._study_shared as ss

    seen = {}

    def stop(root, **kw):
        seen["providers"] = set(kw["providers"].tolist())
        raise SystemExit("stop")

    monkeypatch.setattr(ss, "persist_study_inputs", stop)
    allow, probe = tmp_path / "allow.json", tmp_path / "probe.json"
    allow.write_text(json.dumps({"skillcorner": ["1886347"], "idsse": ["DFL-MAT-J03WMX"]}))
    probe.write_text(json.dumps({"gradientsports": ["10502", "10503"]}))
    argv = ["--providers", "idsse,skillcorner", "--match-ids-json", str(allow), "--max-per-provider", "10",
            "--output-dir", str(tmp_path / "o"), "--expect-variant", "public", "--probe-providers", "gradientsports",
            "--probe-comparison-providers", "", "--probe-match-ids-json", str(probe), "--allow-dirty"]
    with pytest.raises(SystemExit, match="stop"):
        xc.main(argv)
    assert seen["providers"] == {"idsse", "skillcorner"}  # the probe matches never reach training
    meta = json.loads((tmp_path / "o" / "xcross_attempt_v1" / "_probe_sample" / "meta.json").read_text())
    assert meta["probe_matches"] == [["gradientsports", "10502"], ["gradientsports", "10503"]]


def _forbid_pining(monkeypatch):
    """Hermetic (B r5 CCC-PLAN-42): if a refusal regresses, main() continues into the corpus listing. That
    must fail the test, never reach the live pining API (the round-4 incident class)."""

    def _boom(*a, **k):
        raise AssertionError("a unit test reached the pining API")

    for name in ("select_match_ids", "match_visibility", "pining_source", "list_match_refs", "load_match"):
        monkeypatch.setattr(_loader_pining, name, _boom)


def test_probe_match_ids_must_name_a_probe_provider(tmp_path, capsys, monkeypatch):
    """Asserts the MESSAGE: a bare SystemExit also passes on RED, where argparse rejects the unknown flag
    (B r4 CCC-PLAN-39)."""
    import json

    _forbid_pining(monkeypatch)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"gradientsports": ["10502"]}))
    with pytest.raises(SystemExit):
        xc.main(["--providers", "idsse", "--output-dir", str(tmp_path / "o"), "--probe-providers", "skillcorner",
                 "--probe-comparison-providers", "", "--probe-match-ids-json", str(probe), "--allow-dirty"])
    assert "must all be in --probe-providers" in capsys.readouterr().err


def test_a_probe_match_the_training_corpus_can_load_is_refused(tmp_path, capsys, monkeypatch):
    """D5(c) held-out by construction: a probe id that training could also load is refused up front, not left
    to the DGX acceptance check (B r4 CCC-PLAN-39)."""
    import json

    _forbid_pining(monkeypatch)
    probe, allow = tmp_path / "probe.json", tmp_path / "allow.json"
    probe.write_text(json.dumps({"skillcorner": ["1886347"]}))
    allow.write_text(json.dumps({"skillcorner": ["1886347"], "idsse": ["DFL-MAT-J03WMX"]}))
    with pytest.raises(SystemExit):
        xc.main(["--providers", "idsse,skillcorner", "--match-ids-json", str(allow), "--output-dir", str(tmp_path / "o"),
                 "--probe-providers", "skillcorner", "--probe-comparison-providers", "",
                 "--probe-match-ids-json", str(probe), "--allow-dirty"])
    err = capsys.readouterr().err
    assert "disjoint from training" in err and "1886347" not in err  # the refusal names a count, never an id


@pytest.mark.parametrize("meta", [None, {"probe_matches": [["gradientsports", "99999"]]}], ids=["absent", "stale"])
def test_a_missing_or_stale_cached_probe_sample_is_refused(tmp_path, monkeypatch, meta):
    """Step 3(c): on a feature-cache hit, the persisted probe sample must match --probe-match-ids-json. An
    ABSENT one is refused too: it used to escape as FileNotFoundError (B r5, outside its round)."""
    import json

    import _cache

    _forbid_pining(monkeypatch)
    monkeypatch.setattr(xc, "_corpus_fingerprint", lambda args: "fp")
    monkeypatch.setattr(_cache, "cache_is_valid", lambda cache_dir, fingerprint: True)
    cache = tmp_path / "o" / "xcross_attempt_v1" / "_feature_cache"
    cache.mkdir(parents=True)
    pd.DataFrame({"a": [0.0]}).to_parquet(cache / "features.parquet")
    # Match how the trainer READS the cache (B r6 CCC-PLAN-44): labels are numeric and loaded without
    # allow_pickle (`train_xcross_attempt.py:880`); groups / providers / match_ids are object arrays loaded
    # with allow_pickle=True. An object-dtype labels.npy would die in np.load before the probe check.
    np.save(cache / "labels.npy", np.array([0]))
    for name in ("groups", "providers", "match_ids"):
        np.save(cache / f"{name}.npy", np.array(["x"], dtype=object), allow_pickle=True)
    if meta is not None:
        (cache.parent / "_probe_sample").mkdir()
        (cache.parent / "_probe_sample" / "meta.json").write_text(json.dumps(meta))
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"gradientsports": ["10502", "10503"]}))
    with pytest.raises(SystemExit, match="_probe_sample"):
        xc.main(["--providers", "idsse,skillcorner", "--output-dir", str(tmp_path / "o"),
                 "--probe-providers", "gradientsports", "--probe-comparison-providers", "",
                 "--probe-match-ids-json", str(probe), "--allow-dirty"])
```

Before trusting it, remove the absent-meta guard once: `[absent]` must FAIL with `FileNotFoundError` (B r6 measured exactly that), then restore it. `_cache` is the `scripts/` module object the trainer's function-local `from _cache import cache_is_valid` resolves (the test module already puts `scripts/` on `sys.path`, as for `_loader_pining`). The cache path is `out / "xcross_attempt_v1" / "_feature_cache"` (`train_xcross_attempt.py:864-865`).

`--max-per-provider 10` keeps `assert_public_corpus` in subset mode for the 2-match fake corpus. Read `_write_probe_sample` first; if it needs more cohort keys or frame columns than the fake provides, add them to the fake, never to the driver.
- [ ] **Step 2: Run — expect FAIL** (`unrecognized arguments: --probe-match-ids-json`).
- [ ] **Step 3: Implement** in `scripts/train_xcross_attempt.py`.

(a) argparse, after `--probe-comparison-providers`:

```python
    ap.add_argument(
        "--probe-match-ids-json",
        default=None,
        help='JSON {"gradientsports": ["10502", ...]}: matches loaded ONLY for the TF-19 substitution probe, in '
        "a separate pass whose rows are discarded -- a held-out probe for a public-only fit (combined-cycle D5c).",
    )
```

(b) Directly after the existing disjointness check on `probe_provs` / `comparison_provs`:

```python
    if args.probe_match_ids_json:
        if not args.providers:
            ap.error("--probe-match-ids-json needs --providers (probe matches are listed via pining)")
        _probe = json.load(open(args.probe_match_ids_json))
        _pkeys = set(_probe)
        if not _pkeys <= set(probe_provs):
            ap.error(f"--probe-match-ids-json providers {sorted(_pkeys)} must all be in --probe-providers")
        # Held out by construction (B r4 CCC-PLAN-39): refuse any probe match the TRAINING pass could also load
        # -- a training provider with no allowlist loads its whole manifest, so every probe id under it overlaps.
        _train_provs = {p for p in args.providers.split(",") if p}
        _train_allow = json.load(open(args.match_ids_json)) if args.match_ids_json else None
        _overlap = [
            (p, str(m))
            for p, ids in _probe.items()
            for m in ids
            if p in _train_provs and (_train_allow is None or str(m) in {str(x) for x in _train_allow.get(p, [])})
        ]
        if _overlap:
            ap.error(
                f"--probe-match-ids-json names {len(_overlap)} match(es) the training corpus can also load; "
                "a held-out probe must be disjoint from training"
            )
```

(c) In the cache-hit branch, after the `np.load` lines, add the stale-probe guard:

```python
        if args.probe_match_ids_json:
            _meta_path = cache.parent / "_probe_sample" / "meta.json"
            if not _meta_path.is_file():
                raise SystemExit("no cached _probe_sample/meta.json for --probe-match-ids-json; use a fresh --output-dir")
            _meta = json.load(open(_meta_path))
            _probe = json.load(open(args.probe_match_ids_json))
            _want = sorted([p, m] for p, ids in _probe.items() for m in ids)
            if sorted(_meta.get("probe_matches", [])) != _want:
                raise SystemExit("cached _probe_sample does not match --probe-match-ids-json; use a fresh --output-dir")
```

(d) Training `_extract(...)` call: `probe_providers=() if args.probe_match_ids_json else tuple(probe_provs),`. Immediately after that call returns, add:

```python
        if args.probe_match_ids_json:
            # D5(c): the TF-19 probe cohort comes from a SEPARATE pass over probe-only matches (e.g. GS
            # 10502/10503) whose feature rows are discarded -- they can never enter training.
            probe_allow = json.load(open(args.probe_match_ids_json))
            p_refs, p_load = pining_source(sorted(probe_allow), match_ids=probe_allow, cache_dir=args.cache_dir)
            *_discarded, p_bundle = _extract(
                p_refs,
                args.horizon_seconds,
                feature_set=args.feature_set,
                load=p_load,
                shard_root=art / "probe_shards",
                probe_providers=tuple(probe_provs),
                probe_comparison_providers=(),
            )
            probe_bundle = (p_bundle[0], probe_bundle[1], probe_bundle[2] + p_bundle[2])
```

In `assemble_studies`, `run_paired` is `False` for a public corpus, so `admitted` is `False`. `_gated_probe_matches` then passes every probe match through. `in_training` is `False` for GS games absent from the training groups, so the "NOT held-out" note never fires.
- [ ] **Step 4: Run — expect PASS:** `python -m pytest tests/scripts/test_expect_variant_wiring.py tests/tracking/test_xcross_attempt_integration.py tests/tracking/test_xcross_eval.py -q`.

### Task 3: G1 in gk_completion `skillcorner`

**Files:** Modify `scripts/train_gk_completion.py` (`_train_skillcorner` `:424-628`). Test `tests/scripts/test_gk_completion_public_arm_guard.py` (create).

**Interfaces (produced):** `metrics["requested_match_ids"]: list[[provider, match_id]]`. It is always all-public, because a restricted request is refused.

- [ ] **Step 1: Failing tests:**

```python
"""G1: gk_completion `skillcorner` is the wheel-bundled PUBLIC arm -- a restricted request is refused
before extraction; the requested ids are recorded (spec section 5)."""

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
train = importlib.import_module("train_gk_completion")
import _loader_pining  # noqa: E402


def _args(tmp_path, cache=None):
    return Namespace(
        max_per_provider=10,
        tracking_limit=None,
        cache_features=str(cache) if cache else None,
        mode="rebundle",
        reason="guard test",
        feature_space=None,
        probe_old=None,
        shard_dir=str(tmp_path / "shards"),
        cache_dir=None,
    )


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_SKILLCORNER_WEIGHTS_DIR", tmp_path / "skillcorner")
    monkeypatch.setattr(train, "_WEIGHTS_ROOT", tmp_path)


def test_skillcorner_refuses_a_restricted_request_before_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("sc_extended", False))

    def _no_extraction(*a, **k):
        raise AssertionError("extraction started -- the G1 refusal must come first")

    monkeypatch.setattr(train, "_extract", _no_extraction)
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="not all-public"):
        train._train_skillcorner(_args(tmp_path))


def test_skillcorner_passes_a_public_request_to_extraction(tmp_path, monkeypatch):
    """The other side of the band: an all-public request reaches extraction."""
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("public", True))
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: [("skillcorner", "1886347")])

    def _reached(*a, **k):
        raise AssertionError("extraction reached")

    monkeypatch.setattr(train, "_extract", _reached)
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="extraction reached"):
        train._train_skillcorner(_args(tmp_path))


@pytest.mark.slow
def test_skillcorner_records_requested_match_ids(tmp_path, monkeypatch):
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES as FEATS

    rng = np.random.RandomState(0)
    n = 160
    df = pd.DataFrame({f: rng.randn(n) for f in FEATS})
    df["is_goalkick"] = (np.arange(n) % 4 == 0).astype(float)
    df["is_throw_in"] = 0.0
    df["_y"] = (rng.rand(n) < 0.6).astype(int)
    df["_group"] = np.arange(n) % 5
    cache = tmp_path / "feat.parquet"
    df.to_parquet(cache)
    monkeypatch.setattr(train, "_corpus_taxonomy", lambda providers, mpp: ("public", True))
    monkeypatch.setattr(
        _loader_pining, "select_match_ids", lambda **kw: [("skillcorner", "1886347"), ("skillcorner", "1899585")]
    )
    _isolate(monkeypatch, tmp_path)
    assert train._train_skillcorner(_args(tmp_path, cache)) == 0
    written = tmp_path / "skillcorner" / "metrics.json"
    if not written.exists():
        written = tmp_path / "skillcorner_remeasurement.json"
    m = json.loads(written.read_text(encoding="utf-8"))
    assert m["requested_match_ids"] == [["skillcorner", "1886347"], ["skillcorner", "1899585"]]
    assert m["artifact_label"] == "public" and m["all_public"] is True
```

- [ ] **Step 2: Run — expect FAIL** (the refusal test reaches `_extract`; the slow test KeyErrors).
- [ ] **Step 3: Implement.** In `_train_skillcorner`, immediately after `run_prov = git_provenance()`:

```python
    # G1 (combined-cycle-completion spec section 5): `skillcorner` is the wheel-bundled PUBLIC arm.
    # Refuse BEFORE extraction when the requested SkillCorner corpus is not all-public. Previously only
    # the rebundle tolerance stopped a 64-match owner-token run from bundling restricted data.
    sc_label, sc_all_public = _corpus_taxonomy(["skillcorner"], args.max_per_provider)
    if not sc_all_public:
        raise SystemExit(
            f"gk_completion `skillcorner` is the wheel-bundled PUBLIC arm, but the requested SkillCorner "
            f"corpus is {sc_label!r}, not all-public (is the OWNER pining token set?). Use the public "
            "token and --max-per-provider 10 (MODEL_CARD). Refusing before extraction."
        )
    from _loader_pining import select_match_ids

    requested_match_ids = [
        [p, m] for p, m in select_match_ids(providers=["skillcorner"], max_per_provider=args.max_per_provider)
    ]
```

Then:
- delete the later `sc_label, sc_all_public = _corpus_taxonomy(["skillcorner"], args.max_per_provider)` (old `:586`);
- add to the `metrics` dict, after `"all_public": sc_all_public,`:

```python
        # The requested corpus IDENTITY (spec section 5) -- always all-public here (refused otherwise).
        "requested_match_ids": requested_match_ids,
```

- [ ] **Step 4: Run — expect PASS:** the new file, then:

```bash
python -m pytest tests/tracking/test_gk_completion_variants.py tests/scripts/test_train_gk_completion.py \
  tests/scripts/test_gk_completion_taxonomy.py tests/tracking/test_gk_completion_bundle_calibration.py \
  tests/tracking/test_gk_completion_pertype_gate.py -q
```

### Task 4: ghost-GK trainer emits the reproducibility caveat

**Files:** Modify `scripts/train_ghost_gk.py` (metrics dict `:1118-1135`). Test: modify `tests/scripts/test_train_ghost_gk_sweeper.py` (after `:204`).

- [ ] **Step 1: Failing assertion.** In `tests/scripts/test_train_ghost_gk_sweeper.py`, directly after `metrics = json.loads((out / "ghost_gk_v1" / "metrics.json").read_text())` (`:204`), add:

```python
    # ADR-067 M4 caveat emitted by the trainer (combined-cycle-completion spec 0.11): a --data-dir corpus
    # carries no public-visibility proof, so it is restricted, and the note names its own commit.
    assert metrics["reproducibility"] == "restricted"
    assert metrics["training_commit"] in metrics["reproducibility_note"]
```

Run `python -m pytest tests/scripts/test_train_ghost_gk_sweeper.py -q`: FAIL with `KeyError: 'reproducibility'`.
- [ ] **Step 2: Implement.** In `train_ghost_gk.py`'s `metrics` dict, after `"run_tree_state": run_prov["tree_state"],` (the file's function-local import style is `from scripts._… import …`):

```python
        # ADR-067 M4, emitted by the trainer, never hand-added at bundling: a --data-dir corpus carries
        # no public-visibility proof, so it is restricted (fail-closed, like is_public_row's default).
        **reproducibility("restricted", provider_labels.tolist(), training_commit=training_commit),
```

Add `from scripts._corpus import reproducibility` beside the other function-local `scripts._…` imports in `main()`.
- [ ] **Step 3: Run — expect PASS:** `python -m pytest tests/scripts/test_train_ghost_gk_sweeper.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_position_only_bundled.py -q`.

### Task 5: Receiver trainer — allowlist, manifest-derived visibility, reproducibility (D7a)

**Files:**
- Modify `scripts/train_receiver_model.py`:
  - `_corpus_source` `:366-380` and its 2 callers `:387`, `:421`;
  - `_extract_provider_rows` `:416`;
  - argparse `:444-477`;
  - label `:529`;
  - manifest dict.
- Test: `tests/scripts/test_train_receiver_model.py` (append).

**Interfaces (produced):**
- `_corpus_source(provider, cache_dir, match_ids=None)`.
- CLI `--match-ids-json` (shape `{provider: [ids]}`).
- `manifest["corpus_visibility"]` ∈ {`"public"`, `"restricted"`}: public iff every requested `(provider, match_id)` is manifest-public.
- `manifest` gains `reproducibility` (+ note) and `corpus_n_matches`.

- [ ] **Step 1: Read first.** Read `tests/scripts/test_train_receiver_model.py` around `:400-430`: the test that asserts a zero-contribution pool is not stamped into `providers_trained`/`corpus_visibility`. Keep its intent.
- [ ] **Step 2: Failing tests** (append):

```python
def test_corpus_visibility_is_keyed_on_the_manifest_not_the_provider_name():
    """ADR-038: statsbomb is not public BY NAME. The pining statsbomb manifest marks every match private
    (licensed, ADR-062), so a statsbomb-trained receiver is `restricted` (combined-cycle spec D7)."""
    from scripts.train_receiver_model import _corpus_visibility

    assert _corpus_visibility([("statsbomb", "1"), ("statsbomb", "2")], {("statsbomb", "1"): "private"}) == "restricted"
    assert _corpus_visibility([("statsbomb", "1")], {("statsbomb", "1"): "public"}) == "public"
    assert _corpus_visibility([], {}) == "restricted"  # fail-closed


def test_corpus_source_threads_the_allowlist(monkeypatch):
    import scripts._loader_pining as lp
    from scripts.train_receiver_model import _corpus_source

    seen = {}

    def fake_pining_source(providers, **kw):
        seen.update(kw, providers=providers)
        return [], (lambda ref: None)

    monkeypatch.setattr(lp, "pining_source", fake_pining_source)
    _corpus_source("statsbomb", None, match_ids={"statsbomb": ["7"]})
    assert seen["providers"] == ["statsbomb"] and seen["match_ids"] == {"statsbomb": ["7"]}


@pytest.mark.parametrize(("vis", "want"), [("public", "public"), ("private", "restricted")])
def test_main_labels_the_bundle_from_the_manifest_visibility(tmp_path, monkeypatch, vis, want):
    """D7 through main(): the same statsbomb corpus is `public` or `restricted` by its MANIFEST alone."""
    _install_corpus(monkeypatch, [_match(1), _match(2)], visibility=vis)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_receiver_model.py", "--out", str(tmp_path / "out"), "--shard-root", str(tmp_path / "sh"),
         "--allow-dirty", "--min-rows", "1", "--min-passes", "1"],
    )
    TRM.main()
    man = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert man["corpus_visibility"] == want and man["corpus_n_matches"] == 2
```

  **Update the existing test seam in the same step** (B r4, outside its round: after Step 4, `main()` calls `select_match_ids` and `match_visibility`, so every test that reaches `main()` would otherwise hit the LIVE pining API — five did, invisibly, on any networked runner):
  - `_install_corpus(monkeypatch, matches, *, visibility="public")` gains the keyword and, after its existing `setattr`s, adds `monkeypatch.setattr(lp, "match_visibility", lambda providers, **kw: {k: visibility for k in by_key})`. `select_match_ids` already resolves through the patched `lp.list_match_refs`. The default `"public"` keeps every existing assertion's meaning.
  - `test_owner_rows_skips_the_training_reparse` does not use the seam. Add, beside its `load_matches` patch: `monkeypatch.setattr("scripts._loader_pining.list_match_refs", lambda **kw: [])` and `monkeypatch.setattr("scripts._loader_pining.match_visibility", lambda providers, **kw: {})`. Its corpus label is then `restricted` (fail-closed, empty), which that test does not assert on.

- [ ] **Step 3: Run — expect FAIL** (ImportError `_corpus_visibility`; `TypeError` on the `match_ids` kwarg; the new parametrized test fails on `corpus_visibility`).
- [ ] **Step 4: Implement.**

(a) `_corpus_source` signature and body:

```python
def _corpus_source(provider: str, cache_dir, match_ids: dict | None = None):
    ...  # docstring unchanged
    from scripts._loader_pining import pining_source

    refs, base_load = pining_source([provider], cache_dir=cache_dir, match_ids=match_ids)
```

(b) Thread `match_ids` through `_extract_provider_rows(provider, feature_set, shard_root, cache_dir, out_path, tag, match_ids=None)` into its `_corpus_source(provider, cache_dir, match_ids)` call. The deployment caller `:387` keeps `None`.

(c) Add the helper above `main`:

```python
def _corpus_visibility(pairs, visibility: dict) -> str:
    """ADR-038: visibility from the MANIFEST, never the provider name (the rule this trainer kept at :529
    after ADR-038 deleted it elsewhere). Fail-closed: empty or any non-public match -> restricted."""
    from scripts._corpus import requested_is_all_public

    return "public" if requested_is_all_public(list(pairs), visibility) else "restricted"
```

(d) argparse, after `--cache-dir`:

```python
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"statsbomb": [id, ...]} restricting the PRIMARY provider\'s corpus (e.g. the 30-match '
        "fallback re-fit, combined-cycle spec 7). Default: the whole manifest.",
    )
```

Pass `match_ids=json.loads(pathlib.Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None` into the PRIMARY `_extract_provider_rows(...)` call only. (The module imports `pathlib`, not `Path`: B r4 found the F821.)

(e) Replace the label line `corpus_label = artifact_label(providers=providers, all_public=providers.issubset({"statsbomb"}))` with:

```python
    from scripts._corpus import reproducibility
    from scripts._loader_pining import match_visibility, select_match_ids

    _primary_ids = json.loads(pathlib.Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    _pairs = list(select_match_ids(providers=[args.provider], match_ids=_primary_ids))
    if args.pool_provider and args.pool_provider in providers:  # the pool earned inclusion -> its matches count too
        _pairs += list(select_match_ids(providers=[args.pool_provider]))
    corpus_label = _corpus_visibility(_pairs, match_visibility(sorted(providers)))
```

and add to the `manifest` dict, after `"corpus_visibility": corpus_label,`:

```python
        "corpus_n_matches": len(_pairs),
        **reproducibility("public" if corpus_label == "public" else "restricted", sorted(providers), training_commit=prov["commit"]),
```

Remove the now-unused `artifact_label` import only if nothing else in the file uses it.

- [ ] **Step 5: Run — expect PASS, with the network sandboxed** so a live pining call fails instead of passing silently (B r4): `env -u PINING_FOR_THE_DATA_TOKEN PINING_API_URL=https://127.0.0.1:9 python -m pytest tests/scripts/test_train_receiver_model.py tests/tracking/test_receiver*.py tests/scripts/test_provenance_wiring.py -q`. Any `URLError` means a seam was missed: fix the seam, never the assertion. (`_loader_pining._base_url()` reads `PINING_API_URL`, `:68`.)

### Task 6: Receiver widening gate driver

**Files:** Create `scripts/validate_receiver_widening.py` and `tests/scripts/test_receiver_widening_driver.py`. Register in `tests/scripts/test_provenance_wiring.py` `ARTIFACT_DRIVERS` (follow that file's registration instructions exactly; read it first).

**Interfaces (produced):**

| Name | Signature |
|---|---|
| `per_pass_hits` | `(model, rows) -> DataFrame[game_id, action_id, hit]` |
| `identify` | `(rows, candidate_ids, committed, *, seed=0, n_controls=20) -> (report: dict, exclude: list[str])` |
| `gate` | `(rows, committed, exclude, *, n_splits=5, seed=0, n_boot=2000) -> dict` |

| `run` | `(rows_path, out, *, prov, n_boot=2000, n_controls=20) -> dict` |

CLI `main(argv=None)`: `--rows <candidate_rows.parquet> --out <dir> [--allow-dirty]`. `main()` itself calls `require_clean_tree` (the provenance entry-point gate). The first-30 ids come from the statsbomb manifest in-process (`_statsbomb_first30`), never from a file. It writes `<out>/receiver_gate.json` and never records a match id.

- [ ] **Step 1: Failing tests** — `tests/scripts/test_receiver_widening_driver.py`:

```python
"""The receiver widening gate (combined-cycle-completion spec section 7): scoring rule, exclusions,
identification with its negative control, decision rule, no ids in the output."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import validate_receiver_widening as G
from silly_kicks.tracking._receiver import ReceiverModel


def _rows(n_games=12, passes=30, seed=0):
    """Synthetic candidate rows in the trainer's public schema: 4 candidates per pass, one labelled."""
    from scripts.train_receiver_model import _feature_names

    rng = np.random.default_rng(seed)
    names = _feature_names("public")
    recs = []
    for g in range(n_games):
        for a in range(passes):
            for c in range(4):
                feats = rng.standard_normal(len(names))
                recs.append({"game_id": f"statsbomb:{g}", "action_id": a, "label": int(c == 0), **dict(zip(names, feats + (1.5 if c == 0 else 0.0), strict=True))})
    return pd.DataFrame(recs)


def test_per_pass_hits_matches_the_trainers_top1():
    from scripts.train_receiver_model import _feature_names, _top1_accuracy

    rows = _rows()
    m = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    hits = G.per_pass_hits(m, rows)
    assert abs(hits["hit"].mean() - _top1_accuracy(m, rows, "public")) < 1e-12
    assert len(hits) == rows.groupby(["game_id", "action_id"]).ngroups


def test_identical_models_give_zero_difference_and_the_q3_rule_passes():
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    out = G.gate(rows, committed, exclude=[], n_boot=200)
    assert out["n_test_matches"] == 12
    assert out["decision_rule"] == G.DECISION_RULE
    # new is a per-fold refit, old the full fit: both near-perfect on this separable fixture
    assert abs(out["diff_new_minus_old"]) < 0.05


def test_exclusions_are_removed_from_every_test_fold():
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    out = G.gate(rows, committed, exclude=["statsbomb:0", "statsbomb:1"], n_boot=50)
    assert out["n_test_matches"] == 10 and out["excluded_from_test"] == 2


def test_identification_succeeds_on_the_true_training_set_and_its_negative_control_fails():
    from scripts.train_receiver_model import _feature_names

    rows = _rows(n_games=24, seed=1)
    true_ids = [f"statsbomb:{g}" for g in range(6)]
    sub = rows[rows["game_id"].isin(true_ids)]
    committed = ReceiverModel("public").fit(sub[_feature_names("public")], sub["label"])
    G_ID_TOP1 = G.cv_top1(sub, "public")[0]
    report, exclude = G.identify(rows, [i.split(":")[1] for i in true_ids], committed, expected_top1=G_ID_TOP1, n_controls=5)
    assert report["identified"] is True and sorted(exclude) == sorted(true_ids)
    assert report["n_controls_passing"] == 0  # random 6-match subsets must not reproduce the committed fit


def test_output_never_records_a_match_id(tmp_path, monkeypatch):
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    raw = {f"statsbomb:{g}": f"38570{g:02d}" for g in range(12)}  # distinctive 7-digit ids, as SB360 uses
    rows.assign(game_id=rows["game_id"].map(raw)).to_parquet(tmp_path / "rows.parquet")  # raw ids, as the trainer writes
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    monkeypatch.setattr(G, "_committed_model", lambda: committed)
    monkeypatch.setattr(G, "_statsbomb_first30", lambda: [raw[f"statsbomb:{g}"] for g in range(3)])
    prov = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}
    G.run(tmp_path / "rows.parquet", tmp_path / "out", prov=prov, n_boot=50, n_controls=2)
    text = (tmp_path / "out" / "receiver_gate.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["run_commit"] == "0" * 40 and doc["run_tree_dirty"] is False
    assert "statsbomb:" not in text
    assert not any(mid in text for mid in raw.values())


@pytest.mark.parametrize(
    ("diff", "lb", "want"),
    [
        (0.0, -0.005, True),  # point exactly 0 passes (>=), LB inside the margin
        (0.004, -0.0099, True),  # LB just inside -0.01
        (0.004, float(np.nextafter(-0.01, 0.0)), True),  # the closest LB inside the bound
        (0.004, -0.01, False),  # LB AT -0.01: the bound is strict (B r4 CCC-PLAN-30)
        (0.004, -0.0101, False),  # LB just outside -0.01 (noisy pass blocked)
        (-0.001, -0.005, False),  # point estimate negative (Q3 rule)
        (float(np.nextafter(0.0, -1.0)), 0.002, False),  # the closest negative point fails
        (0.01, 0.002, True),
        (0.01, float("nan"), False),  # an undefined bound never ships
    ],
)
def test_decision_rule_combined(diff, lb, want):
    assert G.decide(diff, lb) is want  # D8 (owner-approved): point >= 0 AND 95% LB > -0.01


def test_the_d8_margin_and_rule_label_are_pinned():
    assert G.MARGIN == 0.01 and G.DECISION_RULE == "point_estimate_ge_0_and_boot_lb95_gt_-0.01"


def test_identification_fails_when_the_negative_control_stops_discriminating(monkeypatch):
    """B r4 CCC-PLAN-30: if random subsets reproduce the committed fit too, the candidate proves nothing."""
    calls = []

    def always(rows, ids, committed, expected_top1):
        calls.append(tuple(ids))
        return True, {"n_present": len(ids)}

    monkeypatch.setattr(G, "_reproduces", always)
    rows = pd.DataFrame({"game_id": [f"statsbomb:{g}" for g in range(12)]})
    report, exclude = G.identify(rows, ["0", "1", "2"], committed=None, n_controls=4)
    assert len(calls) == 1 + 4  # the candidate plus every control was actually tried
    assert report["candidate_reproduces"] is True and report["n_controls_passing"] == 4
    assert report["identified"] is False and exclude == []


def test_identification_fails_when_the_candidate_does_not_reproduce(monkeypatch):
    monkeypatch.setattr(G, "_reproduces", lambda rows, ids, committed, expected_top1: (False, {"n_present": len(ids)}))
    rows = pd.DataFrame({"game_id": [f"statsbomb:{g}" for g in range(12)]})
    report, exclude = G.identify(rows, ["0", "1", "2"], committed=None, n_controls=3)
    assert report["candidate_reproduces"] is False and report["identified"] is False and exclude == []


def test_the_wrong_match_set_does_not_reproduce_the_committed_fit():
    """Real reproduction (no stub): the committed model was fit on games 0-5; games 6-11 must not identify."""
    from scripts.train_receiver_model import _feature_names

    rows = _rows(n_games=24, seed=1)
    true_ids = [f"statsbomb:{g}" for g in range(6)]
    sub = rows[rows["game_id"].isin(true_ids)]
    committed = ReceiverModel("public").fit(sub[_feature_names("public")], sub["label"])
    top1 = G.cv_top1(sub, "public")[0]
    report, exclude = G.identify(rows, [str(g) for g in range(6, 12)], committed, expected_top1=top1, n_controls=2)
    assert report["candidate_reproduces"] is False and report["identified"] is False and exclude == []
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
- [ ] **Step 3: Implement** `scripts/validate_receiver_widening.py` (ASCII only):

```python
"""Receiver 30 -> 327 widening gate -- PRE-REGISTERED (combined-cycle-completion spec section 7).

Run at C1 on a clean tree:
    python scripts/validate_receiver_widening.py --rows <candidate_rows.parquet> --out <dir>
Step 1 identifies the committed model's 30 training matches (the first 30 of the statsbomb manifest,
verified by refit reproduction, with a negative control). Step 2 compares a fresh per-fold refit on the
327 rows against the committed model on identical held-out passes. The output carries provenance and
aggregate numbers only -- never a match id (statsbomb is manifest-private, ADR-062).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.train_receiver_model import _feature_names, _namespace_game_ids, cv_top1
from silly_kicks.tracking._receiver import ReceiverModel

FEATURE_SET = "public"
_COMMITTED = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking" / "_receiver_weights" / "default"
ID_TOP1, ID_TOP1_TOL, ID_PARAM_RTOL = 0.5097663185813329, 0.005, 1e-2
#: D8 (owner-approved 2026-10-02): the trainer's own Q3 data-earns-inclusion rule (pooling_gate: pooled >=
#: primary) AND a non-inferiority bound that stops a noisy pass. MARGIN = 1 percentage point of top-1: below
#: the committed model's fold-to-fold SD (~0.013) and near the margin that rejected the GS pool (-0.009).
MARGIN = 0.01
DECISION_RULE = "point_estimate_ge_0_and_boot_lb95_gt_-0.01"


def decide(diff: float, lb: float) -> bool:
    """Ship iff the point estimate new - old is >= 0 AND the 95% bootstrap lower bound is > -MARGIN."""
    return diff >= 0.0 and lb > -MARGIN


def per_pass_hits(model, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per pass: ``hit`` = the argmax candidate is the labelled receiver (the trainer's rule)."""
    names = _feature_names(FEATURE_SET)
    test = rows.reset_index(drop=True).copy()
    test["_p"] = model.predict_candidates(test[names])
    out = [
        (g, a, int(grp.loc[grp["_p"].idxmax(), "label"] == 1))
        for (g, a), grp in test.groupby(["game_id", "action_id"])
    ]
    return pd.DataFrame(out, columns=["game_id", "action_id", "hit"])


def _params(m: ReceiverModel) -> np.ndarray:
    return np.concatenate([np.ravel(m._coef), np.ravel(m._intercept), np.ravel(m._mean), np.ravel(m._std)])


def _reproduces(rows: pd.DataFrame, ids: list[str], committed: ReceiverModel, expected_top1: float) -> tuple[bool, dict]:
    names = _feature_names(FEATURE_SET)
    sub = rows[rows["game_id"].isin(ids)]
    if sub["game_id"].nunique() != len(ids):
        return False, {"n_present": int(sub["game_id"].nunique())}
    refit = ReceiverModel(FEATURE_SET).fit(sub[names], sub["label"])
    top1, _folds = cv_top1(sub, FEATURE_SET)
    a, b = _params(refit), _params(committed)
    rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
    ok = abs(top1 - expected_top1) <= ID_TOP1_TOL and rel <= ID_PARAM_RTOL
    return ok, {"n_present": len(ids), "refit_top1_cv": float(top1), "max_rel_param_diff": rel}


def identify(rows, candidate_ids, committed, *, expected_top1=ID_TOP1, seed=0, n_controls=20):
    """Step 1 + its negative control. Returns (report, game ids to exclude from every test fold)."""
    ids = [f"statsbomb:{m}" for m in candidate_ids]
    ok, detail = _reproduces(rows, ids, committed, expected_top1)
    rest = sorted(set(rows["game_id"]) - set(ids))
    rng = np.random.default_rng(seed)
    passing = 0
    for _ in range(n_controls):
        ctrl = list(rng.choice(rest, size=len(ids), replace=False)) if len(rest) >= len(ids) else []
        if ctrl and _reproduces(rows, ctrl, committed, expected_top1)[0]:
            passing += 1
    discriminating = passing == 0
    report = {
        "identified": bool(ok and discriminating),
        "candidate_reproduces": bool(ok),
        "n_controls": n_controls,
        "n_controls_passing": passing,
        "criteria": {"top1": expected_top1, "top1_tol": ID_TOP1_TOL, "param_rtol": ID_PARAM_RTOL},
        **detail,
    }
    return report, (ids if report["identified"] else [])


def gate(rows, committed, exclude, *, n_splits=5, seed=0, n_boot=2000) -> dict:
    names = _feature_names(FEATURE_SET)
    news, olds = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(rows, groups=rows["game_id"].to_numpy()):
        train, test = rows.iloc[tr], rows.iloc[te]
        test = test[~test["game_id"].isin(exclude)]
        if test.empty:
            continue
        m = ReceiverModel(FEATURE_SET).fit(train[names], train["label"])
        news.append(per_pass_hits(m, test))
        olds.append(per_pass_hits(committed, test))
    both = pd.concat(news).merge(pd.concat(olds), on=["game_id", "action_id"], suffixes=("_new", "_old"), validate="one_to_one")
    g = both.groupby("game_id").agg(n=("hit_new", "size"), d=("hit_new", "sum"), o=("hit_old", "sum"))
    n, d, o = g["n"].to_numpy(), g["d"].to_numpy(), g["o"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    boots = (d[idx].sum(1) - o[idx].sum(1)) / n[idx].sum(1)
    diff = float((d.sum() - o.sum()) / n.sum())
    lb, ub = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "n_test_matches": int(len(g)),
        "n_test_passes": int(n.sum()),
        "top1_new": float(d.sum() / n.sum()),
        "top1_old": float(o.sum() / n.sum()),
        "diff_new_minus_old": diff,
        "boot_ci_95": [lb, ub],
        "decision_rule": DECISION_RULE,
        "margin": MARGIN,
        "ship": decide(diff, lb),
        "seed": seed,
        "n_boot": n_boot,
        "excluded_from_test": len(exclude),
    }


def _committed_model() -> ReceiverModel:
    return ReceiverModel.load(_COMMITTED)


def _statsbomb_first30() -> list[str]:
    """The first 30 ids of the statsbomb manifest (owner token). Read in-process, never written out."""
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token

    return [str(m["id"]) for m in _list_matches("statsbomb", _resolve_token(None), _base_url())[:30]]


def run(rows_path: Path, out: Path, *, prov: dict, n_boot: int = 2000, n_controls: int = 20) -> dict:
    """The gate over an already-checked provenance (``main`` refuses a dirty tree before calling this)."""
    rows = _namespace_game_ids(pd.read_parquet(rows_path), "statsbomb")
    committed = _committed_model()
    ident, exclude = identify(rows, _statsbomb_first30(), committed, n_controls=n_controls)
    result = {
        "identification": ident,
        "gate": gate(rows, committed, exclude, n_boot=n_boot),
        "inputs_sha256": {
            "rows": hashlib.sha256(Path(rows_path).read_bytes()).hexdigest(),
            "committed_model": hashlib.sha256((_COMMITTED / "model.json").read_bytes()).hexdigest(),
        },
        "n_matches": int(rows["game_id"].nunique()),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "receiver_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    from scripts._provenance import git_provenance, require_clean_tree

    # FIRST, in main() itself: the provenance entry-point gate (test_provenance_wiring) requires this call
    # here, not behind a helper (B r4 CCC-PLAN-29).
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    print(json.dumps(run(args.rows, args.out, prov=prov), indent=2))


if __name__ == "__main__":
    main()
```

`identify` accepts `expected_top1`; the test passes it explicitly. Check `ReceiverModel`'s private attribute names first:

```bash
python -c "from silly_kicks.tracking._receiver import ReceiverModel as R; m=R.load('silly_kicks/tracking/_receiver_weights/default'); print([a for a in vars(m) if a.startswith('_')])"
```

Expect `_coef`, `_intercept`, `_mean`, `_std`; adjust `_params` to the printed names if they differ. Fix the test's last no-id assertion to whatever exact form proves no `game_id` string appears; keep it strict.
- [ ] **Step 4: Run — expect PASS:** `python -m pytest tests/scripts/test_receiver_widening_driver.py tests/scripts/test_provenance_wiring.py -q`, then the ASCII gate `python -m pytest tests/scripts -q -k ascii`.

### Task 6b: Card-only Hub push seam (D9)

**Files:**
- Modify `scripts/_hub_publish.py`: add `import hashlib`; add `CARD_SOURCE`, `card_bytes`, `publish_card_only`, `_hub_readme`, `_sha256`; `publish_model_with_card` stages the card through `card_bytes`.
- Create `scripts/publish_model_card.py` (ASCII only).
- Test: create `tests/scripts/test_publish_model_card.py`; append one test to `tests/scripts/test_hub_publish_guard.py`.
- Docs: ADR-088 amendment section; `docs/context/trained-models.md` (the ADR-088 bullet); `AGENTS.md` (the `publish_model_with_card` Key-conventions bullet).

**Interfaces (produced):**
- `CARD_SOURCE: dict[str, str]` — Hub repo id → repo-relative card path, for all 10 org repos.
- `card_bytes(model_card: str | Path) -> bytes` — the card with LF line endings; `SystemExit` if missing.
- `publish_card_only(api, repo_id: str, *, root: str | Path = ".", verify_only: bool = False) -> dict` — keys `repo_id`, `card`, `card_sha256`, `hub_sha256_before`, `changed`, `uploaded`, and `hub_sha256_after` when uploaded.
- CLI `scripts/publish_model_card.py --repo-id <registered> [--verify-only]`.

Measured 2026-10-02 (anonymous download, CR stripped): 8 of 10 Hub READMEs equal their in-repo card; the two sweeper repos differ (never republished after `adafb72`). `xsuccess-v1`'s README equals `silly_kicks/xsuccess/weights/MODEL_CARD.md`, which has no YAML frontmatter, so the seam does not require frontmatter. All Hub READMEs are LF, while this machine's checkout (`core.autocrlf=true`) holds CRLF cards.

- [ ] **Step 1: Failing tests** — `tests/scripts/test_publish_model_card.py`:

```python
"""The card-only Hub push seam (combined-cycle spec section 9, D9; ADR-088 amendment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._hub_publish import CARD_SOURCE, card_bytes, publish_card_only

_REPO = "silly-kicks/xshot-occurrence-v1"


class _Sibling:
    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


class _Info:
    def __init__(self, files) -> None:
        self.siblings = [_Sibling(f) for f in files]


class _FakeHub:
    """An in-memory Hub repo: model_info lists files, hf_hub_download serves README bytes,
    upload_file replaces them. `corrupt` makes the read-back differ from the upload."""

    def __init__(self, tmp_path: Path, readme: bytes | None, *, exists: bool = True, corrupt: bool = False) -> None:
        self.tmp, self.readme, self.exists, self.corrupt = tmp_path, readme, exists, corrupt
        self.uploads: list[dict] = []
        self.reads: list[str] = []  # every Hub read, so "refused before any network" is assertable
        self.created: list[dict] = []

    def model_info(self, repo_id: str):
        self.reads.append(f"model_info:{repo_id}")
        if not self.exists:
            raise LookupError(f"404 {repo_id}")
        return _Info(["model.json"] + (["README.md"] if self.readme is not None else []))

    def hf_hub_download(self, *, repo_id, filename, repo_type, force_download):
        self.reads.append(f"download:{repo_id}/{filename}")
        assert filename == "README.md" and repo_type == "model" and force_download is True
        assert self.readme is not None
        path = self.tmp / f"dl_{len(self.reads)}.md"
        path.write_bytes(self.readme)
        return str(path)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        self.readme = kwargs["path_or_fileobj"] + (b"X" if self.corrupt else b"")

    def create_repo(self, **kwargs):
        self.created.append(kwargs)  # a card-only push must never call this


def _root(tmp_path: Path, body: bytes) -> Path:
    card = tmp_path / "repo" / CARD_SOURCE[_REPO]
    card.parent.mkdir(parents=True)
    card.write_bytes(body)
    return tmp_path / "repo"


def test_card_source_names_every_org_repo_and_every_card_exists():
    assert len(CARD_SOURCE) == 10
    for repo, rel in CARD_SOURCE.items():
        assert repo.startswith("silly-kicks/") and Path(rel).is_file(), (repo, rel)


def test_card_bytes_normalizes_crlf_and_refuses_a_missing_card(tmp_path):
    p = tmp_path / "c.md"
    p.write_bytes(b"---\r\nlicense: mit\r\n---\r\n# c\r\n")
    assert card_bytes(p) == b"---\nlicense: mit\n---\n# c\n"
    with pytest.raises(SystemExit, match="does not exist"):
        card_bytes(tmp_path / "nope.md")


def test_unregistered_repo_is_refused_before_any_network(tmp_path):
    hub = _FakeHub(tmp_path, b"x")
    with pytest.raises(SystemExit, match="not a registered"):
        publish_card_only(hub, "silly-kicks/not-a-repo", root=tmp_path)
    assert hub.reads == [] and hub.uploads == []  # B r4 CCC-PLAN-34: no Hub read at all


def test_a_missing_registered_card_is_refused_before_any_network(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    with pytest.raises(SystemExit, match="does not exist"):
        publish_card_only(hub, _REPO, root=tmp_path / "empty-root")
    assert hub.reads == [] and hub.uploads == []


def test_hub_sha256_before_is_the_hash_of_the_hub_bytes(tmp_path):
    import hashlib

    hub = _FakeHub(tmp_path, b"# old hub readme\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"), verify_only=True)
    assert out["hub_sha256_before"] == hashlib.sha256(b"# old hub readme\n").hexdigest()
    assert out["card_sha256"] == hashlib.sha256(b"# new\n").hexdigest()


def test_a_push_never_creates_a_repo(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))
    assert hub.created == []


def test_cli_verify_only_reads_and_never_uploads(tmp_path, monkeypatch, capsys):
    import json

    import huggingface_hub

    from scripts import publish_model_card as P

    hub = _FakeHub(tmp_path, b"# a different hub readme\n")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: hub)
    out = P.main(["--repo-id", _REPO, "--verify-only"])  # the real registered card under the repo root
    assert out["changed"] is True and out["uploaded"] is False and hub.uploads == []
    assert json.loads(capsys.readouterr().out)["repo_id"] == _REPO


def test_missing_repo_is_refused_before_any_upload(tmp_path):
    hub = _FakeHub(tmp_path, None, exists=False)
    with pytest.raises(LookupError):
        publish_card_only(hub, _REPO, root=_root(tmp_path, b"# card\n"))
    assert hub.uploads == []


def test_unchanged_card_is_not_reuploaded(tmp_path):
    hub = _FakeHub(tmp_path, b"# card\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# card\r\n"))
    assert out["changed"] is False and out["uploaded"] is False and hub.uploads == []


def test_changed_card_is_uploaded_as_lf_readme_and_read_back(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\r\n"))
    assert out["uploaded"] is True and out["hub_sha256_after"] == out["card_sha256"]
    (up,) = hub.uploads
    assert up["path_in_repo"] == "README.md" and up["repo_id"] == _REPO and up["repo_type"] == "model"
    assert up["path_or_fileobj"] == b"# new\n"


def test_a_repo_without_a_readme_gets_one(tmp_path):
    hub = _FakeHub(tmp_path, None)
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))
    assert out["hub_sha256_before"] is None and out["uploaded"] is True


def test_read_back_mismatch_fails_loud(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n", corrupt=True)
    with pytest.raises(SystemExit, match="READ-BACK MISMATCH"):
        publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"))


def test_verify_only_never_uploads(tmp_path):
    hub = _FakeHub(tmp_path, b"# old\n")
    out = publish_card_only(hub, _REPO, root=_root(tmp_path, b"# new\n"), verify_only=True)
    assert out["changed"] is True and out["uploaded"] is False and hub.uploads == []


def test_cli_offers_only_registered_repos():
    from scripts import publish_model_card as P

    with pytest.raises(SystemExit):
        P.main(["--repo-id", "silly-kicks/not-a-repo", "--verify-only"])
```

Append to `tests/scripts/test_hub_publish_guard.py` (the model publish seam stages the same LF bytes):

```python
def test_publish_model_with_card_stages_the_card_with_lf_line_endings(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    (art / "model.npz").write_bytes(b"\x00")
    card = tmp_path / "the-card.md"
    card.write_bytes(b"---\r\nlicense: mit\r\n---\r\n# card\r\n")
    staged: dict = {}

    class _Capture(_FakeApi):
        def upload_folder(self, **kwargs) -> None:
            staged["readme"] = (Path(kwargs["folder_path"]) / "README.md").read_bytes()

    publish_model_with_card(_Capture(["model.npz", "README.md"]), str(art), "silly-kicks/x", model_card=str(card))
    assert staged["readme"] == b"---\nlicense: mit\n---\n# card\n"
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'CARD_SOURCE'`; the staging test sees CRLF): `python -m pytest tests/scripts/test_publish_model_card.py tests/scripts/test_hub_publish_guard.py -q`.
- [ ] **Step 3: Implement** in `scripts/_hub_publish.py` (add `import hashlib` at the top), after `MODEL_ONLY_ALLOWLIST`:

```python
_CARD_DIR = "docs/huggingface/model-cards"

#: Every Hub repo of the org -> the in-repo card its README must equal (combined-cycle spec 9, D9).
#: The single source for the card-only seam and for validate_hub_variants (whose test asserts these
#: keys equal its HUB_REGISTRY). A card is never taken from a free path, so it cannot reach the wrong repo.
CARD_SOURCE: dict[str, str] = {
    **{
        f"silly-kicks/{name}": f"{_CARD_DIR}/{name}-model-card.md"
        for name in (
            "xshot-occurrence-v1",
            "xshot-occurrence-position-only-v1",
            "xcross-attempt-v1",
            "xcross-attempt-position-only-v1",
            "ghost-gk-v1",
            "ghost-gk-sweeper-v1",
            "ghost-gk-sweeper-position-only-v1",
            "ghost-outfield-v1",
            "ghost-outfield-position-only-v1",
        )
    },
    "silly-kicks/xsuccess-v1": "silly_kicks/xsuccess/weights/MODEL_CARD.md",
}


def card_bytes(model_card: str | Path) -> bytes:
    """The card exactly as the Hub serves it: LF line endings. A Windows checkout with
    core.autocrlf=true holds CRLF; every Hub README is LF (measured 2026-10-02)."""
    card = Path(model_card)
    if not card.is_file():
        raise SystemExit(f"model card {card} does not exist (required for a real publish, uploaded as README.md).")
    return card.read_bytes().replace(b"\r\n", b"\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hub_readme(api: Any, repo_id: str) -> bytes | None:
    """The repo's current README.md bytes (fresh download), or None when it has none."""
    files = {s.rfilename for s in (api.model_info(repo_id).siblings or [])}
    if "README.md" not in files:
        return None
    path = api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", force_download=True)
    return Path(path).read_bytes()


def publish_card_only(api: Any, repo_id: str, *, root: str | Path = ".", verify_only: bool = False) -> dict:
    """The card-only seam (ADR-088 amendment): push the registered card as README.md of an EXISTING
    model repo, then read it back.

    Card-only pushes used to be ad-hoc README uploads (1b56ad8, aae6fdb) with no guard and no read-back.
    Refuses an unregistered repo before any network. ``api.model_info`` raises on a missing repo (a
    card-only push never creates one). An unchanged card is not re-uploaded. After an upload the README is
    re-downloaded and must be byte-identical, else ``SystemExit``. ``verify_only`` reports and stops.
    """
    if repo_id not in CARD_SOURCE:
        raise SystemExit(f"{repo_id} is not a registered Hub repo (CARD_SOURCE) -- register it first.")
    data = card_bytes(Path(root) / CARD_SOURCE[repo_id])
    before = _hub_readme(api, repo_id)
    out = {
        "repo_id": repo_id,
        "card": CARD_SOURCE[repo_id],
        "card_sha256": _sha256(data),
        "hub_sha256_before": None if before is None else _sha256(before),
        "changed": before != data,
        "uploaded": False,
    }
    if verify_only or not out["changed"]:
        return out
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"docs: model card from {CARD_SOURCE[repo_id]}",
    )
    after = _hub_readme(api, repo_id)
    if after != data:
        raise SystemExit(f"CARD READ-BACK MISMATCH: {repo_id} README.md != {CARD_SOURCE[repo_id]} after upload.")
    out.update(uploaded=True, hub_sha256_after=_sha256(after))
    return out
```

  In `publish_model_with_card`, replace the `card = Path(model_card)` / `is_file` check with `data = card_bytes(model_card)`, keeping it BEFORE `create_repo`. Replace `shutil.copy2(card, stage / "README.md")` with `(stage / "README.md").write_bytes(data)`. Update its docstring with one sentence on LF staging.

  Create `scripts/publish_model_card.py`:

```python
#!/usr/bin/env python
"""Push ONE registered model card to its EXISTING Hub repo as README.md (combined-cycle spec 9, D9).

The card-only seam (ADR-088 amendment). The card comes from CARD_SOURCE, never a free path. The repo
must already exist. An unchanged card is not re-uploaded. After an upload the README is read back and
must be byte-identical. --verify-only reports changed / unchanged without uploading.
Run from the repo root of a clean checkout at the release tag:
    PYTHONPATH=. python scripts/publish_model_card.py --repo-id silly-kicks/xshot-occurrence-v1 --verify-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._hub_publish import CARD_SOURCE, publish_card_only

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, choices=sorted(CARD_SOURCE))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    out = publish_card_only(HfApi(), args.repo_id, root=_REPO_ROOT, verify_only=args.verify_only)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Docs.**
  - **ADR-088:** append an "Amendment (combined-cycle completion, 2026-10-02): the card-only seam" section. It covers `publish_card_only` / `publish_model_card.py`, the registered `CARD_SOURCE`, LF staging in both seams, exists-only, no re-upload of an unchanged card, and read-back. The ADR's Status stays Accepted.
  - **`docs/context/trained-models.md`:** extend the ADR-088 bullet with one sentence naming the card-only seam.
  - **`AGENTS.md`:** extend the `publish_model_with_card` bullet with "; a card-only push uses `publish_card_only` (registered card, existing repo, LF, read-back)". Then run `python -m pytest tests/test_agents_md_budget.py -q`. If the per-bullet cap or the byte ceiling fails, STOP and ask the owner; never trim another bullet to make room.
- [ ] **Step 5: Run — expect PASS:** `python -m pytest tests/scripts/test_publish_model_card.py tests/scripts/test_hub_publish_guard.py tests/scripts/test_provenance_wiring.py tests/scripts/test_xsuccess_drivers_smoke.py tests/test_agents_md_budget.py -q`, then the ASCII gate (`-k ascii`). `test_provenance_wiring` must NOT classify `publish_model_card.py` as an artifact driver (it has no `--out` flag and no `_weights` literal). If it does, STOP and report; do not register it silently.

### Task 7: Hub smoke driver

**Files:** Create `scripts/validate_hub_variants.py` and `tests/scripts/test_hub_variants_driver.py`. Register in `ARTIFACT_DRIVERS`. Modify `tests/scripts/_corpus_load_rules.py` (`_UNSHARDED_LOOP_EXEMPT`, Step 3b).

**Interfaces:**
- Consumes (Task 6b): `scripts._hub_publish.CARD_SOURCE`, `card_bytes(path) -> bytes`.
- Produces:
  - `HUB_REGISTRY: dict[str, tuple[str, str]]` — repo → (class name in `silly_kicks.tracking`, role); role ∈ {`hf_only`, `mirror`, `event_only`}. 5 / 4 / 1.
  - `check_population(listed: set[str]) -> None`.
  - `smoke_repo(repo_id, cls_name, *, from_hub, score, refusals=()) -> dict` — a load raising one of `refusals` returns `{loaded: False, load_error: "<Type>: <first line>", n_scores: 0, finite: False}`; any other exception propagates.
  - `_load_refusals() -> tuple[type[BaseException], ...]` — the frame-geometry classes' fail-closed `IntegrityError`s (`_ghost_gk`, `_ghost_outfield`, `_xshot_occurrence`; xcross reuses xshot's).
  - `readme_matches_card(repo_id, *, download, root=Path(".")) -> bool`.
  - `MIRROR_BUNDLE: dict[str, str]` — mirror repo → wheel bundle dir.
  - `run(*, listed, revision, download, load_model, score_for, prov, root, out, load_refusals=(), require_cards_match=False, require_mirrors_match_wheel=False) -> dict` — refused repos are listed in `load_refused` and excluded from `all_finite`; a refused `hf_only` repo always raises `SystemExit`; refused mirrors raise only under `require_mirrors_match_wheel` — the whole smoke over injected Hub access (offline-testable); `_live_run(**kw)` binds the live, anonymous Hub.
  - CLI `main(argv=None)`: `--out <dir> [--allow-dirty] [--require-cards-match] [--require-mirrors-match-wheel]`; writes `hub_smoke.json` with per-repo `readme_matches_card`, `hub_training_commit` (+ `wheel_training_commit`, `mirror_matches_wheel` for mirrors) and top-level `cards_mismatched` / `mirrors_mismatched` / `load_refused` lists.

**Execution amendment (owner-approved 2026-10-02, implementation review B M-1).** The live smoke measured both sweeper mirrors refused on chirality (`adafb72`, pre-ADR-089); the planned driver crashed on the first one. The refusal handling above was added test-first: `test_smoke_records_a_fail_closed_load_refusal_instead_of_crashing`, `test_smoke_does_not_swallow_an_unexpected_error`, `test_a_refused_mirror_is_recorded_at_c1_and_gated_after_the_push`, `test_a_refused_hub_only_repo_always_fails` (each mutation-checked).

- [ ] **Step 1: Failing tests:**

```python
"""Hub smoke (combined-cycle-completion spec section 9): exact population, fail-closed load, finite scores."""

import math

import pytest

from scripts import validate_hub_variants as H


def test_registry_covers_the_org_exactly():
    listed = set(H.HUB_REGISTRY)
    assert H.check_population(listed) is None
    with pytest.raises(SystemExit, match="unregistered"):
        H.check_population(listed | {"silly-kicks/new-model-v1"})
    with pytest.raises(SystemExit, match="missing"):
        H.check_population(listed - {"silly-kicks/ghost-gk-v1"})


def test_registry_classifies_nine_frame_geometry_repos():
    roles = [role for _cls, role in H.HUB_REGISTRY.values()]
    assert len(H.HUB_REGISTRY) == 10
    # spec 0.10 (rev 4 correction): ghost-gk-v1 is the Hub-only `full` variant, NOT a mirror.
    assert (roles.count("hf_only"), roles.count("mirror"), roles.count("event_only")) == (5, 4, 1)
    assert H.HUB_REGISTRY["silly-kicks/ghost-gk-v1"][1] == "hf_only"


def test_card_source_covers_the_registry_exactly():
    from scripts._hub_publish import CARD_SOURCE

    assert set(CARD_SOURCE) == set(H.HUB_REGISTRY)


def test_readme_matches_card_is_line_ending_blind_and_content_exact(tmp_path):
    from scripts._hub_publish import CARD_SOURCE

    repo = "silly-kicks/xsuccess-v1"
    card = tmp_path / CARD_SOURCE[repo]
    card.parent.mkdir(parents=True)
    card.write_bytes(b"# card\r\nline\r\n")  # a Windows (core.autocrlf=true) checkout
    hub = tmp_path / "hub_README.md"
    hub.write_bytes(b"# card\nline\n")
    assert H.readme_matches_card(repo, download=lambda r, f: str(hub), root=tmp_path) is True
    hub.write_bytes(b"# card\nother\n")
    assert H.readme_matches_card(repo, download=lambda r, f: str(hub), root=tmp_path) is False


def test_driver_frame_equals_the_test_frame():
    from tests.test_bundled_models_load_on_float32_commit1 import _float32_canonical_frame

    import pandas as pd

    a, b = H._float32_canonical_frame(), _float32_canonical_frame()
    pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])


@pytest.mark.parametrize("cls_name", ["XShotOccurrenceModel", "XCrossAttemptModel", "GhostGkModel", "GhostOutfieldModel"])
def test_score_fn_runs_on_the_bundled_variants(cls_name):
    """Offline (no network): the exact serve lambdas the Hub smoke uses score the bundled `default`
    variant of each class to non-empty finite values (B r2 C1: a wrong signature fails CI, not the DGX)."""
    import silly_kicks.tracking as T

    vals = [float(v) for v in H._score_fn(cls_name)(getattr(T, cls_name).from_variant("default"))]
    assert vals and all(math.isfinite(v) for v in vals)


def test_smoke_reports_non_finite_scores_as_failure():
    class FakeModel:
        training_commit = "abc"

    def fake_from_hub(repo_id):
        return FakeModel()

    out = H.smoke_repo("silly-kicks/x", "Fake", from_hub=fake_from_hub, score=lambda m: [float("nan")])
    assert out["loaded"] is True and out["finite"] is False


def _fake_world(tmp_path, *, stale_card=None, stale_mirror=None):
    """An offline Hub + repo root: every README equals its card and every mirror equals the wheel, except
    the one repo named stale (B r4 CCC-PLAN-32: the post-push gates must be tested both ways)."""
    import json

    from scripts._hub_publish import CARD_SOURCE

    hub, root = tmp_path / "hub", tmp_path / "root"
    for repo, rel in CARD_SOURCE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"# card " + repo.encode() + b"\n")
        d = hub / repo.replace("/", "__")
        d.mkdir(parents=True)
        (d / "README.md").write_bytes(b"# card " + repo.encode() + (b" STALE" if repo == stale_card else b"") + b"\n")
        (d / "metadata.json").write_text(json.dumps({"training_commit": ("f" if repo == stale_mirror else "c") * 40}))
    for rel in H.MIRROR_BUNDLE.values():
        m = root / "silly_kicks" / "tracking" / rel
        m.mkdir(parents=True)
        (m / "metadata.json").write_text(json.dumps({"training_commit": "c" * 40}))
    return {
        "listed": set(H.HUB_REGISTRY),
        "revision": lambda r: "rev",
        "download": lambda r, f: str(hub / r.replace("/", "__") / f),
        "load_model": lambda cls_name, repo_id: object(),
        "score_for": lambda cls_name: (lambda m: [0.5]),
        "prov": {"commit": "0" * 40, "dirty": False, "tree_state": "clean"},
        "root": root,
        "out": tmp_path / "out",
    }


def test_post_push_gates_pass_when_every_readme_and_mirror_matches(tmp_path):
    doc = H.run(**_fake_world(tmp_path), require_cards_match=True, require_mirrors_match_wheel=True)
    assert doc["cards_mismatched"] == [] and doc["mirrors_mismatched"] == [] and doc["all_finite"] is True
    assert (tmp_path / "out" / "hub_smoke.json").is_file()


def test_require_cards_match_fails_on_one_stale_readme(tmp_path):
    kw = _fake_world(tmp_path, stale_card="silly-kicks/xshot-occurrence-v1")
    assert H.run(**kw)["cards_mismatched"] == ["silly-kicks/xshot-occurrence-v1"]  # recorded, not gated, at C1
    with pytest.raises(SystemExit, match="differs from its in-repo card"):
        H.run(**kw, require_cards_match=True)


def test_require_mirrors_match_wheel_fails_on_a_stale_mirror(tmp_path):
    kw = _fake_world(tmp_path, stale_mirror="silly-kicks/ghost-outfield-v1")
    doc = H.run(**kw)
    assert doc["mirrors_mismatched"] == ["silly-kicks/ghost-outfield-v1"]
    assert doc["repos"]["silly-kicks/ghost-outfield-v1"]["wheel_training_commit"] == "c" * 40
    with pytest.raises(SystemExit, match="training_commit differs from the wheel"):
        H.run(**kw, require_mirrors_match_wheel=True)


def test_mirror_bundles_name_exactly_the_mirror_repos():
    assert set(H.MIRROR_BUNDLE) == {r for r, (_c, role) in H.HUB_REGISTRY.items() if role == "mirror"}


def test_main_threads_the_post_push_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(H, "_live_run", lambda **kw: seen.update(kw) or {})
    H.main(["--out", str(tmp_path), "--allow-dirty", "--require-cards-match", "--require-mirrors-match-wheel"])
    assert seen["require_cards_match"] is True and seen["require_mirrors_match_wheel"] is True
    assert seen["out"] == tmp_path and seen["prov"]["commit"]
```

- [ ] **Step 2: Run — expect FAIL** (ImportError).
- [ ] **Step 3: Implement** (ASCII only):

```python
"""Hub smoke over the org's whole frame-geometry population (combined-cycle-completion spec section 9).

Anonymous downloads only (no token): each repo loads fail-closed (chirality + feature contract inside
load) and scores the canonical float32 frame to finite values. Publishes nothing.
    python scripts/validate_hub_variants.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORG = "silly-kicks"
#: repo -> (silly_kicks.tracking class, role). role: hf_only | mirror (a wheel variant republished) | event_only.
HUB_REGISTRY: dict[str, tuple[str, str]] = {
    "silly-kicks/xshot-occurrence-v1": ("XShotOccurrenceModel", "hf_only"),
    "silly-kicks/xshot-occurrence-position-only-v1": ("XShotOccurrenceModel", "hf_only"),
    "silly-kicks/xcross-attempt-v1": ("XCrossAttemptModel", "hf_only"),
    "silly-kicks/xcross-attempt-position-only-v1": ("XCrossAttemptModel", "hf_only"),
    "silly-kicks/ghost-gk-v1": ("GhostGkModel", "hf_only"),  # the Hub-only `full` variant (spec 0.10)
    "silly-kicks/ghost-gk-sweeper-v1": ("GhostGkModel", "mirror"),
    "silly-kicks/ghost-gk-sweeper-position-only-v1": ("GhostGkModel", "mirror"),
    "silly-kicks/ghost-outfield-v1": ("GhostOutfieldModel", "mirror"),
    "silly-kicks/ghost-outfield-position-only-v1": ("GhostOutfieldModel", "mirror"),
    "silly-kicks/xsuccess-v1": ("", "event_only"),
}


def check_population(listed: set[str]) -> None:
    """ADR-056: the org listing must equal the registry exactly."""
    unregistered, missing = sorted(listed - set(HUB_REGISTRY)), sorted(set(HUB_REGISTRY) - listed)
    if unregistered:
        raise SystemExit(f"unregistered Hub repo(s): {unregistered} -- classify them in HUB_REGISTRY")
    if missing:
        raise SystemExit(f"missing Hub repo(s): {missing} -- the registry names repos the org no longer lists")


def readme_matches_card(repo_id: str, *, download, root: Path = Path(".")) -> bool:
    """D9: the Hub README equals the registered in-repo card, both LF-normalized (spec section 9)."""
    from scripts._hub_publish import CARD_SOURCE, card_bytes

    hub = Path(download(repo_id, "README.md")).read_bytes().replace(b"\r\n", b"\n")
    return hub == card_bytes(Path(root) / CARD_SOURCE[repo_id])


def smoke_repo(repo_id: str, cls_name: str, *, from_hub, score) -> dict:
    model = from_hub(repo_id)
    vals = [float(v) for v in score(model)]
    return {
        "loaded": True,
        "n_scores": len(vals),
        "finite": bool(vals) and all(math.isfinite(v) for v in vals),
        "training_commit": getattr(model, "training_commit", None),
        "class": cls_name,
    }


def _float32_canonical_frame():
    """The driver's own copy of tests/test_bundled_models_load_on_float32_commit1.py::_float32_canonical_frame
    (scripts must not import tests -- the same rule validate_das_native_parity._run_native follows).
    test_hub_variants_driver asserts the two stay byte-identical."""
    import numpy as np
    import pandas as pd

    rows = [
        dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=20.0, y=34.0),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0),
        dict(player_id=11, team_id=1, is_ball=False, is_goalkeeper=False, x=10.0, y=30.0),
        dict(player_id=12, team_id=1, is_ball=False, is_goalkeeper=False, x=12.0, y=38.0),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0),
        dict(player_id=21, team_id=2, is_ball=False, is_goalkeeper=False, x=20.3, y=34.0),
        dict(player_id=22, team_id=2, is_ball=False, is_goalkeeper=False, x=25.0, y=30.0),
    ]
    df = pd.DataFrame(rows)
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    df["time_seconds"], df["frame_rate"] = 0.0, 25.0
    for c in ("x", "y"):
        df[c] = df[c].astype("float32")
    for c in ("z", "vx", "vy", "speed"):
        df[c] = np.float32(0.0)
    df["ball_state"] = "alive"
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64").astype("category")
    return df


def _score_fn(cls_name: str):
    """The model's own serve path on the canonical float32 frame."""
    import silly_kicks.tracking as T

    frame = _float32_canonical_frame()
    def _xy(out):  # the served ghost coordinates only (never ids/frame keys, which are always finite)
        return out.filter(regex=r"^ghost_.*_[xy]$").stack().dropna()

    if cls_name == "XShotOccurrenceModel":
        return lambda m: T.compute_xshot_occurrence(frame, model=m, home_team_id=1)["xshot_occurrence"].dropna()
    if cls_name == "XCrossAttemptModel":
        return lambda m: T.compute_xcross_attempt(frame, model=m, home_team_id=1)["xcross_attempt"].dropna()
    if cls_name == "GhostGkModel":
        return lambda m: _xy(T.serve_ghost_gk_positions(frame, model=m, home_team_id=1))
    return lambda m: _xy(T.serve_ghost_outfield_positions(frame, model=m, home_team_id=1))


#: mirror repo -> the wheel bundle dir it republishes (spec 0.10, D9). The post-push gate compares their
#: training_commit (spec 9 step 4).
MIRROR_BUNDLE: dict[str, str] = {
    "silly-kicks/ghost-gk-sweeper-v1": "_ghost_gk_weights/sweeper",
    "silly-kicks/ghost-gk-sweeper-position-only-v1": "_ghost_gk_weights/sweeper_position_only",
    "silly-kicks/ghost-outfield-v1": "_ghost_outfield_weights/default",
    "silly-kicks/ghost-outfield-position-only-v1": "_ghost_outfield_weights/position_only",
}
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _training_commit(path) -> str | None:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("training_commit")


def run(
    *,
    listed: set[str],
    revision,
    download,
    load_model,
    score_for,
    prov: dict,
    root: Path,
    out: Path,
    require_cards_match: bool = False,
    require_mirrors_match_wheel: bool = False,
) -> dict:
    """The whole smoke over injected Hub access (offline-testable). Writes ``out/hub_smoke.json``; the two
    ``require_*`` flags turn the recorded card / mirror comparisons into gates (the post-push D9 check)."""
    check_population(listed)
    results: dict[str, dict] = {}
    for repo_id, (cls_name, role) in sorted(HUB_REGISTRY.items()):
        rec: dict = {"role": role, "readme_matches_card": readme_matches_card(repo_id, download=download, root=root)}
        if role == "event_only":
            results[repo_id] = {**rec, "skipped": "event-only model: no frame geometry"}
            continue
        rec["revision"] = revision(repo_id)
        rec["hub_training_commit"] = _training_commit(download(repo_id, "metadata.json"))
        if role == "mirror":
            rec["wheel_training_commit"] = _training_commit(
                Path(root) / "silly_kicks" / "tracking" / MIRROR_BUNDLE[repo_id] / "metadata.json"
            )
            rec["mirror_matches_wheel"] = rec["hub_training_commit"] == rec["wheel_training_commit"]
        results[repo_id] = {
            **rec,
            **smoke_repo(repo_id, cls_name, from_hub=lambda r, c=cls_name: load_model(c, r), score=score_for(cls_name)),
        }
    ok = all(r.get("finite", True) for r in results.values())
    cards_mismatched = sorted(r for r, v in results.items() if not v["readme_matches_card"])
    mirrors_mismatched = sorted(r for r, v in results.items() if v.get("mirror_matches_wheel") is False)
    doc = {
        "org": ORG,
        "repos": results,
        "all_finite": ok,
        "cards_mismatched": cards_mismatched,
        "mirrors_mismatched": mirrors_mismatched,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "hub_smoke.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit("a Hub variant did not score finite values -- see hub_smoke.json")
    if require_cards_match and cards_mismatched:
        raise SystemExit(f"Hub README differs from its in-repo card for {cards_mismatched} -- see hub_smoke.json")
    if require_mirrors_match_wheel and mirrors_mismatched:
        raise SystemExit(f"mirror training_commit differs from the wheel for {mirrors_mismatched} -- see hub_smoke.json")
    return doc


def _live_run(**kw) -> dict:
    """``run`` against the live Hub, ANONYMOUSLY (B r4 CCC-PLAN-35): ``from_hub`` calls snapshot_download
    without a token argument, so the implicit login is disabled before huggingface_hub is imported.
    Otherwise a gated or private repo would load under the operator's login and read as public."""
    import os

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # noqa: S105 -- an env FLAG name, not a secret
    os.environ.pop("HF_TOKEN", None)
    from huggingface_hub import HfApi, hf_hub_download

    import silly_kicks.tracking as T

    api = HfApi()
    return run(
        listed={m.id for m in api.list_models(author=ORG, token=False)},
        revision=lambda repo_id: api.model_info(repo_id, token=False).sha,
        download=lambda repo_id, f: hf_hub_download(repo_id, f, token=False, force_download=True),
        load_model=lambda cls_name, repo_id: getattr(T, cls_name).from_hub(repo_id),
        score_for=_score_fn,
        root=_REPO_ROOT,
        **kw,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--require-cards-match",
        action="store_true",
        help="fail unless every Hub README equals its registered in-repo card (the post-release D9 check)",
    )
    ap.add_argument(
        "--require-mirrors-match-wheel",
        action="store_true",
        help="fail unless every mirror's Hub training_commit equals its wheel bundle's (the post-release D9 check)",
    )
    args = ap.parse_args(argv)

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)  # the entry-point gate
    doc = _live_run(
        prov=prov,
        out=args.out,
        require_cards_match=args.require_cards_match,
        require_mirrors_match_wheel=args.require_mirrors_match_wheel,
    )
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
```

The serve signatures were read 2026-10-02:
- `serve_ghost_gk_positions(frames, *, model, home_team_id, ...)` (`_ghost_gk.py:2826`) and `serve_ghost_outfield_positions(frames, *, model, home_team_id, ...)` (`_ghost_outfield.py:1040`). `home_team_id` is required and keyword-only.
- `compute_xcross_attempt` writes `xcross_attempt`.

- [ ] **Step 3b: Rule C (B r5 CCC-PLAN-41).** `run()` loops over the registered Hub repos and calls its injected `load_model` per repo. The repo's static corpus-loading Rule C (`tests/scripts/_corpus_load_rules.py`) treats any parameter whose name starts with `load` as a corpus loader (`_is_load_call`), so `test_rule_c_ledger_is_exact_both_ways` reports `newly un-sharded: ['validate_hub_variants.run']`. This is a false positive: the loop loads up to 9 published MODELS from the Hub, never a corpus match, and there is nothing to shard. Resolve it with a documented exemption, not a rename: a rename would hide the loop from the rule instead of recording why it is fine. Add to `_UNSHARDED_LOOP_EXEMPT`:

```python
    "validate_hub_variants.run": (
        "loops over the registered Hub repos and loads published MODELS via the injected load_model "
        "(from_hub), never a corpus match -- at most 9 small downloads, nothing to shard (combined-cycle "
        "spec 9)"
    ),
```

  `_RULE_C_PENDING` stays empty. `test_exemptions_and_allowlist_name_functions_that_exist` checks the entry names a real function.

`test_score_fn_runs_on_the_bundled_variants` pins all four lambdas offline. If a serve returns no ghost coordinate for the canonical frame (e.g. it needs `actions`), extend the frame in BOTH copies (driver + `tests/…commit1.py`), never the assertion. Each `from_hub(repo_id)` call must pass the repo id, because the variants live in distinct repos.
- [ ] **Step 4: Run — expect PASS:** `python -m pytest tests/scripts/test_hub_variants_driver.py tests/scripts/test_provenance_wiring.py tests/scripts/test_corpus_driver_resilience.py -q` (whole files, no `-k`: the repo-wide script gates must see the new driver).

### Task 8: T10 sharding (commit-keyed generation, accounted-key completeness)

**Files:** Modify `scripts/measure_f1b_feature_delta.py` (`main` `:480-521`, plus helpers). Test: `tests/scripts/test_measure_f1b_feature_delta.py` (append).

**Interfaces (produced):**
- `_token_inputs(paths, data_dir, commit) -> dict`;
- `_select(paths, data_dir, keys_json)`;
- `_map(paths, data_dir, out, *, token)`;
- `_write_worker_manifest(res, *, prov, worker_tag)`;
- `_artifact(combined, *, n_matches, manifest, prov, dirty, n_accounted)`;
- `reduce_t10(paths, data_dir, out, *, prov) -> dict`;
- CLI `--list-match-keys`, `--match-keys-json`, `--shards-only --worker-tag`, `--reduce-only`.

`metrics.json` keeps the serial schema plus an additive `n_accounted`.

- [ ] **Step 1: Failing tests** (append):

```python
def _write_tc3_match(cache, gid: int, *, frame0: int) -> None:
    from silly_kicks.spadl import config as spc
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    frames = pd.concat(
        [
            _make_ghost_gk_frames(frame_id=frame0, timestamp=float(frame0)),
            _make_ghost_gk_frames(frame_id=frame0 + 1, timestamp=float(frame0 + 1)),
        ],
        ignore_index=True,
    )
    frames.to_parquet(cache / "shards" / "tok" / f"gradientsports__{gid}.parquet")
    pd.DataFrame(
        {
            "game_id": [str(gid)],
            "period_id": [1],
            "team_id": [2],
            "time_seconds": [float(frame0)],
            "type_id": [spc.actiontype_id["pass"]],
            "result_id": [spc.result_id["success"]],
        }
    ).to_parquet(cache / "_actions" / f"gradientsports__{gid}.parquet")
    (cache / "_home" / f"gradientsports__{gid}.json").write_text(json.dumps({"home_team_id": 1}))


def _tc3_cache(tmp_path):
    cache = tmp_path / "cache"
    (cache / "shards" / "tok").mkdir(parents=True)
    (cache / "_actions").mkdir()
    (cache / "_home").mkdir()
    _write_tc3_match(cache, 100, frame0=1)
    _write_tc3_match(cache, 101, frame0=3)
    return cache


def _t10(monkeypatch, *args):
    from scripts.measure_f1b_feature_delta import main

    monkeypatch.setattr(sys, "argv", ["measure_f1b_feature_delta.py", *map(str, args)])
    main()


_KEYS = ["shards__tok__gradientsports__100", "shards__tok__gradientsports__101"]


def _worker(monkeypatch, tmp_path, cache, out, i, keys):
    kj = tmp_path / f"k{i}.json"
    kj.write_text(json.dumps(keys))
    _t10(monkeypatch, "--data-dir", cache, "--out", out, "--shards-only", "--worker-tag", f"w{i}",
         "--match-keys-json", kj, "--allow-dirty")


def _assert_same_artifact(serial, sharded):
    s = json.loads((serial / "metrics.json").read_text())
    p = json.loads((sharded / "metrics.json").read_text())
    assert p["models"] == s["models"]
    assert p["n_matches"] == s["n_matches"] == 2 and p["n_accounted"] == 2
    assert p["generation"] == s["generation"] and p["run_commit"] == s["run_commit"]
    a = pd.read_parquet(serial / "f1b_feature_delta.parquet")
    b = pd.read_parquet(sharded / "f1b_feature_delta.parquet")
    cols = list(a.columns)
    pd.testing.assert_frame_equal(a.sort_values(cols).reset_index(drop=True), b[cols].sort_values(cols).reset_index(drop=True))


def test_sharded_run_reduces_to_the_serial_artifact(tmp_path, monkeypatch, capsys):
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    capsys.readouterr()
    _t10(monkeypatch, "--data-dir", cache, "--list-match-keys")
    assert json.loads(capsys.readouterr().out) == _KEYS
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS[:1])
    _worker(monkeypatch, tmp_path, cache, sharded, 1, _KEYS[1:])
    assert not (sharded / "metrics.json").exists()  # a worker never writes the corpus artifact
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_a_worker_killed_before_its_manifest_still_reduces(tmp_path, monkeypatch):
    """Killed after its last shard but before writing manifest_<tag>.json: the commit-keyed generation
    still attributes the shards, so the reduce succeeds and matches serial (CCC-PLAN-11)."""
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS)
    for mf in (sharded / "_shards").glob("*/manifest_*.json"):
        mf.unlink()  # simulate the kill
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_a_worker_resumed_with_the_same_tag_reduces_identically(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS[:1])  # "killed" after one item
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS)  # relaunched: resumes, attempts only the rest
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_reduce_refuses_an_unfinished_corpus(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS[:1])
    with pytest.raises(SystemExit, match="have no shard"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")


def test_reduce_refuses_a_manifest_from_another_commit(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS)
    (mf,) = list((out / "_shards").glob("*/manifest_w0.json"))
    m = json.loads(mf.read_text())
    m["run_commit"] = "deadbeef"
    mf.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="another commit"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")


def test_the_generation_is_keyed_on_the_commit(tmp_path, monkeypatch):
    """B r4 CCC-PLAN-31: a worker at commit A that died before its manifest is refused by a reduce at commit
    B. No manifest is left, so the commit-keyed GENERATION is what refuses (the CCC-SPEC-04 mechanism); the
    same reduce at commit A succeeds, so the commit is the only difference."""
    import scripts.measure_f1b_feature_delta as t10

    def _at(commit):
        monkeypatch.setattr(t10, "git_provenance", lambda: {"commit": commit * 40, "dirty": False, "tree_state": "clean"})

    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _at("a")
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS)
    for mf in (out / "_shards").glob("*/manifest_*.json"):
        mf.unlink()  # killed before its manifest
    _at("b")
    with pytest.raises(SystemExit, match="expected exactly the generation"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")
    _at("a")
    _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")
    assert json.loads((out / "metrics.json").read_text())["run_commit"] == "a" * 40


def test_shards_only_requires_a_worker_tag(tmp_path, monkeypatch):
    """Must FAIL before the change for the right reason: assert the message, not just SystemExit."""
    cache = _tc3_cache(tmp_path)
    with pytest.raises(SystemExit):
        _t10(monkeypatch, "--data-dir", cache, "--out", tmp_path / "o", "--shards-only", "--allow-dirty")
    # argparse prints its error; the plan's message is the discriminator
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), pytest.raises(SystemExit):
        _t10(monkeypatch, "--data-dir", cache, "--out", tmp_path / "o", "--shards-only", "--allow-dirty")
    assert "--shards-only needs a unique --worker-tag" in buf.getvalue()


def test_unknown_match_key_is_refused(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    kj = tmp_path / "k.json"
    kj.write_text(json.dumps(["nope"]))
    with pytest.raises(SystemExit, match="absent from --data-dir"):
        _t10(monkeypatch, "--data-dir", cache, "--out", tmp_path / "o", "--shards-only", "--worker-tag", "w0",
             "--match-keys-json", kj, "--allow-dirty")
```

- [ ] **Step 2: Run — expect FAIL** (`unrecognized arguments`); the worker-tag test fails on its message assertion.
- [ ] **Step 3: Implement.** Replace `main()` and add the helpers directly above it (add `import hashlib`):

```python
def _token_inputs(paths: list[pathlib.Path], data_dir: pathlib.Path, commit: str) -> dict:
    """The for_each generation key: driver schema + the RUN COMMIT + the corpus identity (a digest of
    the FULL sorted key list, never a worker's subset). Shards are therefore attributable to a commit
    even when a worker dies before writing its manifest, and a worker resumed at another commit lands
    in another generation (combined-cycle spec section 6)."""
    keys = sorted(_match_key(p, data_dir) for p in paths)
    return {
        "schema": _SHARD_SCHEMA_VERSION,
        "driver": "f1b-feature-delta",
        "atol": _ATOL,
        "commit": commit,
        "corpus": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
    }


def _select(paths: list[pathlib.Path], data_dir: pathlib.Path, keys_json: str | None) -> list[pathlib.Path]:
    """The worker's subset: ``paths`` filtered to the keys in ``keys_json`` (a JSON list), corpus order."""
    if keys_json is None:
        return paths
    wanted = set(json.loads(pathlib.Path(keys_json).read_text(encoding="utf-8")))
    known = {_match_key(p, data_dir) for p in paths}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"--match-keys-json names {len(unknown)} key(s) absent from --data-dir: {unknown[:5]}")
    return [p for p in paths if _match_key(p, data_dir) in wanted]


def _map(paths: list[pathlib.Path], data_dir: pathlib.Path, out: pathlib.Path, *, token: dict):
    return for_each(
        paths,
        key=lambda fp: _match_key(fp, data_dir),
        work=lambda fp: _measure_one_match(fp, data_dir),
        shard_root=out / "_shards",
        token_inputs=token,
        label="match",
    )


def _write_worker_manifest(res, *, prov: dict, worker_tag: str) -> None:
    """Persist THIS worker's manifest beside its shards; the reduce reads every worker's."""
    (res.shard_dir / f"manifest_{worker_tag}.json").write_text(
        json.dumps({**res.manifest(), "run_commit": prov["commit"], "run_tree_dirty": prov["dirty"]}, default=str),
        encoding="utf-8",
    )


def _artifact(combined: pd.DataFrame, *, n_matches: int, manifest: dict, prov: dict, dirty: bool, n_accounted: int) -> dict:
    """The metrics.json body -- ONE schema for the serial run and the sharded reduce (+ n_accounted)."""
    out: dict[str, object] = {"atol": _ATOL, "n_matches": n_matches, "models": _aggregate(combined)}
    out.update(manifest)
    out["n_accounted"] = n_accounted  # keys with a shard or exclusion marker; n_attempted counts only this pass
    out["run_commit"] = prov["commit"]
    out["run_tree_dirty"] = dirty
    return out


def reduce_t10(paths: list[pathlib.Path], data_dir: pathlib.Path, out: pathlib.Path, *, prov: dict) -> dict:
    """Reduce every worker's shards into the corpus artifact (combined-cycle spec section 6).

    Completeness is by ACCOUNTED KEYS (shard or exclusion marker for every listed key) in the ONE
    generation this commit's token inputs produce; every worker manifest present must name this commit.
    """
    from scripts._driver import _token, exclusion_path, shard_path
    from scripts._partition import aggregate_manifests

    root = out / "_shards"
    gens = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    expected = _token(_token_inputs(paths, data_dir, prov["commit"]), None)
    if [g.name for g in gens] != [expected]:
        raise SystemExit(
            f"expected exactly the generation {expected} (this commit + this corpus) under {root}, found "
            f"{[g.name for g in gens]}; a worker ran at another commit or on another corpus -- use a fresh --out"
        )
    gen = gens[0]
    keys = [_match_key(p, data_dir) for p in paths]
    missing = [k for k in keys if not shard_path(gen, k).is_file() and not exclusion_path(gen, k).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(keys)} listed matches have no shard (first: {missing[:3]}); a worker has "
            "not finished -- re-run it with the same --worker-tag (it resumes)."
        )
    agg = aggregate_manifests(gen, defaults=("n_attempted", "n_failed", "n_counters_unrecorded", "n_excluded"))
    foreign = sorted(set(agg["commits_seen"]) - {prov["commit"]})
    if foreign:
        raise SystemExit(f"worker manifest(s) from another commit {foreign}; this reduce runs at {prov['commit']}")
    combined = reconcile(gen, out / "f1b_feature_delta.parquet", tag="all")
    if not len(combined):
        raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")
    manifest = {
        "generation": gen.name,
        "n_attempted": agg["n_attempted"],
        "n_failed": agg["n_failed"],
        "n_counters_unrecorded": agg["n_counters_unrecorded"],
        "n_excluded": agg["n_excluded"],
    }
    return _artifact(combined, n_matches=len(paths), manifest=manifest, prov=prov,
                     dirty=bool(prov["dirty"] or agg["run_tree_dirty"]), n_accounted=len(keys) - len(missing))


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the F1b float32-storage per-feature delta by model.")
    ap.add_argument("--data-dir", type=pathlib.Path, required=True, help="float64-stored tracking-frame corpus")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="artifact directory (metrics.json written here)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--list-match-keys", action="store_true", help="print the corpus match keys as JSON and exit")
    ap.add_argument("--match-keys-json", default=None, help="JSON list of match keys this --shards-only worker handles")
    ap.add_argument("--shards-only", action="store_true", help="MAP only: shards + manifest_<worker-tag>.json, no metrics.json")
    ap.add_argument("--worker-tag", default=None, help="unique per-worker manifest tag (required with --shards-only)")
    ap.add_argument("--reduce-only", action="store_true", help="REDUCE only: metrics.json from every worker's shards")
    args = ap.parse_args()

    paths = frame_parquets(args.data_dir)
    if not paths:
        raise SystemExit(f"no frame parquets under {args.data_dir}. Point --data-dir at a float64 frame corpus.")
    if args.list_match_keys:
        print(json.dumps(sorted(_match_key(p, args.data_dir) for p in paths), indent=2))
        return
    if args.out is None:
        ap.error("--out is required unless --list-match-keys is given")
    if args.shards_only and args.reduce_only:
        ap.error("--shards-only and --reduce-only are mutually exclusive")
    if args.shards_only and not args.worker_tag:
        ap.error("--shards-only needs a unique --worker-tag")
    if args.match_keys_json and not args.shards_only:
        ap.error("--match-keys-json is a --shards-only worker flag; the reduce always covers the whole corpus")

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reduce_only:
        out = reduce_t10(paths, args.data_dir, args.out, prov=prov)
    else:
        token = _token_inputs(paths, args.data_dir, prov["commit"])
        res = _map(_select(paths, args.data_dir, args.match_keys_json), args.data_dir, args.out, token=token)
        _write_worker_manifest(res, prov=prov, worker_tag=args.worker_tag or "serial")
        if args.shards_only:
            print(json.dumps({"shards_only": True, "worker_tag": args.worker_tag, **res.manifest()}, default=str))
            return
        combined = reconcile(res.shard_dir, args.out / "f1b_feature_delta.parquet", tag="all")
        if not len(combined):
            raise SystemExit("every shard was empty -- the corpus yielded no measurable frames.")
        out = _artifact(combined, n_matches=len(paths), manifest=res.manifest(), prov=prov,
                        dirty=bool(prov["dirty"]), n_accounted=len(paths))

    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
```

`_driver._token` is private. Before using it, confirm `_token(token_inputs, None)` returns exactly the generation directory name `generation_dir` creates (read `_driver.py:65-110`). If the gates object to the private import, expose a one-line public `generation_name(token_inputs)` in `_driver.py` beside `generation_dir`, and use that instead.
- [ ] **Step 4: Run — expect PASS:** `python -m pytest tests/scripts/test_measure_f1b_feature_delta.py tests/scripts/test_provenance_wiring.py tests/scripts/test_corpus_driver_resilience.py -q`.

### Task 9: TF-19 `--reduce-only` over the allowlist population

**Files:** Modify `scripts/build_tf19_instrument_responsiveness.py` (argparse `:458-484`; the worker branch `:559`; the authoritative reduce `:574-655`; for_each `token_inputs`). Test `tests/scripts/test_tf19_instrument_responsiveness_driver.py` (append).

**Interfaces (produced):**
- CLI `--reduce-only` (requires `--match-ids-json` = the POPULATION).
- The for_each `token_inputs` gain `"commit"`.
- Reduce-only behaviour:
  - refuses unless every population ref has a shard or an exclusion marker;
  - refuses if any worker manifest in `--out` names another commit;
  - reduces over exactly the population's shards (keeper-identity counters replayed by `for_each`);
  - `metrics.json` gains `"reduce_mode": "driver-reduce-only"` and `"population_size"`.

- [ ] **Step 1: Failing test.** Add `import pytest` to the file's imports (it has none; B r2 CCC-PLAN-20), then append:

```python
def _full_shard(nrows, *, game, keeper=7):
    """A shard with the FULL driver schema (`_SHARD_COLUMNS`): the authoritative reduce reads `keeper_key`,
    the frame keys, `realistic_signed` and `keeper_gr_depth`, which `_shard` (pooling-only) omits."""
    from scripts.build_tf19_instrument_responsiveness import _FRAME_KEYS, _SHARD_COLUMNS

    df = _shard(nrows)
    df["keeper_key"] = keeper
    for k in _FRAME_KEYS:
        df[k] = np.arange(nrows) if k == "frame_id" else (game if k == "game_id" else 1)
    df["realistic_signed"] = -0.05
    df["keeper_gr_depth"] = 5.0
    return df[_SHARD_COLUMNS]


def test_reduce_only_over_an_allowlist_equals_an_unpartitioned_serial_run(tmp_path, monkeypatch):
    """Workers on slices + --reduce-only over the whole allowlist == one unpartitioned run (CCC-SPEC-03)."""
    import json
    import sys

    import scripts._loader_pining as lp
    import scripts.build_tf19_instrument_responsiveness as D

    pop = {"gradientsports": ["1", "2"]}
    refs_all = [lp.MatchRef("gradientsports", m, {}) for m in pop["gradientsports"]]

    def fake_pining_source(providers, match_ids=None, **kw):
        wanted = (match_ids or {}).get("gradientsports") or pop["gradientsports"]
        return [r for r in refs_all if r.match_id in wanted], (lambda ref: ("gradientsports", ref.match_id, None, None))

    monkeypatch.setattr(lp, "pining_source", fake_pining_source)
    counts = {"n_keeper_teams": 2, "n_keeper_teams_resolved": 2, "n_keeper_teams_unresolved": 0}
    monkeypatch.setattr(D, "_measure_match", lambda item, rng_seed: (_full_shard(300, game=int(item[1])), counts))

    def run(out, *extra):
        monkeypatch.setattr(sys, "argv", ["d", "--out", str(out), "--providers", "gradientsports", "--allow-dirty", *map(str, extra)])
        D.main()

    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    run(serial)
    for i, m in enumerate(pop["gradientsports"]):
        sl = tmp_path / f"s{i}.json"
        sl.write_text(json.dumps({"gradientsports": [m]}))
        run(sharded, "--match-ids-json", sl)
    assert not (sharded / "metrics.json").exists()
    popf = tmp_path / "pop.json"
    popf.write_text(json.dumps(pop))
    run(sharded, "--match-ids-json", popf, "--reduce-only")
    s = json.loads((serial / "metrics.json").read_text())
    p = json.loads((sharded / "metrics.json").read_text())
    assert p["verdicts"] == s["verdicts"] and p["n_frames_scored"] == s["n_frames_scored"]
    assert p["keeper_identity"]["n_keeper_teams"] == s["keeper_identity"]["n_keeper_teams"] == 4
    assert p["reduce_mode"] == "driver-reduce-only" and p["population_size"] == 2


def test_reduce_only_refuses_a_missing_population_shard(tmp_path, monkeypatch):
    import json
    import sys

    import scripts._loader_pining as lp
    import scripts.build_tf19_instrument_responsiveness as D

    refs_all = [lp.MatchRef("gradientsports", m, {}) for m in ("1", "2")]
    monkeypatch.setattr(lp, "pining_source", lambda providers, match_ids=None, **kw: (refs_all, (lambda ref: ("gradientsports", ref.match_id, None, None))))
    monkeypatch.setattr(D, "_measure_match", lambda item, rng_seed: (_full_shard(300, game=int(item[1])), {}))
    popf = tmp_path / "pop.json"
    popf.write_text(json.dumps({"gradientsports": ["1", "2"]}))
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(tmp_path / "o"), "--providers", "gradientsports", "--allow-dirty", "--match-ids-json", str(popf), "--reduce-only"])
    with pytest.raises(SystemExit, match="have no shard"):
        D.main()
```

(`_shard` is the file's existing helper; `_full_shard` extends it to the driver schema. Reviewer B ran both tests green against the Task 9 implementation once the import and the full schema were in place, round 2.)
- [ ] **Step 2: Run — expect FAIL** (`unrecognized arguments: --reduce-only`).
- [ ] **Step 3: Implement.**
- **argparse:**

```python
    ap.add_argument(
        "--reduce-only",
        action="store_true",
        help="AUTHORITATIVE reduce over exactly the --match-ids-json POPULATION (every shard must exist): "
        "the allowlist-honouring replacement for the unpartitioned pass, which lists whole manifests.",
    )
```

- **Validation**, after `parse_args`: `if args.reduce_only and not args.match_ids_json: ap.error("--reduce-only needs --match-ids-json (the population)")`.
- **Before `for_each`**, if `args.reduce_only`, check every population ref is accounted in the generation the call will use:

```python
    if args.reduce_only:
        from scripts._driver import exclusion_path, generation_dir, shard_path

        gen = generation_dir(dest / "shards", token_inputs=_token)  # the same dict passed to for_each below
        missing = [r.key for r in refs if not shard_path(gen, r.key).is_file() and not exclusion_path(gen, r.key).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(refs)} population matches have no shard; finish the workers first")
```

  Hoist the existing `token_inputs` dict into a local `_token`, and add `"commit": prov["commit"]` to it.
- **`for_each`** then resumes everything: no item is loaded, and the counters are replayed.
- **The worker early-return** becomes `if args.match_ids_json is not None and not args.reduce_only:`.
- **In the authoritative reduce:**
  - restrict `shard_files` to the population: `[shard_path(res.shard_dir, r.key) for r in refs if shard_path(res.shard_dir, r.key).is_file()]`;
  - before writing, require `aggregate_manifests(dest)["commits_seen"] ⊆ {prov["commit"]}` (any manifest written to `dest` by the workers), else `SystemExit`;
  - add `"reduce_mode": "driver-reduce-only" if args.reduce_only else "driver-unpartitioned"` and `"population_size": len(refs)` to `out`.
- [ ] **Step 4: Run — expect PASS:** `python -m pytest tests/scripts/test_tf19_instrument_responsiveness_driver.py tests/scripts/test_input_contracts.py tests/scripts/test_provenance_wiring.py -q`.

### Task 10: DAS parity driver — schema `-4`, full numba coverage, counts, completeness, commits, slices (D2)

**Files:** Modify `scripts/validate_das_native_parity.py`, `scripts/_das_reference_leg.py`. Tests `tests/scripts/test_das_native_parity_driver.py`, `tests/scripts/test_das_reference_leg.py` (modify / append).

**Interfaces (produced):**
- `_SHARD_SCHEMA_VERSION = "das-native-parity-4"`.
- New shard columns:
  - `numba_minus_numpy_as` (team + player rows; player rows also carry `numba_minus_numpy_das`);
  - match rows: `n_dkey_frames`, `n_dir_compared`, `n_dir_disagree`.
- `_GOLDEN_TOL_NUMPY = 1e-12`, `_GOLDEN_TOL_NUMBA = 1e-10`; `_n_outside_golden_bound(abs_d, rel_d, tol, extra=None) -> int`.
- `_n_dkey_frames(frames) -> int` (pure).
- Per-provider reduce keys:
  - `n_outside_golden_bound = {"numpy": {grain: {"das", "as"}}, "numba": {grain: {"das", "as"}}}`;
  - `numba_compared = {grain: {"das", "as"}}` (finite numba comparisons per cell);
  - `numba_vs_numpy_max_abs = {grain: {"das", "as"}}`;
  - `finite_counts = {grain: {"ref", "native", "rows"}}`;
  - `d_key_frames`;
  - `direction = {"n_compared", "n_disagree"}`.
- `population["accounted"]`; `commit_consistent`; `commits_seen`.
- `_map_generation(commit, direction_col=None) -> str` and CLI `--print-generation` (B r3 CCC-PLAN-28).
- `_measure_match(item, *, reference_leg, inferred_leg=None, direction_col=None)`.
- `_das_reference_leg.reference_leg_arrays(frames, *, repeat=1, infer_direction=False)`.
- `_reference_leg_subprocess(frames, *, reference_python, repeat=1, infer_direction=False)`.

- [ ] **Step 1: Failing tests.**
- **Modify** `test_reference_leg_subprocess_round_trips`: in `fake_run`, `out = Path(cmd[3])` instead of `cmd[-1]`, and write `(out / "timing.json").write_text('{"compute_s": 0.25, "repeat": 1}', encoding="utf-8")`; after the call, `assert got["compute_s"] == 0.25`.
- **Append** the tests below. They reuse the file's `_mk_row`, `_shard`, `_corpus`, `_run`, `_CLEAN_PROV`, `D`, `Reason`:

```python
def test_n_outside_golden_bound_applies_allclose_per_row():
    abs_d = [0.0, 5e-13, 2e-12, 1e-6]
    rel_d = [0.0, 5e-15, float("nan"), 1e-8]  # exact; inside (ref 100); ref==0 & |d|>atol; outside
    assert D._n_outside_golden_bound(abs_d, rel_d, 1e-12) == 2
    extra = [0.0, 1e-9, 0.0, 0.0]  # |numba - numpy|, triangle bound
    assert D._n_outside_golden_bound(abs_d, rel_d, 1e-10, extra=extra) == 1
    assert D._n_outside_golden_bound([float("nan")], [float("nan")], 1e-12) == 0  # finite-mask case, counted elsewhere


def test_reduce_reports_all_four_cells_and_the_new_counts():
    ok = dict(abs_as=0.0, rel_as=0.0, finite_ref=True, finite_native=True, quad_shift_das=0.0,
              numba_minus_numpy_das=0.0, numba_minus_numpy_as=0.0, reason=int(Reason.OK))
    player_ok = {**ok, "numba_minus_numpy_as": 1e-6}
    rows = [
        _mk_row("team", period_id=1, frame_id=1, abs_das=0.0, rel_das=0.0, **ok),
        _mk_row("team", period_id=1, frame_id=2, abs_das=1e-6, rel_das=1e-8, **ok),
        _mk_row("player", period_id=1, frame_id=1, player_id=9, abs_das=0.0, rel_das=0.0, **player_ok),
        _mk_row("match", n_scored_frames=2, n_dkey_frames=1, n_dir_compared=2, n_dir_disagree=1,
                **{c: 0 for c in D._REASON_COLS.values()}),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    out = prov["n_outside_golden_bound"]
    assert out["numpy"]["team"] == {"das": 1, "as": 0}
    assert out["numpy"]["player"] == {"das": 0, "as": 0}
    assert out["numba"]["team"] == {"das": 1, "as": 0}
    assert out["numba"]["player"] == {"das": 0, "as": 1}
    # CCC-PLAN-23: per-cell count of FINITE numba comparisons (the gate requires every cell > 0)
    assert prov["numba_compared"] == {"team": {"das": 2, "as": 2}, "player": {"das": 1, "as": 1}}
    assert prov["finite_counts"]["team"] == {"ref": 2, "native": 2, "rows": 2}
    assert prov["d_key_frames"] == 1
    assert prov["direction"] == {"n_compared": 2, "n_disagree": 1}


def test_numba_cells_with_no_finite_comparison_are_counted_zero():
    """CCC-PLAN-23: an empty/misaligned numba player merge leaves NaN diffs -- 0 compared, never 'clean'."""
    nan_nb = dict(abs_as=0.0, rel_as=0.0, finite_ref=True, finite_native=True, quad_shift_das=0.0,
                  numba_minus_numpy_das=np.nan, numba_minus_numpy_as=np.nan, reason=int(Reason.OK))
    rows = [
        _mk_row("player", period_id=1, frame_id=1, player_id=9, abs_das=0.0, rel_das=0.0, **nan_nb),
        _mk_row("match", n_scored_frames=1, **{c: 0 for c in D._REASON_COLS.values()}),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["numba_compared"]["player"] == {"das": 0, "as": 0}


def test_n_dkey_frames_counts_frames_whose_frame_id_recurs_in_another_period():
    """CCC-PLAN-22: an exact two-period fixture. Period 2 reuses frame_ids 1..5 of period 1 (10 colliding
    frame keys) and adds frame 6 (unique); a second game reusing the same ids does NOT collide."""
    keys = pd.DataFrame(
        {
            "game_id": [1] * 11 + [2] * 5,
            "period_id": [1] * 5 + [2] * 6 + [1] * 5,
            "frame_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
        }
    )
    frames = keys.loc[keys.index.repeat(3)].reset_index(drop=True)  # several rows per frame, as real frames have
    assert D._n_dkey_frames(frames) == 10


def test_measure_match_records_dkey_and_direction(monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    item = load(refs[0])

    def inferred(frames):  # the library's own direction disagrees on every frame
        r = stub_ref(frames)
        return {**r, "team_das": r["team_das"] + 5.0}

    shard = D._measure_match(item, reference_leg=stub_ref, inferred_leg=inferred, direction_col="dir")
    m = shard[shard["grain"] == "match"].iloc[0]
    assert int(m["n_dir_compared"]) > 0 and int(m["n_dir_disagree"]) == int(m["n_dir_compared"])
    assert int(m["n_dkey_frames"]) == 0  # the golden scenes are single-period: exactly 0, not ">= 0"


def test_reduce_refuses_an_unaccounted_match(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(refs[:1], load, tmp_path / "out", prov=_CLEAN_PROV, shard_root=root, direction_col="dir",
                 reference_leg=stub_ref, shards_only=True, worker_tag="w0")
    with pytest.raises(SystemExit, match="have no shard"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")


def test_reduce_at_another_commit_is_refused_by_the_generation_check(tmp_path, monkeypatch):
    """CCC-PLAN-21: the commit-keyed generation makes this the refusal that fires."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, shard_root=root, direction_col="dir",
                 reference_leg=stub_ref, shards_only=True, worker_tag="w0")
    other = {**_CLEAN_PROV, "commit": "1" * 40}
    with pytest.raises(SystemExit, match=r"does not match this reduce's token .*commit 1{40}"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=other, direction_col="dir")


def test_a_token_mismatch_at_the_same_commit_names_the_token_inputs(tmp_path, monkeypatch):
    """B r3 CCC-PLAN-26: a direction_col mismatch is not blamed on the commit alone."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, shard_root=root, direction_col="dir",
                 reference_leg=stub_ref, shards_only=True, worker_tag="w0")
    with pytest.raises(SystemExit, match=r"does not match this reduce's token .*direction_col None"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col=None)


def test_print_generation_is_the_token_the_map_writes(tmp_path, monkeypatch, capsys):
    """B r3 CCC-PLAN-28: the launcher's done-marker is keyed on the generation this prints."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, shard_root=root, direction_col=None,
                 reference_leg=stub_ref, shards_only=True, worker_tag="w0")
    assert [p.name for p in root.iterdir() if p.is_dir()] == [D._map_generation(_CLEAN_PROV["commit"])]


def test_a_planted_foreign_manifest_is_refused(tmp_path, monkeypatch):
    """Defence in depth (CCC-PLAN-21): a manifest naming another commit inside THIS commit's generation
    (e.g. copied in by hand) is refused by the commits_seen check, which this test keeps exercised."""
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()
    root = tmp_path / "shards"
    D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, shard_root=root, direction_col="dir",
                 reference_leg=stub_ref, shards_only=True, worker_tag="w0")
    (gen,) = [p for p in root.iterdir() if p.is_dir()]
    (gen / "manifest_rogue.json").write_text('{"n_attempted": 1, "run_commit": "' + "1" * 40 + '"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="another commit"):
        D.reduce_parity_artifact(refs, root, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir")


def test_population_records_accounted_and_commit(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert out["population"]["accounted"] == 2  # informational: the reduce refuses before it could differ (B r2 C7)
    assert out["commits_seen"] == [_CLEAN_PROV["commit"]]


def test_main_slices_providers_to_the_allowlist(tmp_path, monkeypatch):
    """CCC-PLAN-08: a worker slice without IDSSE ids must not process the whole IDSSE manifest."""
    seen = {}

    def fake_pining_source(providers, **kw):
        seen["providers"] = list(providers)
        raise SystemExit("stop")

    monkeypatch.setattr(D, "pining_source", fake_pining_source)
    sl = tmp_path / "s.json"
    sl.write_text('[{"provider": "skillcorner", "match_id": "1"}]')
    monkeypatch.setattr(sys, "argv", ["d", "--out", str(tmp_path / "o"), "--match-ids-json", str(sl), "--allow-dirty", "--shards-only", "--worker-tag", "w0"])
    with pytest.raises(SystemExit, match="stop"):
        D.main()
    assert seen["providers"] == ["skillcorner"]
```

(Add `import sys` to the test file if absent. `reduce_parity_artifact` gains `direction_col` because the expected generation depends on the token inputs.)

`tests/scripts/test_das_reference_leg.py` (append):

```python
def test_best_of_returns_the_minimum_and_the_last_result():
    from scripts._das_reference_leg import best_of

    calls = []

    def fn():
        calls.append(1)
        return len(calls)

    result, seconds = best_of(fn, 3)
    assert result == 3 and len(calls) == 3 and seconds >= 0.0
    assert best_of(fn, 0)[0] == 4  # repeat < 1 still runs once


def test_inferred_direction_common_drops_the_supplied_direction():
    from scripts._das_reference_leg import _common

    c = _common(infer_direction=True)
    assert c["infer_attacking_direction"] is True and c["attacking_direction_col"] is None
    assert _common(infer_direction=False)["attacking_direction_col"] == "_das_parity_dir"
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `scripts/_das_reference_leg.py`.**
- **Imports:** add `import json`, `import time`, `from collections.abc import Callable` and `from typing import TypeVar`; module level `_T = TypeVar("_T")`.
- **New helpers** (`best_of` is typed so `(team, ind), compute_s = best_of(...)` unpacks cleanly under pyright; B r3 CCC-PLAN-25):

```python
def best_of(fn: Callable[[], _T], repeat: int) -> tuple[_T, float]:
    """``(result, best_seconds)`` over ``max(1, repeat)`` calls of ``fn`` -- the minimum wall time;
    the result is the last call's."""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    for _ in range(max(1, repeat) - 1):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _common(*, infer_direction: bool) -> dict:
    """The library call recipe: the shared direction column, or the library's OWN inference (the
    das-native 7.2 'GoalMap direction versus reference inference' leg)."""
    base = {**_REFERENCE_COMMON, "frame_col": "_uframe"}
    if infer_direction:
        return {**base, "attacking_direction_col": None, "infer_attacking_direction": True}
    return base
```

- **`reference_leg_arrays(frames, *, repeat: int = 1, infer_direction: bool = False)`:**
  - `common = _common(infer_direction=infer_direction)`.
  - Replace the two library calls with:

```python
    def _library():
        team = asp.get_dangerous_accessible_space(lib.copy(), **common)
        # the inferred-direction leg needs team DAS only (the comparison is per frame)
        ind = None if infer_direction else asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
        return team, ind

    # Timed in-process: the library compute only -- no interpreter start-up, imports or parquet I/O.
    (team, ind), compute_s = best_of(_library, repeat)
```

  - When `ind is None`, return empty player arrays (`np.empty((0, 4), object)` keys, empty floats) and skip the per-row loop's player branch.
  - Add `"compute_s": np.asarray(compute_s, dtype=float)` to the returned dict.
- **`_main(in_parquet, out_dir, repeat=1, infer_direction=False)`:**
  - passes both through;
  - after the parquet writes, adds:

```python
    (out / "timing.json").write_text(
        json.dumps({"compute_s": float(arrays["compute_s"]), "repeat": repeat}), encoding="utf-8"
    )
```

  - `__main__`: `_main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1, len(sys.argv) > 4 and sys.argv[4] == "1")`.
- **The inference keyword pair.**
  - The golden generator's DAS `_COMMON` (`tests/tracking/_fixtures/das_golden/_generate.py:358`) uses `infer_attacking_direction=False`; `:469` is the event/xC call, not DAS (B r2 C4).
  - Reviewer B confirmed, round 2, that the 2.0.15 `get_dangerous_accessible_space` accepts `attacking_direction_col=None` with `infer_attacking_direction=True`, and ran the inferred leg locally (S01: 3 frames compared, 0 disagree).
  - Re-confirm the signature in the DGX `py2ref` env (Task 16 Step 3): `python -c "import inspect, accessible_space as a; print(inspect.signature(a.get_dangerous_accessible_space))"`.
  - When the inferred leg writes no players, `player.parquet` still has its columns, so the driver reads a `(0, 4)` key array.
- [ ] **Step 4: Implement `scripts/validate_das_native_parity.py`.**
- **(a) Schema.** Bump `_SHARD_SCHEMA_VERSION = "das-native-parity-4"`. Add `"numba_minus_numpy_as"` after `"numba_minus_numpy_das"` in `_EMITTED_SHARD_COLUMNS`, and `"n_dkey_frames", "n_dir_compared", "n_dir_disagree"` after `"ms_frame_periodic"`. Add the constants after `_THREAD_SWEEP`:

```python
#: das-native spec 4.1 golden-fixture bounds, applied PER ROW on the corpus (np.allclose, rtol = atol).
_GOLDEN_TOL_NUMPY = 1e-12
_GOLDEN_TOL_NUMBA = 1e-10
```

  Add `"golden_tolerances": {"numpy": _GOLDEN_TOL_NUMPY, "numba": _GOLDEN_TOL_NUMBA}` to `input_contract()` params.
- **(b) `_reference_leg_subprocess(frames, *, reference_python, repeat=1, infer_direction=False)`:**
  - the argv gains `str(repeat), "1" if infer_direction else "0"`;
  - the returned dict gains `"compute_s": np.asarray(json.loads((d / "timing.json").read_text(encoding="utf-8"))["compute_s"], dtype=float)` — a 0-d array, so the `-> dict[str, np.ndarray]` annotation stays true (B r3 CCC-PLAN-25; a bare `float` is a pyright `reportReturnType`);
  - when `infer_direction`, `player.parquet` may be empty — read it the same way.
- **(c) `_team_rows`:** the numba merge takes both outputs: `nb_team.rename(columns={"team_das": "nb_das", "team_as": "nb_as"})[[*_FRAME_KEYS, "nb_das", "nb_as"]]`. Compute `nb_as` like `nb`; each row adds `"numba_minus_numpy_as": float(nb_as[i] - merged["team_as"].iloc[i]) if np.isfinite(nb_as[i]) else np.nan`.
- **(d) `_player_rows(provider, ref, np_player, nb_player, flags)`:**
  - after the first merge:

```python
    if nb_player is not None:
        nb_cols = nb_player.rename(columns={"player_das": "nb_das", "player_as": "nb_as"})
        merged = merged.merge(
            nb_cols[[*_FRAME_KEYS, "player_id", "nb_das", "nb_as"]], on=[*_FRAME_KEYS, "player_id"], how="left"
        )
    nb_d = merged["nb_das"].to_numpy(float) if "nb_das" in merged.columns else np.full(len(merged), np.nan)
    nb_a = merged["nb_as"].to_numpy(float) if "nb_as" in merged.columns else np.full(len(merged), np.nan)
```

  - each row adds `"numba_minus_numpy_das": float(nb_d[i] - merged["player_das"].iloc[i]) if np.isfinite(nb_d[i]) else np.nan` and the same for `_as` with `nb_a` / `player_as`.
  - In `_measure_match`, the numba-absent branch sets `nb_team = nb_player = None` (B r2 C2).
- **(e) `_measure_match(item, *, reference_leg, inferred_leg=None, direction_col=None)`:**
  - keep `nb_player` from the numba leg and pass it to `_player_rows`;
  - compute the counts and add them to `match_row`:

```python
    n_dkey = _n_dkey_frames(scored)
    n_cmp = n_dis = 0
    if inferred_leg is not None:
        inf = inferred_leg(scored)
        a = _join_on_keys(ref["team_keys"], {"r": ref["team_das"]}, pd.DataFrame(
            {"game_id": inf["team_keys"][:, 0], "period_id": inf["team_keys"][:, 1], "frame_id": inf["team_keys"][:, 2],
             "i": inf["team_das"]}), list(_FRAME_KEYS))
        both = np.isfinite(a["r"].to_numpy(float)) & np.isfinite(a["i"].to_numpy(float))
        r, i = a["r"].to_numpy(float)[both], a["i"].to_numpy(float)[both]
        n_cmp, n_dis = int(both.sum()), int((np.abs(i - r) > 1e-9 * np.maximum(1.0, np.abs(r))).sum())
```

  `match_row.update({"n_dkey_frames": n_dkey, "n_dir_compared": n_cmp if inferred_leg else np.nan, "n_dir_disagree": n_dis if inferred_leg else np.nan})`. Before relying on the `_join_on_keys(ref_keys, ref_vals, native, key_cols)` call shape above, read the helper (`:431`) and match it. The D-KEY count is a pure, separately tested helper (CCC-PLAN-22), placed beside `_frame_flags`:

```python
def _n_dkey_frames(frames: pd.DataFrame) -> int:
    """Distinct (game, period, frame) keys whose (game, frame_id) recurs in ANOTHER period of the same game:
    the frames the old accessible-space keying (frame_id alone) conflated -- the D-KEY production figure."""
    keys = frames[list(_FRAME_KEYS)].drop_duplicates()
    per = keys.groupby(["game_id", "frame_id"])["period_id"].nunique()
    collide = per[per > 1].index
    return int(keys.set_index(["game_id", "frame_id"]).index.isin(collide).sum())
```
- **(f) `_n_outside_golden_bound`**, after `_grade_grain`:

```python
def _n_outside_golden_bound(abs_d, rel_d, tol: float, extra=None) -> int:
    """Rows violating ``|native - ref| <= tol + tol * |ref|`` (np.allclose with rtol = atol = tol).

    ``|ref|`` is recovered from the shard pair (``rel = abs / |ref|``; NaN when ``ref == 0``, where the
    bound is ``atol`` alone). ``extra`` adds ``|numba - numpy|``, so ``abs + extra`` bounds
    ``|numba - ref|`` by the triangle inequality (a conservative count). Non-finite rows are
    finite-mask cases, counted by ``finite_mask_mismatches``.
    """
    a = np.asarray(abs_d, dtype=float)
    r = np.asarray(rel_d, dtype=float)
    dist = a + np.abs(np.asarray(extra, dtype=float)) if extra is not None else a
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_mag = np.where(np.isfinite(r) & (r > 0), a / r, 0.0)
    return int((np.isfinite(dist) & (dist > tol + tol * ref_mag)).sum())
```

- **(g) `reduce_parity`.** Add to each provider's dict:

```python
            "n_outside_golden_bound": {
                eng: {
                    grain: {
                        o: _n_outside_golden_bound(
                            g[f"abs_{o}"], g[f"rel_{o}"], tol,
                            extra=(g[f"numba_minus_numpy_{o}"] if eng == "numba" else None),
                        )
                        for o in ("das", "as")
                    }
                    for grain, g in (("team", team_ok), ("player", player_ok))
                }
                for eng, tol in (("numpy", _GOLDEN_TOL_NUMPY), ("numba", _GOLDEN_TOL_NUMBA))
            },
            # CCC-PLAN-23: how many FINITE numba comparisons each cell actually made -- a NaN diff (empty or
            # misaligned numba merge) is "not compared", so a vacuous cell reads 0 here, never "clean".
            "numba_compared": {
                grain: {
                    o: int(np.isfinite(g[f"numba_minus_numpy_{o}"].to_numpy(float)).sum()) for o in ("das", "as")
                }
                for grain, g in (("team", team_ok), ("player", player_ok))
            },
            "finite_counts": {
                grain: {
                    "ref": int(g["finite_ref"].astype(bool).sum()),
                    "native": int(g["finite_native"].astype(bool).sum()),
                    "rows": len(g),
                }
                for grain, g in (("team", team), ("player", player))
            },
            "d_key_frames": int(match["n_dkey_frames"].fillna(0).sum()),
            "direction": {
                "n_compared": int(match["n_dir_compared"].fillna(0).sum()),
                "n_disagree": int(match["n_dir_disagree"].fillna(0).sum()),
            },
```

  Keep `numba_vs_numpy_das_max_abs` and the local `nbmax` that computes it exactly as they are (team DAS; an existing published key, `validate_das_native_parity.py:590`/`:596`). Do not delete `nbmax`: it is live (B r3 CCC-PLAN-24 corrects the rev-3 instruction). Add the per-cell maximum beside it:

```python
            "numba_vs_numpy_max_abs": {
                grain: {o: _pct(g[f"numba_minus_numpy_{o}"].abs(), 100) for o in ("das", "as")}
                for grain, g in (("team", team_ok), ("player", player_ok))
            },
```
- **(h) Token, manifests, completeness and commits.**
  - Factor the map `token_inputs` into `_map_token(direction_col, commit) -> dict` = the existing three keys + `"commit": commit`. `run_corpus` uses `_map_token(direction_col, prov["commit"])`.
  - Add the single-sourced generation name, which the reduce check and the launcher's done-marker both use (B r3 CCC-PLAN-28):

```python
def _map_generation(commit: str, direction_col: str | None = None) -> str:
    """The shard-generation directory name the map writes for ``commit`` (and the reduce expects)."""
    from scripts._driver import _token

    return _token(_map_token(direction_col, commit), None)
```

  - `run_corpus` gains `inferred_leg=None`, passed to `_measure_match`.
  - The worker manifest gains `"run_commit": prov["commit"], "run_tree_dirty": prov["dirty"]`.
  - `reduce_parity_artifact(refs, shard_root, dest, *, prov, direction_col=None)`, after locating `gen_dir`:

```python
    from scripts._driver import exclusion_path, shard_path

    expected = _map_generation(prov["commit"], direction_col)
    if gen_dir.name != expected:
        # The token covers the commit AND every other map input (B r3 CCC-PLAN-26): name them all.
        raise SystemExit(
            f"the shard generation {gen_dir.name} does not match this reduce's token {expected} "
            f"(commit {prov['commit']}, direction_col {direction_col!r}); "
            "reduce at the build commit with the build's flags"
        )
    missing = [
        r.key for r in refs if not shard_path(gen_dir, r.key).is_file() and not exclusion_path(gen_dir, r.key).is_file()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(refs)} listed matches have no shard (first: {missing[:3]}); "
            "re-run that worker (it resumes)"
        )
```

  - After `agg = aggregate_manifests(...)`:

```python
    # Defence in depth (CCC-PLAN-21): the commit-keyed generation already refuses a worker built at another
    # commit, so this fires only for a foreign manifest inside THIS generation (test_a_planted_foreign_manifest...).
    foreign = sorted(set(agg["commits_seen"]) - {prov["commit"]})
    if foreign:
        raise SystemExit(f"worker manifest(s) from another commit {foreign}")
```

  - Add `"commit_consistent": agg["commit_consistent"]`, `"commits_seen": agg["commits_seen"]` to `out`; set `run_tree_dirty` to `prov["dirty"] or agg["run_tree_dirty"]`.
  - `run_corpus`'s own reduce call becomes `reduce_parity_artifact(refs, root, dest, prov=prov, direction_col=direction_col)`.
  - Every existing test that calls `reduce_parity_artifact` directly (grep `tests/scripts/`, e.g. `test_reduce_split.py`) gains the `direction_col` argument its corpus was built with.
  - `_population(refs, scored_providers, manifest, *, accounted)` adds `"accounted": accounted`; pass `len(refs)` (all accounted, else the refusal fired).
- **(i) `main()`:**
  - `pining_source(providers_for_slice(providers, match_ids), …)` (`from scripts._partition import providers_for_slice`);
  - the reduce call passes `direction_col=None`;
  - the map passes `inferred_leg=functools.partial(_reference_leg_subprocess, reference_python=reference_python, infer_direction=True)`;
  - a `--print-generation` flag, handled right after argument parsing: `print(_map_generation(git_provenance()["commit"]))`, then return. It needs no token and no data. The launcher's DAS mode takes its output (Task 12b).
- [ ] **Step 5: Run — expect PASS:** `python -m pytest tests/scripts/test_das_native_parity_driver.py tests/scripts/test_das_reference_leg.py tests/scripts/test_reduce_split.py tests/scripts/test_provenance_wiring.py -q`.

### Task 11: DAS `--benchmark` (D1) — both paths in a pandas-2 env, foreign-CPU contention gate

**Files:** Create `scripts/_das_path_timing.py` and `tests/scripts/test_das_path_timing.py`. Modify `scripts/validate_das_native_parity.py` (benchmark functions + CLI) and `tests/scripts/test_das_native_parity_driver.py`. Modify `tests/scripts/_corpus_load_rules.py`: `_UNSHARDED_LOOP_EXEMPT["validate_das_native_parity.run_benchmark"]` with its reason. The D1 timing loop is one serial pass run alone, and its contention block spans the loop, so it is never sharded (owner-approved 2026-10-02, implementation review B m-1; `test_rule_c_ledger_is_exact_both_ways` flags it otherwise).

**Interfaces (produced):**
- `_das_path_timing.best_of`, `time_add_das_and_xfns(frames, actions, *, repeat, warmup) -> {"add_das_s", "das_xfns_s"}`, `_main(in_dir, out_dir, repeat)`. `timing.json` carries `silly_kicks`, `native` (bool: `silly_kicks.tracking._das_engine` importable) and `pandas`.
- Driver:
  - `_path_subprocess(frames, actions, *, python, repeat, expect_native: bool) -> dict`;
  - `_old_path_frames(frames)`;
  - `_keeper_counterfactual(scored, shift_m=1.0) -> (frames, moved_any)`;
  - `_bench_match(item, *, reference_python, old_path_python, new_path_python, repeat) -> dict`;
  - `summarize_benchmark(rows) -> dict`;
  - `_foreign_cpu_fraction(*, busy0, busy1, own0, own1, elapsed, ncpu) -> float`;
  - `_sweep_counts() -> list[int]` (the `_THREAD_SWEEP` counts ≤ `os.cpu_count()`);
  - `_benchmark_match_ids(sample_path) -> dict[str, list[str]]` (the sample file IS the allowlist);
  - `run_benchmark(...)` writes `performance.json`.
- CLI: `--benchmark --benchmark-sample-json --old-path-python ($SK_DAS_OLDPATH_PYTHON) --new-path-python ($SK_DAS_NEWPATH_PYTHON) --repeat 3`.

- [ ] **Step 1: Failing tests.** `tests/scripts/test_das_path_timing.py`:

```python
"""The add_das / das_xfns timing harness (combined-cycle-completion spec 12 D1)."""

import json

import pandas as pd

from scripts import _das_path_timing as pt


def test_time_add_das_and_xfns_uses_precomputed_links_and_warms_up(monkeypatch):
    import silly_kicks.tracking.features as feats
    import silly_kicks.tracking.utils as tu
    import silly_kicks.vaep.feature_framework as ff

    seen = {"add": 0, "xfn": 0}
    monkeypatch.setattr(tu, "link_actions_to_frames", lambda a, f: ("LINKS", None))
    monkeypatch.setattr(ff, "gamestates", lambda a, nb_prev_actions: ["STATES"])

    def fake_add(actions, frames, *, links):
        assert links == "LINKS"
        seen["add"] += 1

    def fake_xfn(states, frames):
        assert states == ["STATES"]
        seen["xfn"] += 1

    monkeypatch.setattr(feats, "add_das", fake_add)
    monkeypatch.setattr(feats, "das_xfns", [fake_xfn])
    out = pt.time_add_das_and_xfns(pd.DataFrame(), pd.DataFrame(), repeat=3, warmup=True)
    assert seen == {"add": 4, "xfn": 4}  # 1 warm-up + 3 timed each
    assert set(out) == {"add_das_s", "das_xfns_s"} and min(out.values()) >= 0.0


def test_main_writes_timing_json_with_version_and_engine_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "time_add_das_and_xfns", lambda f, a, *, repeat, warmup: {"add_das_s": 1.0, "das_xfns_s": 2.0})
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "frames.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "actions.parquet")
    pt._main(str(tmp_path), str(tmp_path), 3)
    t = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert t["add_das_s"] == 1.0 and t["repeat"] == 3 and t["silly_kicks"] and t["native"] is True
```

Append to `tests/scripts/test_das_native_parity_driver.py`:

```python
def test_old_path_frames_restore_pre_f1b_dtypes():
    f = pd.DataFrame({"x": pd.Series([1.0], dtype="float32"), "team_id": pd.Series([1], dtype="Int64").astype("category")})
    out = D._old_path_frames(f)
    assert out["x"].dtype == "float64" and str(out["team_id"].dtype) == "Int64"


def test_thread_sweep_skips_counts_above_the_cpu_count(monkeypatch):
    """B r2 C6: _THREAD_SWEEP's own comment says counts above os.cpu_count() are skipped."""
    monkeypatch.setattr(D.os, "cpu_count", lambda: 8)
    assert D._sweep_counts() == [1, 2, 4, 8]


def test_benchmark_refs_come_from_the_sample_file_only(tmp_path):
    """B r2 C5: the sample whose SHA-256 is recorded IS the population benchmarked."""
    sample = tmp_path / "s.json"
    sample.write_text('[{"provider": "idsse", "match_id": "DFL-MAT-J03WMX"}]')
    assert D._benchmark_match_ids(sample) == {"idsse": ["DFL-MAT-J03WMX"]}


def test_path_subprocess_refuses_the_wrong_engine(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        assert "PYTHONPATH" not in kw["env"]  # the subprocess must import its OWN silly-kicks
        (Path(cmd[3]) / "timing.json").write_text('{"add_das_s": 1.0, "das_xfns_s": 1.0, "silly_kicks": "4.127.0", "native": true}')

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "x")
    with pytest.raises(SystemExit, match="old path"):
        D._path_subprocess(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), python="py", repeat=1, expect_native=False)


def test_summarize_benchmark_computes_the_spec_4_2_figures():
    row = {
        "provider": "gs", "n_scored_frames": 100,
        "ms_frame_ref": 30.0, "ms_frame_numpy": 10.0, "ms_frame_numba_serial": 2.0, "ms_frame_numpy_periodic": 11.0,
        "numba_threads_ms_frame": {str(k): 2.0 / (0.8 * k) if k > 1 else 2.0 for k in D._THREAD_SWEEP},
        "add_das_new_s": 1.0, "add_das_old_s": 20.0, "das_xfns_new_s": 1.0, "das_xfns_old_s": 60.0,
        "paired_s": 2.0, "independent_s": 2.0,
    }
    s = D.summarize_benchmark([row, dict(row, provider="sk")])
    assert s["ref_over_numba_serial"] == 15.0 and s["ref_over_numpy"] == 3.0
    assert abs(s["prange_efficiency"]["16"] - 0.8) < 1e-12
    assert s["add_das_speedup"] == 20.0 and s["das_xfns_speedup"] == 60.0
    assert s["paired_over_independent"] == 1.0
    assert s["seconds_per_frame"] == {"numba_serial": 0.002, "numpy": 0.011}
    assert s["n_matches"] == 2


def test_foreign_cpu_fraction():
    # 10 s elapsed on 4 cpus = 40 cpu-s; machine busy 30 cpu-s, of which 20 were ours -> 10/40 foreign
    assert D._foreign_cpu_fraction(busy0=0.0, busy1=30.0, own0=0.0, own1=20.0, elapsed=10.0, ncpu=4) == 0.25
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Create `scripts/_das_path_timing.py`:**

```python
"""add_das / das_xfns per-match timing, for whichever silly-kicks the RUNNING interpreter imports.

The parity driver's --benchmark runs this file as a subprocess in TWO pandas-2 interpreters (spec 12
D1): the OLD path (released silly-kicks 4.127.0 + accessible-space 2.0.15) and the NEW path (this
cycle's C1 installed with pandas<3), so the ratios measure the engine, not a pandas major. The driver
strips PYTHONPATH and checks the recorded `native` marker. Top-level imports are stdlib + pandas only.

    <python> _das_path_timing.py <in_dir> <out_dir> <repeat>
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd

_T = TypeVar("_T")


def best_of(fn: Callable[[], _T], repeat: int) -> tuple[_T, float]:
    """``(result, best_seconds)`` over ``max(1, repeat)`` calls of ``fn`` -- the minimum wall time;
    the result is the last call's. (A stdlib-only copy: this file runs under the OLD-path interpreter,
    which cannot import the repo's scripts.)"""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    for _ in range(max(1, repeat) - 1):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def time_add_das_and_xfns(frames: pd.DataFrame, actions: pd.DataFrame, *, repeat: int, warmup: bool) -> dict:
    """Best-of-``repeat`` seconds for ``add_das`` (links precomputed, untimed) and ``das_xfns`` on one match."""
    import silly_kicks.tracking.features as feats
    import silly_kicks.tracking.utils as tu
    import silly_kicks.vaep.feature_framework as ff

    links, _report = tu.link_actions_to_frames(actions, frames)
    states = ff.gamestates(actions, nb_prev_actions=3)

    def _add():
        return feats.add_das(actions, frames, links=links)

    def _xfn():
        return feats.das_xfns[0](states, frames)

    if warmup:  # JIT and first-call caches stay out of the timed region on both paths
        _add()
        _xfn()
    _, add_s = best_of(_add, repeat)
    _, xfn_s = best_of(_xfn, repeat)
    return {"add_das_s": add_s, "das_xfns_s": xfn_s}


def _main(in_dir: str, out_dir: str, repeat: int) -> None:
    frames = pd.read_parquet(Path(in_dir) / "frames.parquet")
    actions = pd.read_parquet(Path(in_dir) / "actions.parquet")
    timing = time_add_das_and_xfns(frames, actions, repeat=repeat, warmup=True)
    timing.update(
        {
            "repeat": repeat,
            "silly_kicks": importlib.metadata.version("silly-kicks"),
            # the native engine exists only after ADR-107: the discriminator between the two paths
            "native": importlib.util.find_spec("silly_kicks.tracking._das_engine") is not None,
            "pandas": pd.__version__,
        }
    )
    (Path(out_dir) / "timing.json").write_text(json.dumps(timing), encoding="utf-8")


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 4: Driver benchmark** — add below `_population` in `scripts/validate_das_native_parity.py`. Add `import hashlib`, `import os` and `import sys` if absent. `_reference_leg_subprocess` already gained `repeat` in Task 10.

```python
_PATH_TIMING_MODULE = Path(__file__).resolve().parent / "_das_path_timing.py"
_COORD_COLS = ("x", "y", "z", "vx", "vy", "speed", "x_smoothed", "y_smoothed")
_FOREIGN_CPU_MAX = 0.05  # contention gate: other processes' CPU over the whole benchmark


def _sweep_counts() -> list[int]:
    """The `_THREAD_SWEEP` counts this box can run (numba refuses more threads than CPUs)."""
    return [k for k in _THREAD_SWEEP if k <= (os.cpu_count() or 1)]


def _benchmark_match_ids(sample_path: Path) -> dict[str, list[str]]:
    """The benchmark population IS the sample file whose SHA-256 the artifact records (list-matches shape)."""
    return _load_match_ids(json.loads(Path(sample_path).read_text(encoding="utf-8")))


def _old_path_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """The frames as the pre-F1b (4.127.0) path stored them: float64 coords, no category ids."""
    out = frames.copy()
    for c in _COORD_COLS:
        if c in out.columns:
            out[c] = out[c].astype("float64")
    for c in ("team_id", "player_id"):
        if c in out.columns and isinstance(out[c].dtype, pd.CategoricalDtype):
            integer = pd.api.types.is_integer_dtype(out[c].cat.categories.dtype)
            out[c] = out[c].astype(object).astype("Int64") if integer else out[c].astype(object)
    return out


def _path_subprocess(frames, actions, *, python: str, repeat: int, expect_native: bool) -> dict:
    """Time add_das / das_xfns in a pandas-2 interpreter; refuse unless it ran the expected engine."""
    d = Path(tempfile.mkdtemp(prefix="das_path_"))
    try:
        frames.to_parquet(d / "frames.parquet")
        actions.to_parquet(d / "actions.parquet")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run(  # noqa: S603 -- resolved prerequisite interpreter + our own module path
            [python, str(_PATH_TIMING_MODULE), str(d), str(d), str(repeat)], check=True, env=env
        )
        timing = json.loads((d / "timing.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if bool(timing.get("native")) is not expect_native:
        which = "new" if expect_native else "old"
        raise SystemExit(f"the {which} path interpreter ran native={timing.get('native')} (silly-kicks {timing.get('silly_kicks')})")
    return timing


def _keeper_counterfactual(scored: pd.DataFrame, shift_m: float = 1.0) -> tuple[pd.DataFrame, bool]:
    """The gkdv-shaped pair leg: every defending keeper moved +shift_m in x (kinematics only)."""
    from silly_kicks.id_compat import ids_differ

    cf = scored.copy()
    tip = cf["team_in_possession"]
    moved = (
        cf["is_goalkeeper"].fillna(False).astype(bool).to_numpy()
        & ~cf["is_ball"].astype(bool).to_numpy()
        & tip.notna().to_numpy()
        & ids_differ(cf["team_id"], tip).to_numpy()
    )
    if moved.any():
        cf.loc[moved, "x"] = (cf.loc[moved, "x"].astype("float64") + shift_m).astype(cf["x"].dtype)
    return cf, bool(moved.any())


def _bench_match(item, *, reference_python: str, old_path_python: str, new_path_python: str, repeat: int) -> dict:
    """One match's spec 4.2 legs, best-of-``repeat`` after a warm-up. No match id is recorded."""
    from silly_kicks.tracking._das import get_individual_das, individual_das_paired
    from silly_kicks.tracking._das_engine import compute_das
    from silly_kicks.tracking._das_pack import pack_frames

    provider, _match_id, actions, frames = item
    scored = _direction_column(_prepare_possession(_scored_frames(actions, frames)), direction_col=None)
    n = int(scored[list(_FRAME_KEYS)].drop_duplicates().shape[0])
    row: dict = {"provider": provider, "n_scored_frames": n}
    if n == 0:
        return row
    ref = _reference_leg_subprocess(scored, reference_python=reference_python, repeat=repeat)
    row["ms_frame_ref"] = 1e3 * ref["compute_s"] / n
    _, t = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numpy"), repeat=repeat)
    row["ms_frame_numpy"] = 1e3 * t / n
    _run_native(scored, _REFERENCE_PARAMS, engine="numba")  # JIT warm-up, untimed
    _, t = _time_leg(lambda: _run_native(scored, _REFERENCE_PARAMS, engine="numba"), repeat=repeat)
    row["ms_frame_numba_serial"] = 1e3 * t / n
    packed = pack_frames(scored, attacking_direction_col=_DIR_COL)
    _, t = _time_leg(lambda: compute_das(packed, DAS_PARAMS, engine="numpy"), repeat=repeat)
    row["ms_frame_numpy_periodic"] = 1e3 * t / n
    threads = {}
    for k in _sweep_counts():
        compute_das(packed, DAS_PARAMS, engine="numba", n_threads=k)  # prange compiles separately
        _, t = _time_leg(lambda k=k: compute_das(packed, DAS_PARAMS, engine="numba", n_threads=k), repeat=repeat)
        threads[str(k)] = 1e3 * t / n
    row["numba_threads_ms_frame"] = threads
    old_frames = _old_path_frames(frames)
    new = _path_subprocess(old_frames, actions, python=new_path_python, repeat=repeat, expect_native=True)
    old = _path_subprocess(old_frames, actions, python=old_path_python, repeat=repeat, expect_native=False)
    row.update({"add_das_new_s": new["add_das_s"], "das_xfns_new_s": new["das_xfns_s"],
                "add_das_old_s": old["add_das_s"], "das_xfns_old_s": old["das_xfns_s"],
                "pandas_new": new["pandas"], "pandas_old": old["pandas"]})
    cf, any_moved = _keeper_counterfactual(scored)
    if any_moved:
        _, row["paired_s"] = _time_leg(
            lambda: individual_das_paired(scored, cf, attacking_direction_col=_DIR_COL), repeat=repeat
        )
        _, row["independent_s"] = _time_leg(
            lambda: (get_individual_das(scored, attacking_direction_col=_DIR_COL),
                     get_individual_das(cf, attacking_direction_col=_DIR_COL)),
            repeat=repeat,
        )
    return row


def summarize_benchmark(rows: list[dict]) -> dict:
    """The spec 4.2 figures (pure). ms/frame legs: ratio of medians; per-match legs: median ratio."""
    ok = [r for r in rows if r.get("n_scored_frames")]

    def med(key):
        return float(np.median([r[key] for r in ok]))

    ref, npy, nb = med("ms_frame_ref"), med("ms_frame_numpy"), med("ms_frame_numba_serial")
    counts = sorted({k for r in ok for k in r["numba_threads_ms_frame"]}, key=int)  # those the box could run
    tk = {k: float(np.median([r["numba_threads_ms_frame"][k] for r in ok])) for k in counts}
    paired = [r["paired_s"] / r["independent_s"] for r in ok if "paired_s" in r]
    return {
        "ref_over_numba_serial": ref / nb,
        "ref_over_numpy": ref / npy,
        "prange_efficiency": {k: tk["1"] / (int(k) * v) for k, v in tk.items()},
        "add_das_speedup": float(np.median([r["add_das_old_s"] / r["add_das_new_s"] for r in ok])),
        "das_xfns_speedup": float(np.median([r["das_xfns_old_s"] / r["das_xfns_new_s"] for r in ok])),
        "paired_over_independent": float(np.median(paired)) if paired else None,
        "ms_frame_median": {"ref": ref, "numpy": npy, "numba_serial": nb},
        "seconds_per_frame": {"numba_serial": tk["1"] / 1e3, "numpy": med("ms_frame_numpy_periodic") / 1e3},
        "n_matches": len(ok),
    }


def _foreign_cpu_fraction(*, busy0: float, busy1: float, own0: float, own1: float, elapsed: float, ncpu: int) -> float:
    """Other processes' share of the machine over the run: (machine busy - own) / (elapsed * ncpu)."""
    return max(0.0, (busy1 - busy0) - (own1 - own0)) / (elapsed * ncpu)


def _cpu_snapshot() -> tuple[float, float] | None:
    """(machine busy cpu-seconds from /proc/stat, own+children cpu-seconds); None off Linux."""
    if sys.platform == "win32":
        return None
    import resource

    fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
    vals = [int(v) for v in fields]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    busy = (sum(vals) - idle) / os.sysconf("SC_CLK_TCK")
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return busy, me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def _peak_rss_bytes() -> dict | None:
    if sys.platform == "win32":
        return None
    import resource

    return {
        "self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }


def _loadavg() -> float | None:
    return None if sys.platform == "win32" else os.getloadavg()[0]


def run_benchmark(refs, load, dest: Path, *, prov: dict, reference_python: str, old_path_python: str,
                  new_path_python: str, repeat: int, sample_sha256: str) -> dict:
    """The benchmark artifact (spec 12 D1): run ALONE after every other process. Provider-only rows."""
    snap0, t0, load0 = _cpu_snapshot(), time.perf_counter(), _loadavg()
    rows = [
        _bench_match(load(ref), reference_python=reference_python, old_path_python=old_path_python,
                     new_path_python=new_path_python, repeat=repeat)
        for ref in refs
    ]
    snap1, elapsed = _cpu_snapshot(), time.perf_counter() - t0
    foreign = (
        _foreign_cpu_fraction(busy0=snap0[0], busy1=snap1[0], own0=snap0[1], own1=snap1[1], elapsed=elapsed, ncpu=os.cpu_count() or 1)
        if snap0 and snap1 else None
    )
    out = {
        "summary": summarize_benchmark(rows),
        "per_match": rows,
        "sample_sha256": sample_sha256,
        "repeat": repeat,
        "thread_sweep": _sweep_counts(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "contention": {"foreign_cpu_fraction": foreign, "max": _FOREIGN_CPU_MAX, "loadavg_1m": {"before": load0, "after": _loadavg()}},
        "reference_env": _probe_reference_env(reference_python),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "run_platform": prov.get("platform"),
        "run_machine": prov.get("machine"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "performance.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
```

**CLI.** Add these arguments:

```python
    ap.add_argument("--benchmark", action="store_true", help="write performance.json over --benchmark-sample-json (run ALONE)")
    ap.add_argument("--benchmark-sample-json", default=None, help="the fixed sample ([{provider, match_id}, ...])")
    ap.add_argument("--old-path-python", default=None, help="silly-kicks 4.127.0 + accessible-space 2.0.15 + pandas<3 (else $SK_DAS_OLDPATH_PYTHON)")
    ap.add_argument("--new-path-python", default=None, help="this checkout installed with pandas<3 (else $SK_DAS_NEWPATH_PYTHON)")
    ap.add_argument("--repeat", type=int, default=3, help="best-of-N timing repeats (benchmark only)")
```

In `main()`, after `def _load(ref): …` and before the reduce branch:

```python
    if args.benchmark:
        if not args.benchmark_sample_json:
            raise SystemExit("--benchmark needs --benchmark-sample-json (a fixed, seeded sample)")
        old_python = args.old_path_python or os.environ.get("SK_DAS_OLDPATH_PYTHON")
        new_python = args.new_path_python or os.environ.get("SK_DAS_NEWPATH_PYTHON")
        if not old_python or not new_python:
            raise SystemExit("--benchmark needs both pandas-2 path interpreters (old 4.127.0, new = this checkout)")
        out = run_benchmark(refs, _load, dest, prov=prov, reference_python=_resolve_reference_python(args.reference_python),
                            old_path_python=old_python, new_path_python=new_python, repeat=args.repeat,
                            sample_sha256=hashlib.sha256(Path(args.benchmark_sample_json).read_bytes()).hexdigest())
        print(json.dumps(out["summary"], indent=2, default=str))
        return
```

The benchmark population is the sample file itself (B r2 C5). In `main()`, where `match_ids` is computed (before `pining_source`), add:

```python
    if args.benchmark:
        if args.match_ids_json:
            raise SystemExit("--benchmark takes its population from --benchmark-sample-json only; drop --match-ids-json")
        if not args.benchmark_sample_json:
            raise SystemExit("--benchmark needs --benchmark-sample-json (a fixed, seeded sample)")
        match_ids = _benchmark_match_ids(Path(args.benchmark_sample_json))
```

`refs` is then exactly the sample, and the recorded `sample_sha256` names it.
- [ ] **Step 5: Run — expect PASS:** `python -m pytest tests/scripts/test_das_path_timing.py tests/scripts/test_das_native_parity_driver.py tests/scripts/test_das_reference_leg.py tests/scripts/test_provenance_wiring.py -q`, then the ASCII gate.

### Task 12: Loader download race fix

**Files:** Modify `scripts/_loader_pining.py` (`_download_to_temp` `:160-167`). Test `tests/calibration/test_loader_pining.py` (append).

- [ ] **Step 1: Failing test:**

```python
def test_partial_download_name_is_unique_per_call(tmp_path):
    """Concurrent workers fetching the same artifact must never share a temp file (spec 0.12)."""
    from scripts._loader_pining import _partial_path

    dest = tmp_path / "tracking.xml"
    a, b = _partial_path(dest), _partial_path(dest)
    assert a != b and a.parent == b.parent == dest.parent
    assert a.name.startswith("tracking.xml.") and a.name.endswith(".partial")
```

- [ ] **Step 2: Run — FAIL** (ImportError).
- [ ] **Step 3: Implement.**
- Add the helper (imports `os` and `uuid` at the top of the file):

```python
def _partial_path(dest: Path) -> Path:
    """A per-call temp name beside ``dest``: two processes fetching the same artifact never interleave
    writes into one file; each atomically replaces ``dest`` with identical bytes (spec 0.12)."""
    return dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
```

- In `_download_to_temp`, replace `partial = dest.with_name(dest.name + ".partial")` with `partial = _partial_path(dest)`. `partial.replace(dest)` stays (atomic).
- [ ] **Step 4: Run — PASS:** `python -m pytest tests/calibration/test_loader_pining.py tests/scripts/test_loader*.py -q`.

### Task 12b: Launcher wiring (D3)

**Files:** Modify `scripts/_parallel_launch.py` (`_launch` `:149-155`; `_done_marker_for` `:306-322`), `scripts/train_xshot_occurrence.py` and `scripts/train_xcross_attempt.py` (argparse + the early `--study/--assemble` dispatch + the end of `main`). Tests: `tests/scripts/test_parallel_launch.py` (append), `tests/scripts/test_expect_variant_wiring.py` (append).

**Interfaces (produced):**
- Launcher:
  - a `{worker}` token (the worker id `w<i>`) substituted beside `{subset}`;
  - `--mode das` accepts `--corpus-json` in the DAS driver's `--list-matches` shape (`[{provider, match_id}]`); subsets are written in that shape;
  - the done-key is `join_key((provider, match_id))`, satisfied by a shard **or** an `.excluded.json` marker inside the generation `--das-generation` names (required with `--mode das`; consumes Task 10's `--print-generation`);
  - `_done_marker_for(mode, shard_root, *, das_generation=None)`.
- Trainers (both):
  - `--prep-only`: extract + persist the study inputs, print `{"study_root", "studies"}`, and exit without assembling;
  - `--list-studies`: with `--shard-root`, print `enumerate_studies` as JSON;
  - `--study-list FILE`: with `--shard-root`, run each tag in the JSON list via `run_one_study`.

- [ ] **Step 1: Failing tests.** `tests/scripts/test_parallel_launch.py` (append):

```python
def test_worker_token_is_substituted(tmp_path):
    driver = tmp_path / "d.py"
    driver.write_text(
        "import sys, pathlib, json\n"
        "root = pathlib.Path(sys.argv[1]); worker = sys.argv[3]\n"
        "for i in json.loads(pathlib.Path(sys.argv[2]).read_text()):\n"
        "    (root / (i + '.done')).write_text(worker)\n",
        encoding="utf-8",
    )
    root = tmp_path / "s"
    root.mkdir()
    pl.run_parallel(
        cmd_template=[sys.executable, str(driver), str(root), "{subset}", "{worker}"],
        subsets={"w0": ["a"], "w1": ["b"]},
        cap_bytes=1 << 40, backend="none", peak_rss_bytes=1, headroom_bytes=0,
        done_marker=lambda i: root / (i + ".done"), shard_root=root, poll_interval=0.1, mem_available_bytes=10**12,
    )
    assert (root / "a.done").read_text() == "w0" and (root / "b.done").read_text() == "w1"


def test_das_done_marker_takes_list_matches_items_and_exclusions_in_the_expected_generation_only(tmp_path):
    gen, stale = tmp_path / "gen1", tmp_path / "gen0"
    gen.mkdir()
    stale.mkdir()
    (gen / "skillcorner__1.parquet").write_bytes(b"x")
    (gen / "idsse__DFL-MAT-J03WMX.excluded.json").write_text("{}")
    (stale / "skillcorner__2.parquet").write_bytes(b"x")  # another commit's / token's generation
    done = pl._done_marker_for("das", tmp_path, das_generation="gen1")
    assert done({"provider": "skillcorner", "match_id": "1"}).exists()
    assert done({"provider": "idsse", "match_id": "DFL-MAT-J03WMX"}).exists()  # an exclusion is done
    # B r3 CCC-PLAN-28: a shard in a stale generation is NOT done -- the reduce would refuse it after the wave.
    assert not done({"provider": "skillcorner", "match_id": "2"}).exists()


def test_das_mode_requires_the_generation(tmp_path):
    with pytest.raises(SystemExit):
        pl._done_marker_for("das", tmp_path)


def test_das_subset_file_is_written_in_list_matches_shape(tmp_path):
    driver = tmp_path / "d.py"
    driver.write_text(
        "import sys, pathlib, json\n"
        "gen = pathlib.Path(sys.argv[1]) / 'g'\n"
        "gen.mkdir(exist_ok=True)\n"
        "for e in json.loads(pathlib.Path(sys.argv[2]).read_text()):\n"
        "    (gen / f\"{e['provider']}__{e['match_id']}.parquet\").write_bytes(b'x')\n",
        encoding="utf-8",
    )
    root = tmp_path / "s"
    root.mkdir()
    items = [{"provider": "skillcorner", "match_id": "1"}, {"provider": "idsse", "match_id": "X"}]
    res = pl.run_parallel(
        cmd_template=[sys.executable, str(driver), str(root), "{subset}"],
        subsets=pl.split_round_robin(items, 2),
        cap_bytes=1 << 40, backend="none", peak_rss_bytes=1, headroom_bytes=0,
        done_marker=pl._done_marker_for("das", root, das_generation="g"), shard_root=root, poll_interval=0.1,
        mem_available_bytes=10**12,
    )
    assert res.completed == 2


def test_cli_das_mode_without_a_generation_is_refused(tmp_path):
    corpus = tmp_path / "c.json"
    corpus.write_text("[]")
    with pytest.raises(SystemExit):
        pl.main(["--mode", "das", "--driver", "x {subset}", "--corpus-json", str(corpus), "--shard-root", str(tmp_path),
                 "--peak-rss-gib", "1"])
```

`tests/scripts/test_expect_variant_wiring.py` (append):

```python
@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_study_list_fan_out_equals_the_serial_assemble(trainer, tmp_path):
    """D3 (launcher Task 9 in miniature): --list-studies + --study-list workers + --assemble == serial assemble,
    byte-identical booster -- the CLI path the launcher's f1b mode drives."""
    import json

    from scripts._study_shared import persist_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    corpus = P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    X, y, groups, providers, match_ids, is_public = corpus()

    def _persist(root):
        persist_study_inputs(root, X=X, y=y, groups=groups, providers=providers, match_ids=match_ids, is_public=is_public,
                             config={"n_trials": 2, "negative_subsample": None, "seed": 42, "feature_set": "faithful",
                                     "horizon_seconds": 1.0, "study_db_dir": str(root), "artifact_dir": str(root / "art"),
                                     "run_paired": True, "run_prov": {"commit": "t", "dirty": False, "tree_state": "clean"},
                                     "ship_variant": None, "expect_variant": None})

    kw = {"run_probe": False} if trainer is xc else {}
    serial, fan = tmp_path / "serial", tmp_path / "fan"
    _persist(serial)
    _persist(fan)
    _m, model_serial = trainer.assemble_studies(serial, study_shard_dir=serial, **kw)
    tags = trainer.enumerate_studies(fan)
    for i, half in enumerate((tags[::2], tags[1::2])):
        f = tmp_path / f"tags{i}.json"
        f.write_text(json.dumps(half))
        trainer.main(["--shard-root", str(fan), "--study-list", str(f)])
    _m2, model_fan = trainer.assemble_studies(fan, study_shard_dir=fan, **kw)
    assert model_serial._booster.save_raw("json") == model_fan._booster.save_raw("json")


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_prep_only_persists_the_inputs_prints_the_studies_and_never_assembles(trainer, tmp_path, monkeypatch, capsys):
    """D3 (B r3 CCC-PLAN-27): the launcher's f1b mode starts from --prep-only, so a defect here must fail CI,
    not the DGX. Extraction is faked with the existing PAIRED study fixture (a public-only corpus enumerates
    no studies, which would make the check vacuous); visibility, persistence and enumeration are real."""
    import json

    from scripts._study_shared import load_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    X, y, groups, providers, match_ids, is_public = (
        P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    )()
    pairs = sorted({(str(p), str(m), bool(pub)) for p, m, pub in zip(providers, match_ids, is_public)})
    vis = {(p, m): ("public" if pub else "private") for p, m, pub in pairs}
    allow: dict[str, list[str]] = {}
    for p, m, _pub in pairs:
        allow.setdefault(p, []).append(m)
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))
    monkeypatch.setattr(
        _loader_pining,
        "pining_source",
        lambda provs, match_ids=None, **kw: ([(p, m) for p in provs for m in (match_ids or {}).get(p, [])], None),
    )

    def fake_extract(source, horizon, **kw):
        base = (X, y, groups, providers, match_ids)
        return (*base, (xc._new_probe_cohort(), xc._new_probe_cohort(), 0)) if trainer is xc else base

    monkeypatch.setattr(trainer, "_extract", fake_extract)
    monkeypatch.setattr(trainer, "assemble_studies", lambda *a, **k: pytest.fail("--prep-only must not assemble"))
    # The fixture's "public" ids (m0, m1, m2) are synthetic, so the real PUBLIC_CORPUS registry check refuses
    # them ("UNREGISTERED public match(es)", B r4 CCC-PLAN-38). That check is not under test here (Task 2 tests
    # G1 and the registry); stub ONLY it, on the module object the trainers' function-local import resolves.
    import _corpus  # the scripts/ module object, as `_loader_pining` above

    monkeypatch.setattr(_corpus, "assert_public_corpus", lambda *a, **k: None)
    allow_file = tmp_path / "allow.json"
    allow_file.write_text(json.dumps(allow))
    trainer.main(["--providers", ",".join(sorted(allow)), "--match-ids-json", str(allow_file), "--max-per-provider", "10",
                  "--n-trials", "1", "--output-dir", str(tmp_path / "o"), "--allow-dirty", "--prep-only"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    root = Path(out["study_root"])
    assert root.name == "studies" and (tmp_path / "o") in root.parents
    assert out["studies"]  # paired corpus -> a real fan-out list, not the vacuous []
    assert out["studies"] == trainer.enumerate_studies(root)
    assert list(load_study_inputs(root).X.columns) == list(X.columns)


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_list_studies_prints_the_tags(trainer, tmp_path, capsys):
    import json

    from scripts._study_shared import persist_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    corpus = P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    X, y, groups, providers, match_ids, is_public = corpus()
    persist_study_inputs(tmp_path, X=X, y=y, groups=groups, providers=providers, match_ids=match_ids, is_public=is_public,
                         config={"run_paired": True, "n_trials": 1, "negative_subsample": None, "seed": 42})
    trainer.main(["--shard-root", str(tmp_path), "--list-studies"])
    assert json.loads(capsys.readouterr().out) == trainer.enumerate_studies(tmp_path)
```

`_synthetic_paired_corpus` (xshot columns, `:33`) and `_synthetic_paired_corpus_xcross` (`:126`) are the existing study-parity fixtures in `tests/scripts/test_train_study_parallel_parity.py`. Check their return order matches `(X, y, groups, providers, match_ids, is_public)`; the xshot one does (`:60-66`).
- [ ] **Step 2: Run — expect FAIL** (`{worker}` left literal; `unexpected keyword argument 'das_generation'`; `unrecognized arguments: --study-list` / `--prep-only`).
- [ ] **Step 3: Implement the launcher.**
  - In `_launch`: `argv = [a.replace("{subset}", str(subset_file)).replace("{worker}", w) for a in cmd_template]`.
  - Replace `_done_marker_for`:

```python
def _done_marker_for(mode: str, shard_root: Path, *, das_generation: str | None = None) -> Callable[[object], Path]:
    """Map a corpus item to the path whose existence means 'done' for that driver.

    f1b: the study shard ``<tag>.study.json`` (exact). das: an item is a ``--list-matches`` entry
    (``{"provider", "match_id"}``) or an already-joined key; done = its ``<key>.parquet`` shard OR its
    ``<key>.excluded.json`` marker (a decided exclusion is done, ADR-052 D13) inside the ONE generation the
    driver will write, ``das_generation`` (``validate_das_native_parity.py --print-generation``). A shard in
    any other generation is not done: the commit-keyed reduce would refuse it (B r3 CCC-PLAN-28).
    """
    if mode == "f1b":
        return lambda i: shard_root / f"{i}.study.json"
    if not das_generation:
        raise SystemExit("--mode das needs --das-generation (validate_das_native_parity.py --print-generation)")

    from scripts._driver import exclusion_path, join_key, shard_path

    gen = shard_root / das_generation

    def _das(i) -> Path:
        key = join_key((i["provider"], i["match_id"])) if isinstance(i, dict) else str(i)
        shard = shard_path(gen, key)
        return shard if shard.is_file() else exclusion_path(gen, key)

    return _das
```

  - `_build_parser()` gains `--das-generation` (help: "the shard generation the DAS driver writes, from `validate_das_native_parity.py --print-generation`; required with --mode das").
  - `main()`, directly after `parse_args` and before reading anything: `if args.mode == "das" and not args.das_generation: parser.error("--mode das needs --das-generation")` (keep the parser in a local `parser = _build_parser()`). It passes `das_generation=args.das_generation` to `_done_marker_for`.
  - `main()`'s `--corpus-json` help text: "JSON list of corpus items (das: the driver's --list-matches output; f1b: study tags)".
  - The `{worker}` token is documented in `--driver`'s help.
  - `shard_path` / `exclusion_path` accept the joined key string: `join_key` returns a string key unchanged (`scripts/_driver.py:200-203`).
- [ ] **Step 4: Implement the trainer flags** (identical in both trainers).
  - argparse, after `--assemble`:

```python
    ap.add_argument("--prep-only", action="store_true",
                    help="extract + persist the study inputs, print {study_root, studies}, exit (launcher f1b mode)")
    ap.add_argument("--list-studies", action="store_true", help="print the study tags under --shard-root as JSON")
    ap.add_argument("--study-list", default=None, help="JSON list of study tags to run from --shard-root (launcher worker)")
```

  - The early dispatch becomes:

```python
    if args.study or args.assemble or args.study_list or args.list_studies:
        if not args.shard_root:
            ap.error("--study/--study-list/--list-studies/--assemble require --shard-root")
        root = Path(args.shard_root)
        if args.list_studies:
            print(json.dumps(enumerate_studies(root)))
        elif args.study_list:
            for tag in json.loads(Path(args.study_list).read_text(encoding="utf-8")):
                run_one_study(root, tag)
        elif args.study:
            run_one_study(root, args.study)
        else:
            try:
                assemble_studies(root, study_shard_dir=root)  # xcross: keep its existing run_probe=True
            except AcceptanceGatesFailedError:
                sys.exit(1)
        return
```

  - At the end of `main`, directly after `persist_study_inputs(...)` and before the `assemble_studies` call:

```python
    if args.prep_only:
        print(json.dumps({"study_root": str(study_root), "studies": enumerate_studies(study_root)}))
        return
```

- [ ] **Step 5: Run — expect PASS:** `python -m pytest tests/scripts/test_parallel_launch.py tests/scripts/test_expect_variant_wiring.py tests/scripts/test_train_study_parallel_parity.py -q` (including slow).

### Task 13: das-native leftovers + docs

**Files:**
- Create `tests/tracking/test_das_benchmark.py`.
- Modify:
  - ADR-107 (`:61-68`); ADR-108 (Consequences: reference-leg note); ADR-106 (amendment);
  - `docs/context/tracking-features.md` (reference-leg note);
  - `silly_kicks/tracking/_das.py` (`:54-61`, `:105-124`);
  - `tests/tracking/test_das_cost_guardrail.py`;
  - `.gitattributes`.

- [ ] **Step 1: `tests/tracking/test_das_benchmark.py`.** It runs only under `--benchmark-only` (CI `benchmark` job). The numba cases skip where numba is absent, which is the repo's numba pattern.

```python
"""pytest-benchmark ms/frame per DAS engine (das-native plan Task 5; spec section 8).

Trend data only -- no timing assertion. The corpus figures that gate the spec section 4.2 targets are
docs/research/das_native_parity/performance.json (test_das_parity_artifact.py).
"""

import pandas as pd
import pytest

from silly_kicks.tracking._das_engine import compute_das
from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_helpers import single_frame

_N_FRAMES = 200


@pytest.fixture(scope="module")
def packed():
    frames = pd.concat([single_frame(frame=f, seed=f) for f in range(_N_FRAMES)], ignore_index=True)
    return pack_frames(frames, attacking_direction_col="dir")


def test_numpy_engine(benchmark, packed):
    benchmark(compute_das, packed, DAS_PARAMS, engine="numpy")


def test_numba_serial(benchmark, packed):
    pytest.importorskip("numba")
    compute_das(packed, DAS_PARAMS, engine="numba")  # JIT warm-up outside the timed region
    benchmark(compute_das, packed, DAS_PARAMS, engine="numba")


@pytest.mark.parametrize("n_threads", [2, 4])
def test_numba_prange(benchmark, packed, n_threads):
    pytest.importorskip("numba")
    compute_das(packed, DAS_PARAMS, engine="numba", n_threads=n_threads)
    benchmark(compute_das, packed, DAS_PARAMS, engine="numba", n_threads=n_threads)
```

Check it two ways:
- `python -m pytest tests/tracking/test_das_benchmark.py --benchmark-only -q` reports 4 benchmarks;
- `--benchmark-skip -q` gives 4 skipped.
- [ ] **Step 2: Chunk-size table** (das-native plan Task 5 Step 5). Measure in a scratch script under the session scratchpad (not committed):
  - 20 000 `single_frame` frames;
  - numpy `chunk_size ∈ {16, 32, 64, 128}`; numba `{512, 1024, 4096, 8192}`;
  - best-of-3 wall time and the tracemalloc working peak (`tests/tracking/test_das_scale_memory.py:43-54` recipe);
  - numba serial and `n_threads ∈ {1, 2, 4, 8, 16}` ms/frame.

  Per engine, pick the fastest chunk whose peak stays under 256 MB. If it differs from the current default, change the default in `_das_engine` — value-neutral; run `tests/tracking/test_das_invariance.py`. Write the table into ADR-107 **Dispatch / chunking** with the machine and date.
- [ ] **Step 3: Per-engine constant (D6) — failing test first** in `tests/tracking/test_das_cost_guardrail.py`:

```python
def test_estimate_das_cost_uses_the_engine_that_will_run(monkeypatch):
    import silly_kicks.tracking._das as das_mod
    import silly_kicks.tracking._das_engine as eng

    frames = _frames(10)
    monkeypatch.setattr(eng, "_numba_available", lambda: True)
    assert estimate_das_cost(frames) == pytest.approx(10 * das_mod._DAS_SECONDS_PER_FRAME)
    monkeypatch.setattr(eng, "_numba_available", lambda: False)
    assert estimate_das_cost(frames) == pytest.approx(10 * das_mod._DAS_SECONDS_PER_FRAME_NUMPY)
```

  The existing tests in that file that assume the numba constant (`test_estimate_das_cost_is_pure_and_scales_with_distinct_frames`, `test_estimate_das_cost_is_thread_aware`) pin `eng._numba_available` to `True` with monkeypatch, so they are independent of the CI leg. Run the tests: FAIL.

  Implement in `_das.py`:
  - add `_DAS_SECONDS_PER_FRAME_NUMPY = <numpy ms/frame from Step 2 / 1e3, rounded up to 2 significant figures>`, with a comment citing Step 2;
  - set `_DAS_SECONDS_PER_FRAME` / `_PRANGE_EFFICIENCY` from the same measurement (numba serial; efficiency at 16 threads, rounded down to 2 decimals);
  - change `estimate_das_cost` to:

```python
    from silly_kicks.tracking import _das_engine

    if not _das_engine._numba_available():
        return _n_distinct_frames(frames) * _DAS_SECONDS_PER_FRAME_NUMPY  # numpy has no prange path
    serial = _n_distinct_frames(frames) * _DAS_SECONDS_PER_FRAME
    if n_threads is not None and n_threads > 1:
        return serial / (n_threads * _PRANGE_EFFICIENCY)
    return serial
```

  The docstring's first line becomes "engine- and thread-aware". `test_das_kernel.py`'s lazy-import test must stay green. Run: `python -m pytest tests/tracking/test_das_cost_guardrail.py tests/tracking/test_das.py tests/tracking/test_das_kernel.py -q` — PASS.
- [ ] **Step 4: Docs.**
  - **ADR-107 `:67-68`:** replace the frame-count sentence with the current behaviour: an estimated-seconds budget (`_DAS_COST_WARN_SECONDS`), engine- and thread-aware per-frame constants measured as in the table, re-derived from `performance.json` at release; reported, never gating; values unchanged.
  - **ADR-108 Consequences and `docs/context/tracking-features.md`:** one paragraph each on the pandas-2 reference leg (`_das_reference_leg.py`, `SK_DAS_REFERENCE_PYTHON`, why pandas 3 Copy-on-Write disables the library's offside step, the collision-free frame key), per pandas-2 reference spec §7.
  - **ADR-106:** append an amendment with:
    - the final anchors (`3ca609f` reused; C1 for the re-fits, written as "commit 1 of the completion cycle");
    - the corpus incident (owner token sourced for every run; spec §0);
    - the receiver decision (D7/D8 outcome, filled at C2);
    - the ghost `position_only` re-fit for the M4 caveat.
- [ ] **Step 5: `.gitattributes`.** After the `_ghost_gk_weights/**` and `_ghost_outfield_weights/**` binary lines, add:

```text
# metrics.json in these dirs is not in SHA256SUMS: show it as text in diffs (binary stays for EOL)
silly_kicks/tracking/_ghost_gk_weights/*/metrics.json diff
silly_kicks/tracking/_ghost_outfield_weights/*/metrics.json diff
```

  Verify: `git check-attr diff text -- silly_kicks/tracking/_ghost_gk_weights/position_only/metrics.json` → `diff: set`, `text: unset`.

### Task 14: G2 (C1 state), reused weights, golden re-capture, history constants

**Files:**
- Create `tests/test_bundled_weights_corpus_policy.py`.
- Replace the reused dirs:
  - `_ghost_gk_weights/{default,sweeper,sweeper_position_only}/`;
  - `_ghost_outfield_weights/{default,position_only}/`;
  - `_gk_completion_weights/default/{model.json,SHA256SUMS,metrics.json}`.
- Modify `tests/tracking/data/ghost_velocity_path_baseline.npz`, `tests/tracking/test_ghost_gk_velocity_path_unchanged.py` (docstring), `tests/tracking/test_position_only_bundled.py`.

- [ ] **Step 1: G2 first.** `tests/test_bundled_weights_corpus_policy.py`:

```python
"""G2 (combined-cycle-completion spec section 5): every bundled model variant dir carries the corpus
policy it must ship with. Complete by enumeration (ADR-056): the discovered dir set equals the registry
exactly; `_UNDERIVABLE` is asserted empty. Every expected value is copied from committed or archived
metadata -- a value here that no artifact carries is a defect in this test.
"""

import json
from pathlib import Path

import pytest

_ROOT = Path("silly_kicks/tracking")
_WHEEL_EXCLUDED = {"full"}  # pyproject.toml:222 excludes the maintainer-local `full` dirs from the wheel
_GHOST_PROVIDERS = ["gradientsports", "skillcorner", "sportec"]
_PUBLIC_PROVIDERS = ["idsse", "skillcorner"]


def _x_policy():
    return {("metadata.json", "shipped_variant"): "public", ("metadata.json", "provider_list"): _PUBLIC_PROVIDERS}


def _ghost_policy():
    return {
        ("metadata.json", "corpus_provenance.providers"): _GHOST_PROVIDERS,
        ("metadata.json", "corpus_provenance.n_games"): 179,
    }


def _gof_policy(variant):
    return {
        ("metadata.json", "corpus_provenance.n_games"): 179,
        ("metadata.json", "corpus_provenance.variant"): variant,
    }


POLICY: dict[str, dict[tuple[str, str], object]] = {
    "_xshot_weights/default": _x_policy(),
    "_xshot_weights/position_only": _x_policy(),
    "_xcross_weights/default": _x_policy(),
    "_xcross_weights/position_only": _x_policy(),
    "_ghost_gk_weights/default": _ghost_policy(),
    "_ghost_gk_weights/position_only": _ghost_policy(),
    "_ghost_gk_weights/sweeper": _ghost_policy(),
    "_ghost_gk_weights/sweeper_position_only": _ghost_policy(),
    "_ghost_outfield_weights/default": _gof_policy("default"),
    "_ghost_outfield_weights/position_only": _gof_policy("position_only"),
    "_gk_completion_weights/default": {
        ("metrics.json", "providers"): ["gradientsports"],
        ("metrics.json", "artifact_label"): "full",  # owner decision 2026-08-02 (test_gk_completion_taxonomy.py:48)
    },
    "_gk_completion_weights/skillcorner": {
        ("metrics.json", "variant"): "skillcorner",
        ("metrics.json", "n_matches"): 10,
    },
    "_receiver_weights/default": {
        ("metrics.json", "providers_trained"): ["statsbomb"],
        ("metrics.json", "corpus_visibility"): "public",  # as committed; C2 sets it per D7
    },
}

_UNDERIVABLE: tuple[str, ...] = ()


def _discover() -> set[str]:
    return {
        p.relative_to(_ROOT).as_posix()
        for p in _ROOT.glob("_*_weights/*")
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".") and p.name not in _WHEEL_EXCLUDED
    }


def _get(doc: dict, dotted: str):
    for part in dotted.split("."):
        doc = doc[part]
    return doc


def violations(dirname: str, policy: dict, read=None) -> list[str]:
    """The policy entries the dir's committed metadata does not satisfy (empty == compliant)."""
    read = read or (lambda f: json.loads((_ROOT / dirname / f).read_text(encoding="utf-8")))
    bad = []
    for (fname, key), want in policy.items():
        try:
            got = _get(read(fname), key)
        except (FileNotFoundError, KeyError) as exc:
            bad.append(f"{dirname}/{fname}:{key} missing ({exc!r})")
            continue
        if got != want:
            bad.append(f"{dirname}/{fname}:{key} = {got!r}, policy {want!r}")
    return bad


def test_every_variant_dir_is_registered_exactly():
    found, declared = _discover(), set(POLICY)
    assert found == declared, f"unregistered: {sorted(found - declared)}; missing: {sorted(declared - found)}"


def test_population_agrees_with_the_classification_registry():
    """Single-sourced population: every variant dir sits under a classified weights root."""
    from tests.test_bundled_weights_classification import WEIGHTS_CLASSIFICATION

    roots = {f"silly_kicks/tracking/{d.split('/')[0]}" for d in POLICY}
    assert roots <= set(WEIGHTS_CLASSIFICATION)


def test_underivable_is_empty():
    assert not _UNDERIVABLE


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_dir_matches_its_corpus_policy(dirname):
    assert not violations(dirname, POLICY[dirname])


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_recorded_runs_were_clean(dirname):
    p = _ROOT / dirname / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        if "run_tree_dirty" in m:
            assert m["run_tree_dirty"] is False, f"{dirname} was trained on a dirty tree"


def test_the_checker_catches_a_wrong_variant():
    """Anti-rot: an sc_extended artifact in a public slot must be reported."""
    fake = {"metadata.json": {"shipped_variant": "sc_extended", "provider_list": _PUBLIC_PROVIDERS}}
    assert violations("_xshot_weights/default", _x_policy(), read=fake.__getitem__)
```

  Run `python -m pytest tests/test_bundled_weights_corpus_policy.py -q`. Expect PASS for every dir **except** `_gk_completion_weights/default`: its committed `metrics.json` predates `artifact_label`, and Step 2's reused rebundle carries it. Any other failure means the registry is wrong; fix the registry from the metadata, never the reverse.
- [ ] **Step 1b: Bundled probe ids are committed ids only** (spec §11.8; B r4 CCC-PLAN-40). `probe_sample_matches` and the `probe_sample_in_training_folds` keys land raw in each xshot/xcross bundle's `metrics.json`, which ships in the wheel and in Hub publishes. Today only operator input (`gs_probe.json`) keeps an owner-tier id out. Create `tests/test_bundled_probe_ids_are_committed_ids.py`:

```python
"""A bundle's probe ids are only already-committed ids (combined-cycle spec 11.8; B r4 CCC-PLAN-40).

`probe_sample_matches` and the `probe_sample_in_training_folds` keys are written raw into the bundle's
metrics.json, which ships in the wheel and is copied into Hub publishes. Only the public arm and the two
GS probe ids already committed in the xcross `default` record (10502/10503) may appear."""

import json
from pathlib import Path

import pytest

from scripts._corpus import bundled_public_arm_pairs

_ALLOWED = {(str(p), str(m)) for p, m in bundled_public_arm_pairs()} | {
    ("gradientsports", "10502"),
    ("gradientsports", "10503"),
}
_DIRS = sorted(
    d
    for w in ("_xshot_weights", "_xcross_weights")
    for d in (Path("silly_kicks/tracking") / w).iterdir()
    if (d / "metrics.json").is_file()
)


def test_the_scan_covers_all_four_bundles():
    assert len(_DIRS) == 4  # never a silent pass over an empty glob


@pytest.mark.parametrize("d", _DIRS, ids=lambda d: f"{d.parent.name}/{d.name}")
def test_bundle_probe_ids_are_committed_ids(d):
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    pairs = {(str(p), str(mid)) for p, mid in (m.get("probe_sample_matches") or [])}
    assert pairs <= _ALLOWED, f"{len(pairs - _ALLOWED)} probe match(es) outside the committed set"
    assert {str(k) for k in (m.get("probe_sample_in_training_folds") or {})} <= {mid for _p, mid in _ALLOWED}
```

  It passes on the current bundles (measured 2026-10-02: xcross `default` probes GS 10502/10503, xcross `position_only` probes SkillCorner 1886347/1899585, both xshot bundles carry no probe keys). Check that it bites: plant `["gradientsports", "99999"]` in a scratch copy of the xcross `default` `metrics.json`, point a one-off parametrization at it, confirm FAIL, remove it. The C2 re-fits must pass it unchanged.
- [ ] **Step 2: Copy the reused dirs from the archive** (byte-for-byte). Before copying, list each archived dir and compare it with the committed file set. A committed file absent from the archive (e.g. `MODEL_CARD.md`) is KEPT. Compare `_gk_completion_weights/default/MODEL_CARD.md` with the archived one, ignoring CR: `python -c "import sys; a,b=(open(p,'rb').read().replace(b'\r',b'') for p in sys.argv[1:]); print(a==b)" <committed> <archived>` must print `True`.

```bash
S=<scratchpad>/reuse && mkdir -p $S && tar xf "$ARCHIVE/f1b_artifacts_3ca609f.tar" -C $S \
  rt_ggk_default/ghost_gk_v1 rt_ggk_sweeper/ghost_gk_v1 rt_ggk_sweeper_po/ghost_gk_v1 \
  rt_gof/default rt_gof/position_only das-parity/sk/silly_kicks/tracking/_gk_completion_weights/default
W=silly_kicks/tracking
for pair in default:default sweeper:sweeper sweeper_po:sweeper_position_only; do
  src=${pair%%:*}; dst=${pair##*:}
  cp $S/rt_ggk_$src/ghost_gk_v1/{rfcde_weights.npz,metadata.json,metrics.json,SHA256SUMS} $W/_ghost_gk_weights/$dst/
done
for v in default position_only; do cp $S/rt_gof/$v/{model.npz,metadata.json,metrics.json,SHA256SUMS} $W/_ghost_outfield_weights/$v/; done
cp $S/das-parity/sk/silly_kicks/tracking/_gk_completion_weights/default/{model.json,SHA256SUMS,metrics.json} $W/_gk_completion_weights/default/
git status --short silly_kicks/tracking
```

  These dirs GAIN a `metrics.json`, because the previous bundles had none: ghost `default`/`sweeper`/`sweeper_position_only` and both outfield dirs. That adds provenance (`run_commit`, `run_tree_dirty`) and is stated in the C1 message. `SHA256SUMS` must list exactly the weight files `load()` verifies. Ghost `position_only` is NOT copied: it is re-fit at C1 (spec §0.11).
- [ ] **Step 3: Verify integrity + loads:**

```bash
python -m pytest tests/test_bundled_weights_corpus_policy.py tests/test_bundled_weights_classification.py \
  tests/test_bundled_models_load_on_float32_commit1.py tests/tracking/test_ghost_gk*.py tests/tracking/test_ghost_outfield*.py \
  tests/tracking/test_gk_completion*.py tests/tracking/test_position_only_bundled.py -q
```

  Expected failures, and only these:
  - `test_ghost_gk_velocity_path_unchanged` (Step 4);
  - `test_position_only_bundled.py::test_bundled_ghost_default_is_the_both_axes_refit` (Step 5).

  Any other failure: stop and diagnose.
- [ ] **Step 4: Re-capture the velocity golden** (its own convention).
  - In a scratch script, take the old positions from the committed `ghost_velocity_path_baseline.npz` (`positions`) and the new ones from `_serve()` on the current tree.
  - Report max |dx|, max |dy|, the mean and median Euclidean displacement, the row count, and all-finite.
  - Overwrite the npz (`np.savez(path, positions=new)`).
  - Append this docstring paragraph, with the measured numbers substituted:

```text
**RE-CAPTURED AGAIN at the F1b float32-frame re-fit (ADR-106) -- revisited, not absorbed.** The bundled
ghost-GK `default` was re-fit on float32-stored frames (training_commit=3ca609f), the DECLARED re-fit this
baseline is expected to move on. Measured effect on this fixture (``sb360-fixture-2``), the prior
both-axes weights (4bda048) versus the SHIPPED float32-frame weights, <N> rows, all finite:
**max |dx| <a> m, max |dy| <b> m, mean <c> m, median <d> m**.
```

  Run the test: PASS.
- [ ] **Step 5: History constants.** In `tests/tracking/test_position_only_bundled.py`, add `_C4 = "3ca609f8ae4003f411f9939dfde38fb320ff00fc"  # F1b float32-frame ghost default re-fit (ADR-106)`. `test_bundled_ghost_default_is_the_both_axes_refit` asserts `training_commit == _C4`; rename it to `test_bundled_ghost_default_is_the_f1b_refit` and update its comment. `_C1`/`_C2`/`_C3` stay as history. Ghost `position_only` stays at `_C3` until C2.

### Task 15: C1 verification, reviews, owner gates

- [ ] **Step 1:** `python -m ruff format silly_kicks/ tests/ scripts/`, then `--check`, `python -m ruff check silly_kicks/ tests/ scripts/`, and `pyright` — all clean.
- [ ] **Step 2:** The CI-faithful full suite on the pandas-2 venv and on the pandas-3 venv (`-m "not e2e"`, then `-m slow`). Record collected / passed / skipped / deselected for each; compare with Task 0; zero failures. Regenerate `.test_durations` only if `test_ci_shard_wiring` demands it.
- [ ] **Step 3:** The local smoke of the two new committed drivers on this machine (clean tree not required here: `--allow-dirty`, outputs to the scratchpad, never committed):
  - `PYTHONPATH=. python scripts/validate_hub_variants.py --out <scratch>/hub --allow-dirty` — this exercises the real Hub population check, the serve lambdas and the README comparison. Expected `cards_mismatched`: exactly the two sweeper repos (measured 2026-10-02). Expected `mirrors_mismatched`: all four mirrors -- after Task 14 the wheel's mirror dirs hold the reused `3ca609f` weights, while the Hub serves `adafb72` (sweepers) and `b68328a` (outfield) (B r5 CCC-PLAN-43). Expected `load_refused`: exactly the two sweeper mirrors (chirality refusal of the `adafb72` weights, measured 2026-10-02). Anything else is reported;
  - `PYTHONPATH=. python scripts/publish_model_card.py --repo-id <repo> --verify-only` for every `CARD_SOURCE` repo — anonymous reads only, no upload. Expected `changed: true` only for the two sweeper repos at this point;
  - `python scripts/validate_receiver_widening.py --rows <archived candidate_rows.parquet> --out <scratch>/rg --allow-dirty` (owner token in the local env) — this exercises the real identification.

  A failure is fixed before C1. These outputs are discarded; the provenance runs happen at C1.
- [ ] **Step 4:** `/final-review` (Phase-2.5 decision inventory; C4 regenerated with Graphviz `dot` only if the DSL changes).
- [ ] **Step 5:** External implementation review (owner-coordinated), report to `$REVIEWS`. Freeze the tree until it returns; apply the findings; re-run Steps 1–2.
- [ ] **Step 6: OWNER GATE — commit C1.** First set the spec's `**Status:**` line to its final state, "APPROVED — spec rev <n> / plan rev <m>, review rounds in Appendix A; executed from C1" (B r4 N3: the committed doc must not carry a stale review status). Then show `git status --short`, `git diff --stat`, the file list, and the draft message:

```text
feat(cycle): completion-cycle code + reused F1b weights (C1, no version)

Reused F1b float32-frame re-fits (training_commit 3ca609f): ghost_gk default/sweeper/sweeper_position_only,
ghost_outfield x2, gk_completion default (rebundle). Velocity-path golden re-captured (measured move in its
docstring). G1/G2 corpus guard; trainers emit the ADR-067 reproducibility caveat; receiver visibility from the
manifest (D7); receiver-widening (D8 combined rule) and Hub-smoke drivers (README-vs-card check); card-only
Hub push seam publish_model_card.py + LF card staging (ADR-088 amendment, D9); xcross --probe-match-ids-json
held-out probe (D5c); launcher {worker} token + list-matches done markers + trainer --prep-only/--list-studies/
--study-list (D3); T10 sharding (commit-keyed); TF-19 --reduce-only; DAS parity schema -4 (full numba
coverage, D-KEY/finite/direction counts, completeness + commit checks) and --benchmark; per-engine
estimate_das_cost constant (D6); download race fix; test_das_benchmark.py; ADR-088/106/107/108 + docs/context
notes. Spec + plan.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

  Wait for an explicit yes. Record the C1 SHA.
- [ ] **Step 7: OWNER GATE — push the branch** (`git push -u origin feat/combined-cycle-completion`; the DGX clones it). This is a separate explicit yes.

---

## PHASE B — DGX, one wave at C1

**Phase B amendment (owner-approved 2026-10-02: option A, amend C1 and force-push the draft PR).** The Task 17 smokes at the first C1 found five script defects (F1–F3, then F5–F6 on real data), all present before this cycle and all in drivers the wave depends on, plus a benchmark sizing limit handled in the harness (F7). They are fixed test-first in C1 itself, so `run_commit == C1` still names the one commit every wave run uses. No wave run had started.
- **F1 — IDSSE string ids.** The DAS reference leg cast every id with `int()`, so every IDSSE match (`DFL-OBJ-*`, `DFL-CLU-*`) failed the map. Integral ids keep the golden `str(int(v))` token; string ids pass through; both legs key players by one rule (`_id_key`: an int for an all-digit id, else the string).
- **F2 — benchmark path legs.** `_bench_match` passed raw frames, without `team_in_possession`, to the old/new path legs, and `add_das` refused them on every match. Possession is now derived first (untimed), so both legs time the same input.
- **F3 — launcher memory cap.** The cgroup backend ran `systemd-run --scope` for a non-root user, which the system manager refuses ("Interactive authentication required"), so every worker failed and was relaunched until the launcher gave up. A non-root user now goes through `--user`, and the launcher refuses once, before any worker, when a capped no-op cannot start.
- **F5 — reference player grain misread** (owner-approved 2026-10-03, same amend). accessible-space returns team results for the rows with possession only, but player results for EVERY row; the leg read both with the possession-row counter, so every row without possession shifted later player values onto other players. Player values are now read by input row, and a result off that measured shape raises.
- **F6 — carrier not forwarded** (owner decision 2026-10-03: option (i)). Native DAS excludes the ball carrier from offside; the leg now forwards `ball_carrier_player_id` as `player_in_possession_col` whenever the frames carry it, in both reference legs. With F5 and F6, one SkillCorner and one IDSSE match reproduce the library exactly (max 0).
- **F7 — benchmark memory ceiling** (owner decision 2026-10-03: option (a)). The OLD path's `das_xfns` (4.127.0 + accessible-space) peaked at 113.4 GiB on an IDSSE match and was OOM-killed inside a 110G cap on a GS match, where the new path needed 2.6 GB. `_das_path_timing.py` now runs a self-watchdog (default ceiling 100 GiB, `--path-memory-limit-gib`): above it, the run writes what finished plus `over_memory` (phase, limit, peak) and exits cleanly. `_bench_match` records it per match; `summarize_benchmark` computes each speedup over the matches where both paths finished that call (`speedup_n`) and counts `old_path_over_memory` / `new_path_over_memory` per provider. Every other leg still runs on every sample match.
- **Re-review round 3** (A: APPROVE; B: APPROVE with m-1..m-4). m-1: a test proves `_main` arms the ceiling. m-2: off Linux the reading is `None` and the record says `memory_measured: false`, never a 0.0 peak; the page size comes from `mmap.PAGESIZE`, so the module type-checks on every host (round 4). m-3: a missing `systemd-run` or a hung probe (30 s timeout) is the same clean `REFUSED`. m-4 (owner decision 2026-10-03): a `das-reference-contract` CI job runs the REAL accessible-space 2.0.15 against native on synthetic scenes (`tests/scripts/test_das_reference_contract.py`: the F5 result shapes, a dead-ball frame first, a carrier beyond the line with a negative control, string ids), with `SK_REQUIRE_DAS_REFERENCE=1` so it cannot pass by skipping; `tests/test_ci_das_reference_contract_wired.py` pins it. The F7 ceiling stays 100 GiB (owner decision 2026-10-03): IDSSE's old `das_xfns` (113.4 GiB) is recorded as not fitting, like GS.
- **F4 (plan only)** and the plan gaps: Task 18 Step 2b parses the LAST `--prep-only` stdout line; Task 16 Step 3's reuse check imports pyarrow; Task 17's gkc smoke uses two matches; the Step 4b smoke reduce passes its own match list; Task 18's xshot/xcross caps are sized in Step 5.

**Phase B amendment 2 (owner decisions 2026-10-04: 1 = A, amend C1 again, force-push, and re-run the whole wave; 2 = (i), fix the double count in the amend; 3 = (b), add a gate-passing D3b pair).** The first wave, at the first amended C1 (`43ff0dd`), failed two acceptance checks, and a third defect mis-stated two corpus manifests. All three are in driver code and are fixed test-first in C1. Every wave run repeats at the new C1; the `43ff0dd` outputs are scratch evidence only.
- **F8 — reference rows out of frame order.** DAS: 216 of 895 SkillCorner matches carried frames outside the golden bound (1197 team and 25715 player rows, max 28.4); GS and IDSSE were exact. accessible-space 2.0.15 takes each frame's carrier in the caller's ROW order (`drop_duplicates(frame_col)`, `interface.py:908`) but builds the positions from a copy sorted by `frame_col` (`transform_into_arrays`), and the corpus scored rows are not frame-sorted (match 2003157: 53 descents), so frames were paired with other frames' carriers. The leg now sorts its rows by the dense frame code before the call (`_in_frame_order`). Measured on 2003157: 18 frames outside the bound → 0, max 0.0. The contract job gains an unsorted two-frame carrier scene with a negative control (the library fed the same rows unsorted diverges).
- **F9 — T10 never measured ghost_gk.** Since `9449461` the T10 ghost_gk adapter unpacked the default `(features, labels)` 2-tuple as `(feats, meta)`, so every match raised `KeyError`, which the per-model status row recorded (`error:KeyError` × 179) while `n_failed` stayed 0. The adapter now passes `return_meta=True` (one match: 26 features measured). Tests run the real extractor; the end-to-end test asserts ghost_gk is measured. Owner-approved 2026-10-04: a model that errors on EVERY match now refuses the artifact (`_refuse_unmeasured_models`, serial and reduce paths alike); a per-match error stays a recorded status. The T10 test fixture gains two full SPADL keeper passes so every model is measured or legitimately empty on it.
- **F10 — overlapping partitions counted twice.** The GKDV and spells corpus manifests reported `n_matches: 128` and doubled frame counts for the 64-match corpus: the Task 19 Step 2 full-population pass replayed every resumed match's counters into a third manifest, which `aggregate_manifests` summed with the two wave workers'. Every worker manifest of a driver that sums manifests now records `partition_keys` (seven drivers, pinned by a derived gate), `aggregate_manifests` refuses overlapping coverage, and GKDV and spells gain `--reduce-only`: population-complete, counts from the reduce pass's own replayed counters (each match once), no manifest written, a worker at another commit or generation or a shard set no worker vouches for refused. Task 19 Step 2 uses it. The tables and the signoff were correct; only the manifests over-counted. A failed key is not listed in `partition_keys` (it carries no counters), so a launcher relaunch that completes it under another worker tag is not an overlap (final-review).
- **F11 (final-review) — DAS exclusions lost on a relaunch.** `_parallel_launch` relaunches a killed worker with only its remaining items under the same tag, and the killed attempt never wrote its manifest, so a match it had excluded vanished from the summed `n_excluded` (top level and `population.excluded`). The DAS reduce now counts the exclusion markers over the population (a shard wins over a stale marker). The first wave had no relaunch (`relaunched=0`), so its 14 exclusions were right.
- **D3b (b).** On `small_paired.json` both D3b runs refuse at the xcross Brier gate, so neither writes `model.json` and the plan's `model.json` comparison cannot run. Task 18 Step 2b adds a serial vs fan-out pair on `original17.json` (the public corpus whose xcross runs pass the gates) for that comparison; the small paired set keeps its other checks.
- **Task 19 Step 3 (measured).** The `6b242cf..C1` diff of `silly_kicks/gkdv/_validate.py` is comments plus the additive `_ARM_DIRECTION_KEY` / `expected_direction_for_arm` (`07a88f6`, PR-S175), not comments only. The sign-off reads only `ICC_ANCHORS` and `ATT_RELATIVE_ANCHORS`, both unchanged; Step 3 records that. Likewise Task 20's TF-19 check reads `population_size` (the artifact has no `n_matches` key).

**Phase B amendment 3 (owner decisions 2026-10-04: merge before the wave; A/B re-reviews before the merge; TODO "Last updated" replaced in this merge).** The commit structure changes; every other rule stands.
- **PR-1 (draft PR #267).** C1 carries the cycle code, the reused F1b weights, amendments 1–3, and the TODO "Last updated" block replaced with a brief summary of what this merge provides. C1 is rebased onto `origin/main` (main's docs merges `f07ce5c` and `5d043f8` touch C1's files only in `AGENTS.md`, a different bullet, and `TODO.md`). Before the merge: CI-faithful verification on the exact rebased tree, `/final-review` (versioning skipped), A and B re-reviews, then force-push, CI green, ready-for-review, and a **squash** merge, each behind its own owner gate. No version bump and no CHANGELOG entry: the release entry stays in PR-2.
- **M** is the squash-merge commit on `main`. Every wave run executes at M. From this amendment on, every Phase B and C reference to C1 as the run commit means M, and `$C1` is set to M. M is on `main`, so the provenance needs no non-squash merge.
- **Fixes found during the wave** land as new commits on `main` through their own PR, each owner-gated, never as an amend. A fix that touches code a run used goes to the owner, who decides which runs repeat at the new `main` commit.
- **PR-2** carries C2 (re-fit weights, artifacts, the gates that read them, cards, version, CHANGELOG, TODO) on a new branch off `main`, the cycle's second branch (its reason: merge-before-wave). Its PR, merge (squash allowed: one commit, and M is already on `main`), tag, PyPI (the owner) and Hub pushes keep their gates (Task 22).
- Task 16 Step 1 checks out M; Task 22 Step 8's ancestry check becomes `git merge-base --is-ancestor $M origin/main` and `3ca609f`.

After the amend: re-run Task 16 Steps 1–2 at the new C1, then the Task 17 smokes the fixes touch before Step 5:
- the DAS map on one SkillCorner and one IDSSE match, whose reduce must show `n_outside_golden_bound` all zero and `finite_mask_mismatches` `{team: 0, player: 0}` for both providers (the C2 gates, checked before the wave);
- the benchmark over one GS match at the default ceiling: the new path finishes, and the old path either finishes or is recorded as `over_memory` with its phase (F7; the benchmark still runs alone);
- the DAS launcher on the default memory backend.

After amendment 2, additionally (before Step 5):
- the DAS map on SkillCorner 2003157 (18 frames outside the bound at `43ff0dd`): every `n_outside_golden_bound` cell 0;
- T10 on one match (`t10_one.json`): ghost_gk `match_status_counts` holds `ok` rows and no `error:` status;
- GKDV and spells on two one-match slices of the smoke corpus, then `--reduce-only` over both: `n_matches` 2 and no third manifest.

All commands run on `ssh $DGX`. Set `C1=<sha>`, `B=~/cc/bin`, `R=~/cc/venv-run/bin/python`.
- Drivers run from their checkout root with `PYTHONPATH=.`.
- Thread pinning: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1` (the benchmark overrides `NUMBA_NUM_THREADS=20`).

### Task 16: Prerequisites (verify; rebuild what is missing)

- [ ] **Step 1: Checkouts at C1** (fresh; never an old checkout; after amendment 3, `C1` here is M, the PR-1 squash-merge commit on `main`):

```bash
mkdir -p ~/cc && cd ~/cc
for d in run gkc; do git clone -q https://github.com/karsten-s-nielsen/silly-kicks.git $d && git -C $d checkout -q $C1; done
for d in run gkc; do git -C ~/cc/$d rev-parse HEAD; git -C ~/cc/$d status --porcelain | wc -l; done   # C1 twice, 0 twice
```

- [ ] **Step 2: venvs.** These are editable installs; `*.egg-info/` and `__pycache__/` are gitignored. Verify the tree stays clean:

```bash
for d in run gkc; do python3 -m venv ~/cc/venv-$d && ~/cc/venv-$d/bin/pip install -q -e "$HOME/cc/$d[test,numba,xgboost,calibration,kloppy,ghost-gk,xshot,xcross]" pyarrow; done
python3 -m venv ~/cc/py2new && ~/cc/py2new/bin/pip install -q -e "$HOME/cc/run[numba]" "pandas<3" pyarrow     # NEW path, pandas 2 (D1)
python3 -m venv ~/cc/py2old && ~/cc/py2old/bin/pip install -q "silly-kicks[numba]==4.127.0" "accessible-space==2.0.15" "pandas<3" pyarrow
~/cc/py2old/bin/python -c "import importlib.metadata as m, importlib.util as u; print(m.version('silly-kicks'), m.version('accessible-space'), u.find_spec('silly_kicks.tracking._das_engine') is None)"   # 4.127.0 2.0.15 True
~/cc/py2new/bin/python -c "import pandas, importlib.util as u; print(pandas.__version__ < '3', u.find_spec('silly_kicks.tracking._das_engine') is not None)"   # True True
git -C ~/cc/run status --porcelain | wc -l   # 0
```

- [ ] **Step 3: pandas-2 reference env.**
  - Reuse `~/das-parity/py2ref` if `bin/python -c "import pandas, pyarrow, importlib.metadata as m; assert pandas.__version__ < '3' and m.version('accessible-space') == '2.0.15'"` passes (pyarrow: the reference leg reads and writes parquet; Phase B amendment); else `python3 -m venv ~/cc/py2ref && ~/cc/py2ref/bin/pip install "accessible-space==2.0.15" "pandas<3" pyarrow`.
  - Run the inference-signature check from Task 10 Step 3 here.
  - Export `SK_DAS_REFERENCE_PYTHON`, `SK_DAS_OLDPATH_PYTHON=$HOME/cc/py2old/bin/python`, `SK_DAS_NEWPATH_PYTHON=$HOME/cc/py2new/bin/python`.
- [ ] **Step 4: Wrappers (fail-closed) + caps + linger:**

```bash
mkdir -p ~/cc/bin ~/cc/logs ~/cc/out ~/cc/inputs
cat > ~/cc/bin/owner <<'EOF'
#!/usr/bin/env bash
# owner CMD... -- the OWNER pining token for this one process only; refuses an empty/unreadable token
f="$HOME/.pining_owner.env"
tok="$(sed -n 's/^export PINING_FOR_THE_DATA_TOKEN=//p' "$f" 2>/dev/null | tr -d '"'"'"'')"
if [ -z "$tok" ] || [ "$tok" = "test-token-pining-for-the-data" ]; then
  echo "owner: no owner token in $f -- refusing (the loader would silently fall back to the PUBLIC token)" >&2
  exit 3
fi
export PINING_FOR_THE_DATA_TOKEN="$tok"
exec "$@"
EOF
cat > ~/cc/bin/public <<'EOF'
#!/usr/bin/env bash
# public CMD... -- no token: the loader falls back to the documented PUBLIC token
unset PINING_FOR_THE_DATA_TOKEN
exec "$@"
EOF
cat > ~/cc/bin/cap <<'EOF'
#!/usr/bin/env bash
# cap MEM CMD... -- an OOM kills only this worker (its shards persist; a relaunch resumes)
mem="$1"; shift
exec systemd-run --user --scope -q -p MemoryMax="$mem" -- "$@"
EOF
chmod +x ~/cc/bin/*
systemd-run --user --scope -q -p MemoryMax=1G -- true && echo CAP-OK
loginctl show-user "$USER" -p Linger    # must be Linger=yes for scopes to outlive the ssh session; if no: ask the owner to run `loginctl enable-linger`
```

- [ ] **Step 5: Owner visibility check** (counts only). Expect GS 64, skillcorner 909, idsse 7, statsbomb 327; anything else STOPS the wave:

```bash
cd ~/cc/run && PYTHONPATH=. $B/owner $R -c "
import sys; sys.path.insert(0,'scripts')
from _loader_pining import _list_matches, _resolve_token, _base_url
t,b=_resolve_token(None),_base_url()
print({p: len(_list_matches(p,t,b)) for p in ('gradientsports','skillcorner','idsse','statsbomb')})"
```

- [ ] **Step 6: Corpora.**
  - `find ~/tc3-cache/shards -name '*.parquet' | wc -l` → 179, and the same for `~/tc3-cache-f32`.
  - If either is missing or partial: restore it from `$ARCHIVE` (`tc3_cache_f64/tc3-cache`, `tc3_cache_f32/tc3-cache-f32`) with `tar cf - | ssh … tar xf -`, then verify against `SHA256SUMS.dgx` (normalise the `*` marker).
- [ ] **Step 7: Private inputs** into `~/cc/inputs/` (scp from `$ARCHIVE/corpus_lists/`, never into a checkout):
  - `corpus179.json`;
  - `gs64.json` = `{"gradientsports": corpus179["gradientsports"]}`;
  - `original17.json` (both keys), written by `PYTHONPATH=. $R -c "from scripts._corpus import BUNDLED_PUBLIC_ARM as A; import json; print(json.dumps({k: list(v) for k, v in A.items()}))"`;
  - `one_public.json` = `{"idsse": ["DFL-MAT-J03WMX"]}`;
  - `gs_probe.json` = `{"gradientsports": ["10502", "10503"]}` (D5c; the two ids already committed in the xcross record, spec §11.8).

### Task 17: Smokes (one per workload) and sizing

Every smoke uses `--out ~/cc/smoke/<name>` and records peak RSS via `/usr/bin/time -v`. Start with `cd ~/cc/run && export PYTHONPATH=.`.

- [ ] **Step 1: Corpus pin check** (public token). If it fails, STOP: `--max-per-provider 10` no longer pins the gkc arm, and it becomes an owner decision.

```bash
$B/public $R -c "import sys; sys.path.insert(0,'scripts'); from _loader_pining import select_match_ids; from _corpus import bundled_public_arm_pairs as b; got=sorted([p,m] for p,m in select_match_ids(providers=['skillcorner'], max_per_provider=10)); assert got==b(('skillcorner',)), got; print('gkc 10 == original 10')"
```

- [ ] **Step 2: Trainer smokes.**
  - **xshot:** `$B/public /usr/bin/time -v $R scripts/train_xshot_occurrence.py --providers idsse --match-ids-json ~/cc/inputs/one_public.json --n-trials 2 --output-dir ~/cc/smoke/xshot --expect-variant public`. It must pass the preflight and extract one match; the gates may fail on one match.
  - **xcross:** the D5c shape on one training match. `$B/owner /usr/bin/time -v $R scripts/train_xcross_attempt.py --providers idsse --match-ids-json ~/cc/inputs/one_public.json --n-trials 2 --output-dir ~/cc/smoke/xcross --expect-variant public --probe-providers gradientsports --probe-comparison-providers "" --probe-match-ids-json ~/cc/inputs/gs_probe.json`.
    - The preflight passes under the owner token: G1 checks the training allowlist only.
    - `probe_shards/` holds the two GS keys; the training shard root holds only the idsse key.
  - **G1 negative:** `$B/owner $R scripts/train_xshot_occurrence.py --providers skillcorner --output-dir ~/cc/smoke/neg --expect-variant public` must exit with the "non-public" refusal, and no shard is written.
  - **gkc:** run in a throw-away clone at C1 (`~/cc/smoke-ck`), because the run writes into its tree: `cd ~/cc/smoke-ck && PYTHONPATH=. $B/public ~/cc/venv-gkc/bin/python scripts/train_gk_completion.py --variant skillcorner --providers skillcorner --max-per-provider 2 --mode rebundle --reason smoke --shard-dir ~/cc/smoke/gkc_shards`. G1 passes, and `requested_match_ids` is printed in the metrics. Two matches, not one: GroupKFold needs two groups. The rebundle's drift refusal on this subset is expected (Phase B amendment). Then `rm -rf ~/cc/smoke-ck`.
  - **ghost `position_only`:** `--subsample-cap` small, `--data-dir ~/tc3-cache-f32/shards/899a46878e7d723f` — checks the flags and RSS.
  - **receiver:** `$B/owner $R scripts/train_receiver_model.py --out ~/cc/smoke/rcv --shard-root ~/cc/smoke/rcv_shards --feature-set public --provider statsbomb --match-ids-json <one statsbomb id json> --min-passes 1`. `metrics.json` must show `corpus_visibility: restricted`.
- [ ] **Step 3: Driver smokes** (owner token where the driver lists the corpus; one id each; `--shards-only --worker-tag smoke` where the driver has it):
  - **DAS map — two matches.** One must be a multi-period match whose `frame_id` restarts per period (pandas-2 spec §8).
    - Find it FROM THE FRAMES, independently of the driver: load candidate matches via `pining_source(...)` (owner token) one at a time and test whether any `frame_id` appears in more than one `period_id` of the same game. Never select by the driver's own `n_dkey_frames` (B r2 CCC-PLAN-22: that would be circular).
    - Then check that its smoke shard's `n_dkey_frames` equals `_n_dkey_frames` computed on the same frames, independently.
    - If no listed match has a per-period `frame_id` restart, STOP and report: the D-KEY gate (`test_the_d_key_path_was_exercised`) would then have nothing to bite on, and that is an owner decision.
    - Check that `n_dir_compared > 0` and that the inferred-direction leg ran without exception.
  - T10 map, GKDV, spells, TF-19.
- [ ] **Step 4: Benchmark smoke.** Run `--benchmark --repeat 1 --benchmark-sample-json <one-match sample>` into `~/cc/smoke/bench`. It must produce `performance.json` with:
  - both path timings (old `native=false`, new `native=true`);
  - the reference `compute_s`;
  - the thread sweep.
- [ ] **Step 4b: Launcher smokes (D3).**
  - **DAS:** `_parallel_launch.py --mode das --das-generation "$($R scripts/validate_das_native_parity.py --print-generation)"` over a 2-match `--corpus-json` (list-matches shape) with `--worker-tag {worker}`, into a fresh smoke shard root (never the wave's). The smoke's `--reduce` passes `--match-ids-json <the same 2-match file>`: without it the reduce's population is the full listing, and it refuses the unaccounted matches. Expect both shards inside that generation directory, `manifest_w0.json`/`manifest_w1.json` (never a path in a filename), and the `--reduce` producing `metrics.json` in a smoke `--out`. Re-run the same launcher command once: it must launch nothing (both items done in the expected generation). Run it on the default (detected) memory backend: the smoke is what proves the cgroup cap starts (Phase B amendment F3).
  - **Study fan-out:** write `small_paired.json` (Task 18 Step 2b), run its `--prep-only` into `~/cc/smoke/fan`, then one `--study <tag>` worker under `/usr/bin/time -v`. Its peak RSS sets the launcher's `--peak-rss-gib` for the Task 18 Step 2b check, and is the D3b measurement.
- [ ] **Step 5: Size the wave.**
  - Budget 99 GiB and 20 cores. Write `~/cc/plan.txt` with per-workload worker counts so that Σ(workers × peak RSS × 1.3) ≤ 99 GiB and Σ workers ≤ 20, prioritising GKDV, TF-19, T10 and DAS.
  - The DAS map runs the reference library twice per match (shared direction + inferred direction, spec §10). Use the measured two-leg smoke time per match, not the first run's ~14 CPU-h estimate (≈ 28 CPU-h expected).
  - Reserve 1 core for the D3a serial DAS check and 1–2 cores for the D3b paired study check.
  - `<xshot cap>` / `<xcross cap>` (Task 18 Step 2): the original 12G cannot hold a 17-match run. The Task 17 smokes peaked at 7.6 GB (xshot) and 12.0 GB (xcross) on ONE match, and the 8-match fan-out prep at 11.9 GB. The archived F1b logs record no peak RSS, so there is no full-run figure to size from. Cap each at 32G (more than 2.5× the one-match smoke peak, within the 99 GiB budget), run each under `/usr/bin/time -v`, and record the measured peak in `plan.txt` and `findings.md` (Phase B amendment).
  - The benchmark (Task 19 Step 5) is long: one old-path `das_xfns` call measured 355 s (SkillCorner) and 1070 s (IDSSE) per match, and every leg runs warm-up + `--repeat`. Schedule it last, alone, with hours of budget.
  - DAS: write `das_all.json` from `$B/owner $R scripts/validate_das_native_parity.py --list-matches`; assert per-provider counts 909 / 64 / 7. The launcher splits it (no hand split).
  - `bench_sample.json` = `random.Random(0).sample(sorted ids, 3)` per velocity-bearing provider, drawn from `das_all.json`, in list-matches shape.
  - Split each remaining list round-robin into N JSON files:
    - T10 keys: `measure_f1b_feature_delta.py --data-dir ~/tc3-cache --list-match-keys`.
    - TF-19: `corpus179.json`.
    - GKDV and spells: `gs64.json`.

### Task 18: The wave

- [ ] **Step 1: Launch the longest workloads first** (all `nohup`, logs in `~/cc/logs/`, each worker under `$B/cap <MEM>`, every `--out` under `~/cc/out/`). A small queue script, `~/cc/queue.sh` (not committed), keeps the planned concurrency.

The DAS map is driven by the launcher (D3, launcher plan Task 8). The launcher applies its own per-worker cap and its live-RAM backpressure; `--nproc` holds its share of the 20 cores:

```bash
cd ~/cc/run && export PYTHONPATH=. SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache
test ! -e ~/cc/out/das/shards || { echo "das shard root exists -- use a fresh one"; exit 1; }   # B r3 CCC-PLAN-28
GEN=$($R scripts/validate_das_native_parity.py --print-generation) && echo "das generation $GEN"
nohup $B/owner $R scripts/_parallel_launch.py --mode das --das-generation "$GEN" --corpus-json ~/cc/inputs/das_all.json \
  --shard-root ~/cc/out/das/shards --peak-rss-gib <das peak> --headroom-gib 20 --nproc <das workers> \
  --driver "$R scripts/validate_das_native_parity.py --out $HOME/cc/out/das --shard-root $HOME/cc/out/das/shards --match-ids-json {subset} --shards-only --worker-tag {worker}" \
  --reduce "$R scripts/validate_das_native_parity.py --out $HOME/cc/out/das --shard-root {shard_root} --reduce-only" \
  > ~/cc/logs/das_launcher.log 2>&1 &
```

The owner token reaches every worker: the launcher runs inside `$B/owner`, and `thread_pin_env()` copies `os.environ` (`scripts/_thread_pin.py:18`) into each worker's environment.

The other workers:

```bash
nohup $B/cap <c> $R scripts/measure_f1b_feature_delta.py --data-dir ~/tc3-cache --out ~/cc/out/t10 \
  --shards-only --worker-tag w$i --match-keys-json ~/cc/inputs/t10_$i.json > ~/cc/logs/t10_w$i.log 2>&1 &
nohup $B/cap <c> $B/owner $R scripts/build_gkdv_arm_values.py --out ~/cc/out/gkdv --arm das --providers gradientsports \
  --match-ids-json ~/cc/inputs/gkdv_$i.json > ~/cc/logs/gkdv_w$i.log 2>&1 &
nohup $B/cap <c> $B/owner $R scripts/build_layer2_spells.py --out ~/cc/out/spells --providers gradientsports \
  --match-ids-json ~/cc/inputs/spells_$i.json > ~/cc/logs/spells_w$i.log 2>&1 &
nohup $B/cap <c> $B/owner $R scripts/build_tf19_instrument_responsiveness.py --out ~/cc/out/tf19 \
  --providers gradientsports,skillcorner,idsse --match-ids-json ~/cc/inputs/tf19_$i.json > ~/cc/logs/tf19_w$i.log 2>&1 &
```

- [ ] **Step 2: Re-fits.**

```bash
cd ~/cc/run && export PYTHONPATH=.
for fs in faithful position_only; do
  nohup $B/cap <xshot cap> /usr/bin/time -v $B/public env SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache \
    $R scripts/train_xshot_occurrence.py --providers idsse,skillcorner --match-ids-json ~/cc/inputs/original17.json \
    --feature-set $fs --expect-variant public --output-dir ~/cc/out/xshot_$fs > ~/cc/logs/xshot_$fs.log 2>&1 &
  # xcross: OWNER token for the held-out GS probe only (D5c); training stays public via the allowlist + G1
  nohup $B/cap <xcross cap> /usr/bin/time -v $B/owner env SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache \
    $R scripts/train_xcross_attempt.py --providers idsse,skillcorner --match-ids-json ~/cc/inputs/original17.json \
    --feature-set $fs --expect-variant public --probe-providers gradientsports --probe-comparison-providers "" \
    --probe-match-ids-json ~/cc/inputs/gs_probe.json --output-dir ~/cc/out/xcross_$fs > ~/cc/logs/xcross_$fs.log 2>&1 &
done
nohup $B/cap 24G $R scripts/train_ghost_gk.py --data-dir ~/tc3-cache-f32/shards/899a46878e7d723f \
  --actions-dir ~/tc3-cache-f32/_actions --home-teams ~/tc3-cache-f32/home_teams.json --variant default \
  --feature-set position_only --skip-permutation-importance --training-platform dgx-spark-aarch64 \
  --output-dir ~/cc/out/ghost_po > ~/cc/logs/ghost_po.log 2>&1 &
nohup $B/cap 16G $B/owner env SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache $R scripts/train_receiver_model.py \
  --out ~/cc/out/receiver --shard-root ~/cc/out/receiver_shards --feature-set public --provider statsbomb \
  --pool-provider gradientsports > ~/cc/logs/receiver.log 2>&1 &
cd ~/cc/gkc && nohup $B/cap 8G $B/public env PYTHONPATH=. SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache \
  ~/cc/venv-gkc/bin/python scripts/train_gk_completion.py --variant skillcorner --providers skillcorner \
  --max-per-provider 10 --mode rebundle --reason "F1b float32-frame rebundle (combined-cycle-completion)" \
  --shard-dir ~/cc/out/gkc_shards > ~/cc/logs/gkc_sk.log 2>&1 &
```

  The ghost flags mirror the archived run. Its printed `Data:` / `Config:` lines and `n_samples` must equal the archived `rt_ggk_po.log` and metrics (179 games, 1039502). Any difference STOPS that run.
- [ ] **Step 2b: D3 validations** (launcher plan Tasks 7–9, in the wave).
  - **D3a — DAS launcher vs serial.**
    - Write `~/cc/inputs/das_strat.json`: about 30 matches in list-matches shape (12 SkillCorner + 12 GS + all 6–7 IDSSE, including the multi-period match from Task 17 Step 3), drawn with `random.Random(0)`.
    - Run them serially in ONE process into a separate shard root:

```bash
nohup $B/owner $R scripts/validate_das_native_parity.py --out ~/cc/out/das_serial --shard-root ~/cc/out/das_serial/shards \
  --match-ids-json ~/cc/inputs/das_strat.json --shards-only --worker-tag serial > ~/cc/logs/das_serial.log 2>&1 &
```

    - After both finish, a scratch compare script reads every strat key's shard from both shard roots. It drops the wall-clock columns (`ms_frame_ref`, `ms_frame_numpy`, `ms_frame_numba`, `ms_frame_periodic`) and asserts `pd.testing.assert_frame_equal(serial, launcher)` with exact equality.
    - Record: n compared, all equal; the per-match speedup (serial wall / match vs launcher wall / match, from the two logs); the launcher's `completed` / `relaunched` (OOM under load = a relaunch with rc 137, reported).
  - **D3b + launcher Task 9 — study fan-out on a small paired set.**
    - Write `~/cc/inputs/small_paired.json` (owner-tier ids, DGX only): 2 GS + 2 public SkillCorner + 2 owner SkillCorner + 2 IDSSE.
    - **Serial reference run:**

```bash
$B/owner $R scripts/train_xcross_attempt.py --providers gradientsports,idsse,skillcorner --match-ids-json ~/cc/inputs/small_paired.json \
  --n-trials 3 --output-dir ~/cc/out/fanout_serial > ~/cc/logs/fanout_serial.log 2>&1
```

    - **Fan-out run.** The launcher substitutes only `{subset}`, `{worker}` (driver) and `{shard_root}` (reduce), so the driver's `--shard-root` is the literal study root printed by `--prep-only`:

```bash
$B/owner $R scripts/train_xcross_attempt.py --providers gradientsports,idsse,skillcorner --match-ids-json ~/cc/inputs/small_paired.json \
  --n-trials 3 --output-dir ~/cc/out/fanout_par --prep-only > ~/cc/out/fanout_par_prep.json
SR=$($R -c "import json; print(json.loads(open('$HOME/cc/out/fanout_par_prep.json').read().strip().splitlines()[-1])['study_root'])")   # the LAST stdout line is the JSON (progress lines precede it; test_expect_variant_wiring.py)
$R scripts/train_xcross_attempt.py --shard-root "$SR" --list-studies > ~/cc/inputs/fan_tags.json
$R scripts/_parallel_launch.py --mode f1b --corpus-json ~/cc/inputs/fan_tags.json --shard-root "$SR" \
  --peak-rss-gib <Task 17 Step 4b peak> --nproc 2 \
  --driver "$R scripts/train_xcross_attempt.py --shard-root $SR --study-list {subset}" \
  --reduce "$R scripts/train_xcross_attempt.py --shard-root {shard_root} --assemble" > ~/cc/logs/fanout_launcher.log 2>&1
```
    - **Compare:**
      - `model.json` byte-identical between `fanout_serial` and `fanout_par`;
      - every `<tag>.study.json` identical;
      - the shared-mmap `X, y` equal to the per-process ones: `scripts._study_shared.load_study_inputs("$SR")` `X` and `y` == the serial run's `_feature_cache/features.parquet` columns and labels (exact).
    - Record the study-worker peak RSS (D3b: small-set measurement, labelled as such).
    - **Gate-passing pair (amendment 2, owner decision 2026-10-04 (b)).** The small set's runs refuse at the Brier gate and write no `model.json`, so the same serial vs fan-out pair also runs on `original17.json`, the public corpus whose xcross runs pass the gates. Same commands as above with `--providers idsse,skillcorner --match-ids-json ~/cc/inputs/original17.json`, `$B/public` in place of `$B/owner`, and the outputs `~/cc/out/fanout17_serial` / `~/cc/out/fanout17_par` (prep JSON `~/cc/out/fanout17_par_prep.json`, tags `~/cc/inputs/fan17_tags.json`, log `~/cc/logs/fanout17_launcher.log`). Compare as above: `model.json` byte-identical, every `<tag>.study.json` identical, mmap `X, y` equal. If this pair also refuses at a gate, record it and report to the owner; that is not a D3 failure.
    - These runs are scratch evidence: nothing from them is committed except the `findings.md` numbers.
- [ ] **Step 3: Monitor every 30–60 min.**
  - Alive count per workload: `ps -eo pid,etime,rss,args | grep python | grep -v grep`.
  - Shard counts against item counts.
  - `dmesg | tail` for OOM kills.

  Relaunch a killed worker with the same command and tag (it resumes). Never use `pgrep -f`.

### Task 19: Receiver gate, reduces, sign-off, Hub smoke, benchmark

- [ ] **Step 1: Receiver gate**, once the receiver run ends: `cd ~/cc/run && PYTHONPATH=. $B/owner $R scripts/validate_receiver_widening.py --rows ~/cc/out/receiver/candidate_rows.parquet --out ~/cc/out/receiver_gate`.
  - **`gate.ship == true`:** the receiver ships (C2).
  - **`false` and `identification.identified == true`:** run the pre-registered fallback in the wave. Write `~/cc/inputs/rcv30.json` (the identified ids, written on the DGX only, never copied out), then run `$B/owner $R scripts/train_receiver_model.py --out ~/cc/out/receiver30 --shard-root ~/cc/out/receiver30_shards --feature-set public --provider statsbomb --pool-provider gradientsports --match-ids-json ~/cc/inputs/rcv30.json`. The gate driver does not persist ids, so export them once with an inline `python -c` that calls `G.identify`.
  - **`false` and not identified:** STOP, report to the owner.
- [ ] **Step 2: Reduces** (after each map completes).
  - **DAS:** the launcher runs the reduce through its `--reduce` hook once every worker has finished (Task 18 Step 1). Check `~/cc/logs/das_launcher.log` for `completed` = the worker count, then check that `~/cc/out/das/metrics.json` exists.
  - If the launcher exited early (a worker failed past its relaunch budget), STOP and report. After the owner's go, relaunch the same launcher command: done markers skip finished keys. Only then run the reduce by hand: `$B/owner $R scripts/validate_das_native_parity.py --out ~/cc/out/das --shard-root ~/cc/out/das/shards --reduce-only`.
  - The other reduces:

```bash
cd ~/cc/run && export PYTHONPATH=. SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache
$R scripts/measure_f1b_feature_delta.py --data-dir ~/tc3-cache --out ~/cc/out/t10 --reduce-only
$B/owner $R scripts/build_gkdv_arm_values.py --out ~/cc/out/gkdv --arm das --providers gradientsports --match-ids-json ~/cc/inputs/gs64.json --reduce-only
$B/owner $R scripts/build_layer2_spells.py --out ~/cc/out/spells --providers gradientsports --match-ids-json ~/cc/inputs/gs64.json --reduce-only
$B/owner $R scripts/build_tf19_instrument_responsiveness.py --out ~/cc/out/tf19 --providers gradientsports,skillcorner,idsse \
   --match-ids-json ~/cc/inputs/corpus179.json --reduce-only
$R scripts/run_signoff_power.py --out ~/cc/out/signoff --spells ~/cc/out/spells/layer2_spells.parquet \
   --arm-values ~/cc/out/gkdv/arm_values_delta_das.parquet --seed 0 \
   --lock-commit 6b242cfbd1369500cf65d26a0c4b3fa49c234da6
```

  The GKDV and spells reduces (`--reduce-only`, amendment 2 F10) refuse an unfinished population, count each match once from the reduce pass's replayed counters, rebuild the tables from exactly the population's shards, and write no worker manifest. A counting pass over the finished partitions would be refused by the overlap guard.
- [ ] **Step 3: Record the `--lock-commit` check** for `findings.md`: `git -C ~/cc/run diff 6b242cf $C1 -- silly_kicks/gkdv/_validate.py`. Measured at `43ff0dd` (amendment 2): comments plus the additive `_ARM_DIRECTION_KEY` / `expected_direction_for_arm` (`07a88f6`, PR-S175), which the sign-off never imports. Record that, and check that `ICC_ANCHORS` and `ATT_RELATIVE_ANCHORS` (the only `_validate` names `run_signoff_power.py` imports) are unchanged; a change to either STOPS the sign-off and is reported.
- [ ] **Step 4: Hub smoke at C1.** Run it locally on this machine. The local repo must be checked out at C1 with a clean tree (it is, right after C1): `PYTHONPATH=. python scripts/validate_hub_variants.py --out <scratch>/hub` → `hub_smoke.json`. No `--require-*` flag here: the cards change in C2 and the mirrors are republished only after the release, so both comparisons are recorded, not gated. Expected at C1: `cards_mismatched` == the two sweeper repos; `mirrors_mismatched` == all four mirrors (wheel `3ca609f` vs Hub `adafb72` / `b68328a`); `load_refused` == the two sweeper mirrors. Any other refused repo: STOP and report.
- [ ] **Step 5: Benchmark, alone**, after every other process has exited (confirm with `ps`):

```bash
cd ~/cc/run && export PYTHONPATH=. SILLY_KICKS_CORPUS_CACHE_DIR=$HOME/pining_cache
nohup $B/owner env NUMBA_NUM_THREADS=20 $R scripts/validate_das_native_parity.py --out ~/cc/out/das \
  --benchmark-sample-json ~/cc/inputs/bench_sample.json --benchmark --repeat 3 > ~/cc/logs/bench.log 2>&1 &
```

  `bench_sample.json` comes from Task 17 Step 5. The benchmark refuses `--match-ids-json` and reads its matches from the sample file only (Task 11, B r2 C5).
- [ ] **Step 6: Clean-tree proof:** `git -C ~/cc/run status --porcelain | wc -l` → 0. The gkc checkout is dirty by design.

### Task 20: Collect and check

- [ ] **Step 1: Copy back** to `$CC_OUT` (Global Constraints: outside the repo, private), and verify `sha256sum` on both sides:
  - the re-fit artifact dirs: `xshot_occurrence_v1/` and `xcross_attempt_v1/` without `_feature_cache`, `studies`, `shards`, `probe_shards` and `_probe_sample*`; `ghost_po/ghost_gk_v1`; `receiver/{model,metrics.json}` (the 327 run, or the 30 fallback);
  - the gkc weights from `~/cc/gkc/silly_kicks/tracking/_gk_completion_weights/skillcorner/`;
  - `das/{metrics,performance}.json`;
  - `t10/metrics.json` — plus `f1b_feature_delta.parquet` into the PRIVATE archive only;
  - `tf19/{metrics.json,named_keeper_signs.parquet}` — check that the parquet holds keeper aggregates and no match ids before it may be committed;
  - `signoff/*`;
  - `receiver_gate/receiver_gate.json`;
  - the GKDV and spells manifests;
  - the D3 evidence: the launcher logs (`das_launcher.log`, the fan-out launcher log), the D3a compare output, the D3b `time -v` outputs, and the fan-out compare output. These are scratch evidence; only their numbers enter `findings.md`.
- [ ] **Step 2: Acceptance per run** (spec §8, §11). Use a scratch check script; any miss STOPS the cycle and is reported to the owner.
  - **Every artifact:** `run_commit == C1`, `run_tree_dirty false`. Sharded ones: `commits_seen == [C1]`. Completeness is the reduce itself: each reduce refuses a population with an unaccounted key, so `accounted == listed` holds by construction and is not re-checked as evidence (B r2 C7). What IS checked is that the listed population equals the pre-registered one (below).
  - **GKDV / spells** (not commit-keyed, spec §2): each manifest's `run_commit == C1`, and every shard listed in it was written by this run (its manifest names C1). The reduce artifacts (`arm_values_manifest.json`, `layer2_spells_manifest.json`; amendment 2 F10): `reduce_mode driver-reduce-only`, `population_size 64`, `n_matches 64`, `commit_consistent`, and GKDV `conservation_holds`.
  - **xshot/xcross ×4:**
    - `shipped_variant public`;
    - `corpus_match_ids == bundled_public_arm_pairs()`;
    - `reproducibility public`;
    - xshot `n_rows 156106`, `n_positive 34649`; xcross `91999` / `2849` (exact);
    - acceptance gates true;
    - `position_only` PR-AUC within ±2·`pr_auc_std` of the committed candidate — otherwise report. `default` deltas are recorded, not gated;
    - `load()` succeeds;
    - **xcross only (D5c):** the GS held-out probe ran. In `metrics.json` (keys written by `assemble_studies`; `train_xcross_attempt.py:719-727` at `ba9c151`, before this cycle's edits):
      - `probe_sample_matches` == the two `gs_probe.json` pairs;
      - every `probe_sample_in_training_folds` value is `false`;
      - `gk_substitution_probe` is present; `tf19_ready` is recorded, not gated;
      - `corpus_match_ids` holds no `gradientsports` pair.
  - **gkc skillcorner:** `requested_match_ids == bundled_public_arm_pairs(("skillcorner",))`, `artifact_label public`, `n_rows 542`, `n_matches 10`, `mode rebundle`, `bundled true`.
  - **ghost `position_only`:**
    - `n_games 179`, `n_samples 1039502`;
    - `reproducibility restricted`, with a note naming C1;
    - `rfcde_weights.npz` compared with the archived `rt_ggk_po` (record whether byte-identical).
  - **receiver:** `corpus_visibility restricted` (D7a), `providers_trained ["statsbomb"]`, `n_matches` 327 (or 30 on the fallback); the gate outcome under the D8 combined rule (`decision_rule`, `margin 0.01` recorded in `receiver_gate.json`).
  - **T10:** `n_matches 179`, `n_failed 0`, every model present; no model's `match_status_counts` holds an `error:` status, and ghost_gk holds `ok` rows (amendment 2 F9).
  - **DAS:**
    - `population.listed_per_provider == {"gradientsports": 64, "idsse": 7, "skillcorner": 909}` (the Task 17 Step 5 counts);
    - every `n_outside_golden_bound` count 0 (4 cells);
    - every cell's `numba_compared` > 0, and every `numba_vs_numpy_max_abs` finite (B r2 CCC-PLAN-23: no vacuous numba cell);
    - `finite_mask_mismatches` 0;
    - `d_key_frames`, `finite_counts` and `direction` present; the multi-period match of Task 17 Step 3 contributes `d_key_frames > 0`;
    - `commit_consistent`.
  - **D3 (launcher):**
    - D3a: every stratified match's launcher shard equals its serial shard (timing columns dropped); n compared = the strat list length;
    - D3b / launcher Task 9: `model.json` and every `<tag>.study.json` byte-identical between the serial and fan-out runs; shared-mmap `X` equals the per-process features; the study-worker peak RSS recorded. `model.json` is compared on the `original17` pair (amendment 2: the small set refuses at the Brier gate); a gate refusal there is recorded and reported, not a D3 failure.
    - A D3 failure STOPS the cycle: the launcher's DAS shards are then not trusted, and the owner decides.
  - **performance:** the five §4.2 targets; `contention.foreign_cpu_fraction < 0.05`.
  - **TF-19:** `population_size 179` (the artifact carries no `n_matches`; measured at `43ff0dd`), `reduce_mode driver-reduce-only`.
  - **sign-off:** both upstream `run_commit == C1`, `commit_consistent`.
  - **Hub smoke:** `all_finite`, population exact, `ghost-gk-v1` role `hf_only`; `cards_mismatched` == the two sweeper repos (the measured pre-C2 state); `mirrors_mismatched` == all four mirrors (pre-registered: wheel `3ca609f` vs Hub `adafb72` / `b68328a`, B r5 CCC-PLAN-43); `load_refused` == the two sweeper mirrors (owner-approved 2026-10-02). Any other value is reported.
- [ ] **Step 3:** Leave the DGX as found: `rm -rf ~/cc/smoke`. Keep `~/cc/out` until C2 merges.

---

## PHASE C — commit C2 and release

### Task 21: C2 content

- [ ] **Step 1: Weights.**
  - Copy the re-fit dirs:
    - xshot/xcross into `_{xshot,xcross}_weights/{default,position_only}/` (`model.json`, `metadata.json`, `metrics.json`, `SHA256SUMS`);
    - gkc skillcorner;
    - ghost `position_only` (`rfcde_weights.npz`, `metadata.json`, `metrics.json`, `SHA256SUMS`);
    - receiver (`model.json`, `SHA256SUMS` from `model/`, plus `metrics.json`).
  - No hand edits. The `reproducibility` keys now come from the trainers.
- [ ] **Step 1b: Cards (D9, spec §9).** Every number and commit a card states is copied from the bundled artifact it describes, and a test ties them together, so a card cannot drift from the wheel again.
  - **Failing test first** — `tests/test_model_cards_match_bundles.py`:

```python
"""Model cards state the bundled artifacts they describe, each value in its own place
(combined-cycle spec 9, D9; B r4 CCC-PLAN-33: a value matched "anywhere" lets a stale table cell pass
whenever the right number survives in a history paragraph)."""

import json
import re
from pathlib import Path

import pytest

from silly_kicks import __version__

_CARDS = Path("docs/huggingface/model-cards")
_W = Path("silly_kicks/tracking")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_GK_ROWS = {
    "Held-out CV euclidean MAE": [
        ("cv_mae_euclidean_mean", ".3f"),
        *[(("per_provider_mae_euclidean", p), ".3f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "> 30 m high-sweeper stratum MAE": [("high_sweeper_stratum_mae_mean", ".2f")],
    "Training corpus": [("n_games", "d"), ("n_samples", "d")],
}
_OUTFIELD_ROWS = {
    "Held-out CV euclidean MAE": [
        ("cv_mae", ".2f"),
        *[(("cv_mae_by_provider", p), ".2f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "Per-possession CV MAE": [(("cv_mae_by_possession", p), ".2f") for p in ("in_possession", "out_of_possession")],
    "Per-slot CV MAE (slots 1&ndash;4)": [(("cv_mae_by_slot", s), ".2f") for s in ("1", "2", "3", "4")],
    "Training corpus": [("n_games", "d"), ("n_rows", "d")],
}
# The four Hub mirrors of wheel-bundled variants (spec 0.10): card -> (bundled dir, metrics-table rows).
_MIRRORS = {
    "ghost-gk-sweeper-v1": ("_ghost_gk_weights/sweeper", _GK_ROWS),
    "ghost-gk-sweeper-position-only-v1": ("_ghost_gk_weights/sweeper_position_only", _GK_ROWS),
    "ghost-outfield-v1": ("_ghost_outfield_weights/default", _OUTFIELD_ROWS),
    "ghost-outfield-position-only-v1": ("_ghost_outfield_weights/position_only", _OUTFIELD_ROWS),
}
# Hub-only cards that describe a wheel-bundled sibling: card -> the bundled dir it names.
_HF_ONLY = {
    "ghost-gk-v1": "_ghost_gk_weights/default",
    "xshot-occurrence-v1": "_xshot_weights/default",
    "xshot-occurrence-position-only-v1": "_xshot_weights/position_only",
    "xcross-attempt-v1": "_xcross_weights/default",
    "xcross-attempt-position-only-v1": "_xcross_weights/position_only",
}
_WHEEL_CARDS = ["_receiver_weights/default", "_gk_completion_weights/default", "_gk_completion_weights/skillcorner"]


def _card(name: str) -> str:
    return (_CARDS / f"{name}-model-card.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def _json(rel: str, name: str) -> dict:
    return json.loads((_W / rel / name).read_text(encoding="utf-8"))


def _get(doc: dict, key):
    return doc[key] if isinstance(key, str) else doc[key[0]][key[1]]


def _short(rel: str) -> str:
    return _json(rel, "metadata.json")["training_commit"][:7]


def _row_value(text: str, label: str) -> str:
    """The value cell of the ONE table row whose label cell (bold stripped) is ``label``."""
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.lstrip().startswith("|")
    ]
    hits = [r[1] for r in rows if len(r) >= 2 and r[0].replace("*", "").strip() == label]
    assert len(hits) == 1, f"expected exactly one table row labelled {label!r}, found {len(hits)}"
    return hits[0]


def _provenance_line(metrics: dict) -> str:
    """The ONE provenance line a wheel MODEL_CARD.md carries, built from its metrics.json."""
    fields = [f"`run_commit={metrics['run_commit'][:7]}`"]
    fields += [f"`{k}: {metrics[k]}`" for k in ("corpus_visibility", "artifact_label") if k in metrics]
    fields += [f"{metrics[k]} {unit}" for k, unit in (("n_matches", "matches"), ("n_rows", "rows")) if k in metrics]
    return f"**Provenance (silly-kicks {__version__}).** " + " · ".join(fields)


def test_card_population_is_exact():
    assert {p.name.removesuffix("-model-card.md") for p in _CARDS.glob("*-model-card.md")} == set(_MIRRORS) | set(_HF_ONLY)


@pytest.mark.parametrize("card", sorted(_MIRRORS))
def test_mirror_card_states_the_f1b_refit_paragraph(card):
    rel = _MIRRORS[card][0]
    head = f"**F1b float32-frame re-fit (silly-kicks {__version__} / ADR-106; `training_commit={_short(rel)}`).**"
    assert _card(card).count(head) == 1


@pytest.mark.parametrize(("card", "label"), [(c, lab) for c, (_rel, rows) in sorted(_MIRRORS.items()) for lab in rows])
def test_mirror_card_table_row_equals_the_bundle(card, label):
    rel, rows = _MIRRORS[card]
    metrics = _json(rel, "metrics.json")
    want = [format(_get(metrics, key), fmt) for key, fmt in rows[label]]
    assert _NUM.findall(_row_value(_card(card), label)) == want, (card, label)


@pytest.mark.parametrize("card", sorted(_HF_ONLY))
def test_hub_only_card_states_the_unchanged_note_for_its_bundled_sibling(card):
    rel = _HF_ONLY[card]
    note = (
        f"In silly-kicks {__version__} the wheel's bundled `{Path(rel).name}` was re-fit on float32-stored frames "
        f"(`training_commit={_short(rel)}`). This Hub artifact is unchanged: trained on float64 frames"
    )
    assert _card(card).count(note) == 1


@pytest.mark.parametrize("rel", _WHEEL_CARDS)
def test_wheel_model_card_states_its_provenance_line(rel):
    text = (_W / rel / "MODEL_CARD.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("**Provenance (silly-kicks") == 1
    assert _provenance_line(_json(rel, "metrics.json")) in text


def test_a_stale_table_cell_fails_even_when_the_right_number_is_elsewhere():
    """Anti-vacuity for CCC-PLAN-33: the row check must not be satisfied by a history paragraph."""
    card = "| Metric | Value |\n|---|---|\n| Held-out CV euclidean MAE | **6.10 m** (per-provider: 6.14 / 5.92 / 6.33) |\n"
    card += "\nHistory: the aggregate was 6.00 m.\n"
    assert _NUM.findall(_row_value(card, "Held-out CV euclidean MAE")) == ["6.10", "6.14", "5.92", "6.33"]
    assert _NUM.findall(_row_value(card, "Held-out CV euclidean MAE"))[0] != "6.00"
    with pytest.raises(AssertionError, match="exactly one table row"):
        _row_value(card + card, "Held-out CV euclidean MAE")  # a duplicated (stale) table is refused
```

  - Before editing a card, run the test: expect FAIL on every parametrization whose bundle moved. Check the population test passes as-is: it pins the nine cards.
  - Check that `metadata.json` of every `_HF_ONLY` dir carries `training_commit`. If one does not, STOP and report; never switch the key silently.
  - The test runs after the Step 10 version bump (it reads `silly_kicks.__version__`); write the cards with the D10 version literally.
  - **Mirror cards (4).** Add the F1b paragraph above the "Both-axes" history paragraph where the card has one (the two sweeper cards), which stays as history; the two outfield cards have none, so there it goes directly above the `| Metric | Value |` table (B r5, outside its round). Its first sentence is exactly "**F1b float32-frame re-fit (silly-kicks <D10> / ADR-106; `training_commit=<short>`).**", followed by: re-fit on float32-stored frames, same 179-game corpus. In the existing `| Metric | Value |` table:
    - replace every value the test names with the bundled `metrics.json` value, in the card's existing format and order (the row labels stay exactly as they are);
    - add one row `| Training corpus | <n_games> games / <n_samples or n_rows> samples |` with plain integers (no thousands separators);
    - leave exactly one row per label in the card (history paragraphs keep prose, never a second table row).
  - **Hub-only cards (5).** One note where the card describes the wheel: "In silly-kicks <D10> the wheel's bundled `<default|position_only>` was re-fit on float32-stored frames (`training_commit=<short>`). This Hub artifact is unchanged: trained on float64 frames at `<the commit this card already states>`." (The Hub `metadata.json` of the four xshot/xcross repos records no `training_commit`, spec §0.10; the card's own stated commit is the reference.) The Hub smoke (`hub_smoke.json`) is cited for its finite scores on the float32 canonical frame.
  - **Wheel `MODEL_CARD.md` (3).** Each carries exactly one provenance line, built by `_provenance_line` from its `metrics.json`: "**Provenance (silly-kicks <D10>).** `run_commit=<short>` · `corpus_visibility: …` · `artifact_label: …` · `<n_matches> matches` · `<n_rows> rows`", including only the fields its `metrics.json` has, in that order.
    - **Receiver:** rewritten from its `metrics.json` and `receiver_gate.json`. Remove the "open-data" wording. State `corpus_visibility: restricted` (D7) and the licensed SB360 source. State the gate under the D8 rule. Keep the owner-variant paragraph, marked "(measured at `08347cd`)".
    - **gk_completion `default`:** the provenance line for the reused F1b rebundle (`run_commit` `3ca609f`).
    - **gk_completion `skillcorner`:** the provenance line for the C1 rebundle on the 10-match public arm.
  - **Anti-vacuity check, before trusting the gate:** in a scratch copy of `ghost-outfield-v1`, set the table's MAE cell to a wrong value and leave the right value in a history sentence. The row test must FAIL. Then remove the copy.
  - Then `git grep -n "30 WC2022\|0\.510\b\|open-data matches"` outside history, and fix any live statement.
  - Run the test right after Step 10's version bump (it reads `silly_kicks.__version__`): PASS. Before the bump only the version-bearing assertions fail; the row and population tests already pass.
- [ ] **Step 2: G2 tightening.**
  - Extend `_x_policy()` with `("metrics.json", "corpus_match_ids"): bundled_public_arm_pairs()`, `("metrics.json", "shipped_variant"): "public"` and `("metrics.json", "reproducibility"): "public"`.
  - gkc skillcorner: `artifact_label "public"`, `all_public True`, `requested_match_ids == bundled_public_arm_pairs(("skillcorner",))`.
  - Receiver: `corpus_visibility` per D7.
  - Ghost `position_only`: `("metrics.json", "reproducibility"): "restricted"`.
  - Add an `ANCHOR` dict: dir → `3ca609f…` for the 6 reused dirs, and the C1 SHA for the 7 re-fit dirs. A parametrized test asserts `metadata.json:training_commit` (ghost, gof, xshot, xcross) or `metrics.json:run_commit` (gkc, receiver) equals it.
  - Add a test that the ghost `position_only` `reproducibility_note` contains its own `training_commit`.
- [ ] **Step 3: `tests/tracking/test_position_only_bundled.py`.**
  - Add `_C5 = "<C1 SHA>"  # combined-cycle completion re-fits (xshot/xcross position_only, ghost position_only)`.
  - The `expected` dict → `_C5` for all three.
  - Comments updated; older constants kept.
- [ ] **Step 4: Artifacts into `docs/research/`.**
  - `das_native_parity/{metrics,performance}.json`, plus a `README.md` that states:
    - the corpus bound;
    - that the map's `timings_ms_per_frame` are contended and not for ratios (`performance.json` is the speed record);
    - the D-KEY figure.
  - `f1b_float32/{metrics.json,receiver_gate.json,hub_smoke.json}`.
  - `tf19_instrument_responsiveness/{metrics.json,named_keeper_signs.parquet}`, replacing the old ones. Every number its `findings.md` quotes is re-read and updated.
  - `tf19_signoff_power/metrics.json`, replacing the old one, with `README.md` updated likewise.
  - A verdict that FLIPS is surfaced to the owner before C2.
- [ ] **Step 5: `docs/research/f1b_float32/findings.md`:**
  - T10 per model;
  - the receiver gate: identification, negative control, gate numbers, the D8 combined rule (point estimate ≥ 0 AND 95 % LB > −0.01, with its rationale from spec §7), decision, fallback if used;
  - the xcross held-out GS probe (D5c): the two probe matches, held out of every training fold, and the probe result;
  - the D3 launcher validations: D3a (n stratified matches, all shards identical, the serial vs launcher per-match wall), D3b (study fan-out `model.json` byte-identical, mmap `X, y` identical, study-worker peak RSS on the small set, labelled as a small-set figure);
  - the re-fit acceptance table: pre-registered vs measured counts; the `position_only` metric deltas; the `default` deltas, with the label-change explanation (spec §0.1b);
  - the ghost `position_only` re-fit vs the archive;
  - the Hub smoke, with its pre-registered C1 expectations: `cards_mismatched` == the two sweeper repos and `mirrors_mismatched` == all four mirrors (wheel `3ca609f` vs Hub `adafb72` / `b68328a`) and `load_refused` == the two sweeper mirrors, all three cleared by the post-release pushes (Task 22 Steps 11-13);
  - the xcross `default` record change (D5);
  - the `--lock-commit` check;
  - the corpus bounds (the 17 / 10 / 179 / 64 / 327 populations, anchors).
- [ ] **Step 6: Invalidations + exemptions.**
  - **Delete** `docs/research/tf19_signoff_power/invalidation.json`. Remove its `_UNPROVENANCED` entry from `tests/scripts/test_artifact_provenance_output.py` in the same change. State in the signoff `README.md` that the annotated 6b242cf artifact was superseded by the C1 re-run.
  - **Create** `docs/research/tf24_stage2_refresh/invalidation.json`, filled from the committed parity artifact. A field that cannot be measured from it is stated as such, never guessed. Add an `_UNPROVENANCED` entry for it, with the reason "sibling annotation of a historical artifact; not driver-produced (owner-approved with spec rev 3)".

```json
{
  "_about": "Sibling annotation for the TF-24 Stage-2 calibration report. That report is historical (CHANGELOG PR-S152: 'within noise, no default change') and is NOT re-run. This file classifies each DAS-derived field it used against the native DAS engine (ADR-107) and periodic quadrature (ADR-108).",
  "annotated_at": "<YYYY-MM-DD>",
  "annotates": "docs/research/tf24_stage2_refresh/calibration_report.json",
  "das_engine_commit": "<C1 SHA>",
  "corpus_shift_source": "docs/research/das_native_parity/metrics.json",
  "fields": {
    "das_team":     {"role": "calibration feature", "changed": true, "measured_shift": "<per-provider quadrature_shift_das median / p90 / max>"},
    "das_opponent": {"role": "calibration feature", "changed": true, "measured_shift": "<same source>"},
    "das_diff":     {"role": "calibration feature", "changed": true, "measured_shift": "<same source>"},
    "das_degraded": {"role": "degrade flag", "changed": "<true|false from the reason_counts comparison>", "note": "<measured basis>"}
  },
  "cites": ["ADR-107", "ADR-108"],
  "decision_unchanged_because": "<one sentence from the measured magnitude>"
}
```

- [ ] **Step 7: `estimate_das_cost` constants** from `performance.json` (`summary.seconds_per_frame.numba_serial` / `.numpy`, `prange_efficiency["16"]`), rounded as in Task 13 Step 3. The comment cites `performance.json` and the C1 SHA. `test_das_cost_guardrail.py` stays green.
- [ ] **Step 8: Artifact gates.** `tests/tracking/test_das_parity_artifact.py`:

```python
"""Commit-2 gate on the committed DAS corpus artifacts (das-native spec sections 4 and 7.2;
combined-cycle-completion spec 12 D1/D2)."""

import json
import math
import re
from pathlib import Path

import pytest

_DIR = Path("docs/research/das_native_parity")
_M = json.loads((_DIR / "metrics.json").read_text(encoding="utf-8"))
_P = json.loads((_DIR / "performance.json").read_text(encoding="utf-8"))
_PROVIDERS = sorted(_M["providers"])
# das-native spec section 4.2 -- copied, never relaxed here.
_TARGETS = {"ref_over_numba_serial": 10.0, "ref_over_numpy": 2.0, "add_das_speedup": 10.0, "das_xfns_speedup": 50.0}
_PRANGE_EFFICIENCY_AT_16 = 0.6


def test_provenance_is_clean_and_single_commit():
    for doc in (_M, _P):
        assert doc["run_tree_dirty"] is False
        assert re.fullmatch(r"[0-9a-f]{40}", doc["run_commit"])
    assert _M["run_commit"] == _P["run_commit"]
    assert _M["n_failed"] == 0
    assert _M["commit_consistent"] is True and _M["commits_seen"] == [_M["run_commit"]]


def test_population_is_the_pre_registered_one():
    # Completeness itself is enforced by the reduce, which refuses an unaccounted key (B r2 C7);
    # this pins WHICH population was reduced (the Task 17 Step 5 owner-token counts).
    assert _M["population"]["listed_per_provider"] == {"gradientsports": 64, "idsse": 7, "skillcorner": 909}


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_parity_inside_the_golden_bounds_in_all_four_cells(provider):
    p = _M["providers"][provider]
    counts = p["n_outside_golden_bound"]
    for eng in ("numpy", "numba"):
        for grain in ("team", "player"):
            assert counts[eng][grain] == {"das": 0, "as": 0}, (eng, grain)
    for grain in ("team", "player"):  # no vacuous numba cell (B r2 CCC-PLAN-23)
        for out in ("das", "as"):
            assert p["numba_compared"][grain][out] > 0, (grain, out)
            assert math.isfinite(p["numba_vs_numpy_max_abs"][grain][out]), (grain, out)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_zero_finite_mask_mismatches_and_counts_recorded(provider):
    p = _M["providers"][provider]
    assert p["finite_mask_mismatches"] == {"team": 0, "player": 0}
    assert set(p["finite_counts"]) == {"team", "player"}
    assert p["d_key_frames"] >= 0 and p["direction"]["n_compared"] > 0


def test_the_d_key_path_was_exercised():
    # Task 17 Step 3 found a match whose frame_id restarts per period FROM THE FRAMES (B r2 CCC-PLAN-22).
    assert any(_M["providers"][p]["d_key_frames"] > 0 for p in _PROVIDERS)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_quadrature_shift_is_recorded(provider):
    q = _M["providers"][provider]["quadrature_shift_das"]
    assert all(math.isfinite(q[k]) for k in ("median", "p90", "max"))


@pytest.mark.parametrize("name", sorted(_TARGETS))
def test_speed_target(name):
    assert _P["summary"][name] >= _TARGETS[name], f"{name} = {_P['summary'][name]:.2f} < {_TARGETS[name]}"


def test_prange_efficiency_at_16_threads():
    assert _P["summary"]["prange_efficiency"]["16"] >= _PRANGE_EFFICIENCY_AT_16


def test_benchmark_ran_on_a_quiet_box():
    c = _P["contention"]
    assert c["foreign_cpu_fraction"] is not None and c["foreign_cpu_fraction"] < c["max"]


def test_both_paths_ran_in_one_pandas_major():
    majors = {r[k].split(".")[0] for r in _P["per_match"] for k in ("pandas_new", "pandas_old") if k in r}
    assert majors == {"2"}


def test_every_path_run_that_did_not_fit_is_recorded():
    # Phase B amendment F7: a path run above the memory ceiling is a recorded RESULT (the old path cannot
    # score a GS match within it). Each speedup still rests on at least one finished pair, and the new path
    # never hits the ceiling.
    s = _P["summary"]
    assert s["speedup_n"]["add_das"] > 0 and s["speedup_n"]["das_xfns"] > 0
    assert s["new_path_over_memory"] == {}
    for r in _P["per_match"]:
        rec = r.get("old_path_over_memory")
        if rec is not None:
            assert rec["limit_gib"] == _P["path_memory_limit_gib"]
            assert rec["phase"] in {"load", "add_das", "das_xfns"}


def test_adr108_quotes_the_artifact():
    adr = next(Path("docs/superpowers/adrs").glob("ADR-108-*.md")).read_text(encoding="utf-8")
    assert "COMMIT-2 PLACEHOLDER" not in adr
    for provider in _PROVIDERS:
        assert f"{_M['providers'][provider]['quadrature_shift_das']['median']:.3g}" in adr
```

`tests/test_research_artifacts_carry_no_ids.py` (spec §11.8):

```python
"""No committed research artifact carries an owner-tier match id (combined-cycle spec 11.8)."""

import re
from pathlib import Path

import pandas as pd
import pytest

_DIRS = [Path("docs/research/f1b_float32"), Path("docs/research/das_native_parity")]
_ID_COLUMNS = {"match_key", "match_id", "game_id"}


@pytest.mark.parametrize("path", sorted(p for d in _DIRS for p in d.glob("**/*.parquet")), ids=str)
def test_no_id_columns_in_committed_tables(path):
    assert not (_ID_COLUMNS & set(pd.read_parquet(path).columns)), path


@pytest.mark.parametrize("path", sorted(p for d in _DIRS for p in d.glob("**/*.json")), ids=str)
def test_no_id_keys_in_committed_json(path):
    text = path.read_text(encoding="utf-8")
    assert '"match_key"' not in text and '"match_id"' not in text, path
    # B r4 CCC-PLAN-37: a "game_id" KEY too. Key-only: the DAS driver legitimately names the column
    # "game_id" in declared frame-key lists (`_FRAME_KEYS`), which are not ids.
    assert not re.search(r'"game_id"\s*:', text), path
    # a joined shard key used as a JSON key or value ("<provider>__<id>") carries an id as well
    assert not re.search(r'"(gradientsports|skillcorner|idsse|statsbomb|sportec)__', text), path


@pytest.mark.parametrize("d", _DIRS, ids=str)
def test_the_scan_sees_every_dir(d):
    """An empty or missing dir must FAIL, not skip silently (B r4 CCC-PLAN-37)."""
    assert any(d.glob("**/*.json")), d
```

  The parquet parametrization may legitimately be empty (no per-match table is committed); the JSON one may not. (The rev-3 `import json` was unused, F401; it is replaced by `import re`.)

Before trusting it, check that it bites: it must fail on the archived `f1b_feature_delta.parquet` (copy it temporarily into a scratch dir and point a one-off parametrization at it, then remove it). A miss in any gate is surfaced to the owner, never relaxed.
- [ ] **Step 9: ADR-108.** Replace the `COMMIT-2 PLACEHOLDER` with a per-provider table: median / p90 / max of `quadrature_shift_das`, each `:.3g`, citing `metrics.json` and the C1 SHA. Add the D-KEY figure and the downstream re-materialise notice. **ADR-106:** fill in the receiver outcome.
- [ ] **Step 10: Release docs.**
  - **NEXT-FREE, read-only:** `git fetch origin && git show origin/main:silly_kicks/_version.py && git show origin/main:CHANGELOG.md | head -20`. Expected 4.128.0 (D10) / PR-S200. Do not merge `main` into the branch.
  - **`silly_kicks/_version.py`** → the D10 version.
  - **`CHANGELOG.md`** — a release entry containing:
    - **BREAKING:**
      - F1b frame schema (float32 coordinates, `team_id` → `category`);
      - native DAS: `[das]` extra deleted, `**kwargs`/`use_progress_bar` removed, new `ValueError`s, every DAS value moves — the per-provider corpus median / p90 / max from the artifact;
      - every frame-geometry bundled model re-fit on float32 frames: predictions move by the measured amounts; the `default` xshot/xcross re-fits also absorb the label changes since `6e3a132`;
      - the receiver 30 → 327 widening, with gate numbers and the corrected `restricted` visibility (D7);
    - **Added:**
      - `silly_kicks.mcp` (#264);
      - `detected_mask` (#260);
      - the guard flags (`--expect-variant`), T10 / TF-19 / DAS CLI flags, the Hub and receiver drivers;
      - the card-only Hub push seam `scripts/publish_model_card.py` (`publish_card_only`, ADR-088 amendment); both publish seams now stage cards with LF line endings;
      - new metrics keys (`corpus_match_ids` / `corpus_match_ids_sha256`, `requested_match_ids`, `reproducibility`, `n_outside_golden_bound`, `finite_counts`, `d_key_frames`, `direction`, `population.accounted`, `commits_seen`, `n_accounted`, `reduce_mode`) and `performance.json`;
      - `estimate_das_cost` is now engine-aware;
    - **Hyrum:**
      - the xcross `default` record shape (D5);
      - DAS shard schema `-4`;
      - Hub: the five HF-only variants (xshot ×2 and xcross ×2 `sc_extended`, ghost-GK `full`) keep their float64-trained weights; only their cards are refreshed. The four mirrors (`ghost-gk-sweeper-v1` and `ghost-gk-sweeper-position-only-v1`, stale since `adafb72`; `ghost-outfield-v1` and `ghost-outfield-position-only-v1`) are republished from this release to match the wheel. Both happen right after the release (D9);
    - **Downstream notice** (das-native §11.2) for the owner to relay: lakehouse re-materialise `das_*` + gkdv `delta_das` together with the F1b frame-geometry re-materialise; the `<5` pin (D10).
  - **`TODO.md`:** groomed — shipped items deleted, top summary replaced, no history.
  - **Version check:** `git grep -nF "4.127.0"` → only history, CHANGELOG, and the old-path env checks.
- [ ] **Step 11: Wheel verification.** `python -m build --wheel` into the scratchpad. For each changed weights dir, compare every file's SHA-256 inside the wheel against the tree. The `.gitattributes` `binary` pins must be present.

### Task 22: C2 verification, reviews, owner gates, release

- [ ] **Step 1:** Task 15 Steps 1–2 again (format + lint + types; both pandas legs; slow).
- [ ] **Step 2:** `/final-review`, including Phase 3.5 (release hygiene) and the C4 check.
- [ ] **Step 3:** External implementation review (owner-coordinated); freeze, apply, re-verify.
- [ ] **Step 4: OWNER GATE — commit C2** (after amendment 3: on a new branch off `main`, PR-2).
  - Show the status, the stat, the file list and the draft message: `feat(release)!: F1b re-fits + native-DAS parity/perf + T10 + 7.3 downstream -- silly-kicks <D10> (PR-S200)`, body summarising Task 21, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Wait for an explicit yes.
- [ ] **Step 5: OWNER GATE — push.** `git push` on its own explicit yes.
- [ ] **Step 6: OWNER GATE — PR.** `gh pr create` on its own explicit yes. The body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- [ ] **Step 7:** CI green. Report the result; do not poll in a tight loop.
- [ ] **Step 8: OWNER GATE — merge.** `gh pr merge --admin --merge --delete-branch`: non-squash, and `--admin` bypasses branch protection, stated in the gate (after amendment 3 PR-2 may squash: one commit, and M is already on `main`). Then verify `git merge-base --is-ancestor <C1> origin/main && git merge-base --is-ancestor 3ca609f origin/main` (after amendment 3, `<C1>` is M).
- [ ] **Step 9:** Post-merge main CI green → **OWNER GATE — tag**. The owner publishes. Verify `https://pypi.org/pypi/silly-kicks/<version>/json` returns 200, and that the published wheel bundles the weights (Task 21 Step 11 against the PyPI wheel).
- [ ] **Step 10: Hub-push prerequisites (D9; this session performs the pushes, owner 2026-10-02).** Start only after Step 9 holds: tag pushed, PyPI returns 200, published wheel verified. The pushes are not a commit, but they publish to a public org and cannot be taken back. Each batch is therefore its own explicit OWNER GATE (Steps 11 and 12), exactly like Steps 4–9 (A r4 A-SPEC-05 / A-PLAN-02). D9 authorizes the session to run them; it does not pre-approve any single push.
  - **Checkout:** a clean checkout of the release tag (`git status --porcelain` empty; `git describe --tags --exact-match` = the tag). Run every command from its root.
  - **Identity:** `python -c "from huggingface_hub import whoami; w = whoami(); print(w['name'], [(o['name'], o.get('roleInOrg')) for o in w.get('orgs', [])])"` must show write access (`write` or `admin`) to `silly-kicks`. Otherwise STOP and ask the owner; never change tokens.
  - **Before-state (for Step 13):** record each Hub-only repo's weight-file blob ids now, into `<scratch>/hub_pre_blobs.json`:

```bash
PYTHONPATH=. python -c "
import json
from huggingface_hub import HfApi
api = HfApi()
repos = ['xshot-occurrence-v1', 'xshot-occurrence-position-only-v1', 'xcross-attempt-v1',
         'xcross-attempt-position-only-v1', 'ghost-gk-v1']
out = {r: {s.rfilename: s.blob_id for s in api.model_info('silly-kicks/' + r, files_metadata=True).siblings
           if s.rfilename != 'README.md'} for r in repos}
print(json.dumps(out, indent=2))" > <scratch>/hub_pre_blobs.json
```

- [ ] **Step 11: OWNER GATE — mirror republish (4 repos, weights + card).** A mirror card is never pushed alone: it describes weights the repo must serve (`publish_model_with_card`, ADR-088).
  - Run all four with `--verify-only` first (local SHA-256 + load + sanity, no network). Show the owner the four exact invocations below and the complete dry-run output. Then **wait for an explicit yes**. A clean `--verify-only` run is not the approval.
  - On the yes, run the same four without `--verify-only`, in order. Stop at the first failure and report it. Never re-push after a failure without a new explicit yes.

```bash
python scripts/publish_ghost_gk.py --artifact-dir silly_kicks/tracking/_ghost_gk_weights/sweeper \
  --repo-id silly-kicks/ghost-gk-sweeper-v1 --model-card docs/huggingface/model-cards/ghost-gk-sweeper-v1-model-card.md
python scripts/publish_ghost_gk.py --artifact-dir silly_kicks/tracking/_ghost_gk_weights/sweeper_position_only \
  --repo-id silly-kicks/ghost-gk-sweeper-position-only-v1 \
  --model-card docs/huggingface/model-cards/ghost-gk-sweeper-position-only-v1-model-card.md
python scripts/publish_ghost_outfield.py --artifact-dir silly_kicks/tracking/_ghost_outfield_weights/default \
  --repo-id silly-kicks/ghost-outfield-v1 --model-card docs/huggingface/model-cards/ghost-outfield-v1-model-card.md
python scripts/publish_ghost_outfield.py --artifact-dir silly_kicks/tracking/_ghost_outfield_weights/position_only \
  --repo-id silly-kicks/ghost-outfield-position-only-v1 \
  --model-card docs/huggingface/model-cards/ghost-outfield-position-only-v1-model-card.md
```

- [ ] **Step 12: OWNER GATE — Hub-only card push (5 repos, card only)** through the Task 6b seam. Their weights are untouched.
  - Run the dry run and show the owner the five exact invocations, plus each JSON result (`changed`, `card_sha256`, `hub_sha256_before`). Every result must report `changed: true`. A `false` means the card was not updated in C2: STOP and report. Then **wait for an explicit yes**. Step 11's yes does not cover this step.

```bash
for r in xshot-occurrence-v1 xshot-occurrence-position-only-v1 xcross-attempt-v1 xcross-attempt-position-only-v1 ghost-gk-v1; do
  PYTHONPATH=. python scripts/publish_model_card.py --repo-id silly-kicks/$r --verify-only
done
```

  - On the yes, push (stops at the first failure; a read-back mismatch fails loud inside the seam):

```bash
for r in xshot-occurrence-v1 xshot-occurrence-position-only-v1 xcross-attempt-v1 xcross-attempt-position-only-v1 ghost-gk-v1; do
  PYTHONPATH=. python scripts/publish_model_card.py --repo-id silly-kicks/$r || break
done
```

- [ ] **Step 13: Post-push verification.**
  - `PYTHONPATH=. python scripts/validate_hub_variants.py --out <scratch>/hub_post --require-cards-match --require-mirrors-match-wheel`. It must exit 0 with:
    - `cards_mismatched == []` (all 10 READMEs, including the unchanged `xsuccess-v1`);
    - `mirrors_mismatched == []` (each mirror's Hub `metadata.json` `training_commit` equals its wheel bundle's, computed by the driver);
    - `load_refused == []` (the republished sweeper mirrors load; `--require-mirrors-match-wheel` fails the run otherwise);
    - `all_finite`.
  - The five Hub-only `revision`s change (card commits), but their weight files must not: re-run the Step 10 blob-id snippet and diff it against `hub_pre_blobs.json`. The two must be identical.
  - Any failure: STOP and report. A failed mirror publish leaves the old weights in place (`upload_model_only` uploads one commit). Any repair push is a new OWNER GATE.
  - The post-push `hub_smoke.json` is evidence for the owner's report. It is not committed: C2 already holds the C1 record.
- [ ] **Step 14:** Update memory (release state, cycle outcome, archive pointer, Hub push outcome). Hand the owner the downstream notice.

---

## Self-review

**Spec coverage.**

| Spec section | Plan tasks |
|---|---|
| §0 facts | input pins and checks in T0, T14, T16–T20 |
| §1 scope | re-fits T18/T20/T21; reuse T14; receiver T5/T6/T19; T10 T8; TF-19 T9; DAS T10/T11/T19; §7.3 T18/T19/T21; guard T1–T4/T14/T21; Hub T7/T19; Hub cards T21 Step 1b; card-only seam T6b; post-release Hub pushes T22 Steps 10–14, two OWNER GATES (D9); xcross held-out probe T2b/T18/T20 (D5c); launcher T12b/T17 Step 4b/T18 Step 2b/T20 (D3, D3a, D3b); das-native leftovers T13; race fix T12; release T21/T22 |
| §2 | T8/T9/T10 commit-keyed generations + accounted keys; Global Constraints |
| §3 | T15 / T22 gates (commit, push, PR, merge and tag separate) |
| §4 | Phase B commands; T16 Step 4/5 fail-closed wrapper and counts; `providers_for_slice` in T10 |
| §5 | T1–T5, T14, T21 Step 2 |
| §6 | T8 |
| §7 | T5, T6 (D8 `decide`), T19 Step 1 |
| §8 | T17, T20 Step 2 (incl. the D5c probe acceptance) |
| §9 | T6b (card-only seam, `CARD_SOURCE`, LF staging), T7 (README check, roles 5/4/1), T15 Step 3, T19 Step 4, T20 Step 2, T21 Step 1b, T22 Steps 10–14 (prerequisites, OWNER GATE mirrors, OWNER GATE Hub-only cards, `--require-cards-match` + blob-id check) |
| §10 | T16–T19 (launcher-driven DAS map; doubled reference pass sized in T17 Step 5) |
| §11 | T20 Step 2, T21 Steps 1b/8/11, T22 (incl. §11.9 via T6b tests + T22 Step 13) |
| §12 | T0 Step 1 (decided outcomes), D1 T11, D2 T10, D3 T12b/T17/T18/T20, D4 T1, D5 T2b, D6 T13, D7 T5, D8 T6, D9 T6b/T7/T21/T22, D10 T21 Step 10 |
| Appendix A | each disposition maps to the task named in its row; B r2: CCC-PLAN-20 T9, -21 T10, -22 T10/T17/T21, -23 T10/T20/T21; C1 T7, C2–C4 T10, C5/C6/C8 T11, C7 T20/T21, C9 T17; B r3: CCC-PLAN-24 T10, -25 T10/T11 + Global Constraints, -26 T10, -27 T12b, -28 T10/T12b/T17/T18; A r4: A-PLAN-02 T22 Steps 10–14; B r4: -29/-30/-36 T6, -31 T8, -32/-35 T7 + T22 Step 13, -33/-37 T21, -34 T6b, -38 T12b, -39 T2b, -40 T14 Step 1b, outside-round T5, N3 T15 Step 6; B r5: -41 T7 Step 3b + Global Constraints, -42 T2b, -43 T15/T19/T20/T21, outside-round T2b(c) + T21 Step 1b; B r6: -44 T2b |

**Private locations.** No committed doc names a local path; `$ARCHIVE`, `$CC_OUT` and `$REVIEWS` are supplied by the owner at T0 Step 2 (Global Constraints).

**Placeholder scan.** The `<…>` slots left in the plan are run-time measurements: golden deltas, the chunk table, ADR/CHANGELOG figures, tf24 values, SHAs, worker counts, memory caps. Each names its exact source.

**Type consistency.**
- `corpus_identity(..., all_public=)` and `reproducibility(shipped, providers, *, training_commit=)` are used identically in T2, T4 and T5.
- `bundled_public_arm_pairs` returns `list[list[str]]` everywhere.
- `reduce_t10` and `_token_inputs` match `main`.
- `reduce_parity_artifact(..., direction_col=)` is called identically in the tests and `main`.
- The `summarize_benchmark` keys match `test_das_parity_artifact.py` and T21 Step 7.
- `_path_subprocess(..., expect_native=)` matches its test.
- `numba_compared` / `numba_vs_numpy_max_abs` are `{grain: {"das", "as"}}` in T10, T20 Step 2 and `test_das_parity_artifact.py`.
- The launcher tokens are `{subset}` / `{worker}` (driver) and `{shard_root}` (reduce) in T12b, T17 Step 4b and T18.
- `_map_generation(commit, direction_col=None)` (T10) is what `--print-generation` prints, what the reduce expects, and what `--das-generation` keys the launcher's done-marker on (T12b, T17 Step 4b, T18).
- `CARD_SOURCE` / `card_bytes` / `publish_card_only(api, repo_id, *, root, verify_only)` are defined in T6b and used identically in T7, T15 Step 3 and T22 Step 12; `CARD_SOURCE` keys == `HUB_REGISTRY` keys (T7 test).
- The xcross probe keys (`probe_sample_matches`, `probe_sample_in_training_folds`) are the ones `assemble_studies` writes (`train_xcross_attempt.py:719-727` at `ba9c151`).
- `_foreign_cpu_fraction(*, busy0, busy1, own0, own1, elapsed, ncpu)` in the T11 interface, code and test.
