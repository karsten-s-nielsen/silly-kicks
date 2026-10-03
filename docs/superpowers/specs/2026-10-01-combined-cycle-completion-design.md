# Combined provenance cycle — completion design (F1b corrected re-fits, T10, DAS parity, §7.3, release)

**Date:** 2026-10-02 · **Status:** APPROVED — spec rev 8 / plan rev 7, review rounds in Appendix A; executed from C1.
**Reviews:** two independent review sessions; reports held in the shared external reviews folder (never in the repo). Every round's verdicts and every finding's disposition are in Appendix A. Every finding was re-measured before being accepted.
**Rev 5 vs rev 4 (owner, 2026-10-02):** D9 is revised. The Hub pushes are a post-release step this cycle performs, not an owner action:
- the four mirrors are republished, weights and card together through `publish_model_with_card`;
- the five Hub-only cards are pushed through a NEW card-only seam (`scripts/publish_model_card.py`, C1);
- the Hub smoke driver gains a Hub-README-vs-card check;
- a post-push smoke run confirms every Hub README equals its card (§1, §3, §9, §12);
- each push batch (mirrors; Hub-only cards) is an explicit OWNER GATE: invocations plus `--verify-only` output shown, then an explicit yes before the networked push (A r4 A-SPEC-05).
Measured on 2026-10-02 (anonymous download): 8 of 10 Hub READMEs equal their in-repo card byte-for-byte after CR stripping. The two sweeper cards drift because those repos were never republished after `adafb72`. All 10 Hub READMEs use LF line endings.

**Rev 4 vs rev 3:**
- The owner decided D1–D10 on 2026-10-02 (§12 now records the outcomes). Three change the design:
  - D3: the launcher fix and its owner validations join this cycle (§1, §3, §10);
  - D5(c): xcross probe held out on GS (§4, §8);
  - D8: a combined gate rule (§7).
- §0.10 is corrected: `ghost-gk-v1` is the Hub-only `full` variant, not a mirror. That leaves four mirrors (§9, D9), and their cards are updated.
- §2 scopes commit-keyed generations to the drivers this cycle changes (B r2 N1).
- §10 sizes the second reference pass (B r2 N2).
- The plan fixes B r2's narrow findings (CCC-PLAN-20..23 and the CONSIDERs).
- Local paths are scrubbed: this doc and the plan are committed at C1, and a committed doc must not depend on a local path. Private locations are named, never pathed; the plan carries them as variables the owner supplies (`$ARCHIVE`, `$CC_OUT`, `$REVIEWS`).

**Rev 3 vs rev 2 (main changes):**
- The re-fit acceptance is pre-registered from the archive's own extraction. The "isolates float32" premise is corrected: the `default` re-fits also absorb label changes (§0.1b, §8; CCC-SPEC-01).
- The receiver corpus fact is corrected: all 327 statsbomb matches are manifest-`private` (licensed, ADR-062). The receiver label, the re-fit and the gate are restructured, and the owner re-confirms (§0, §7, D7, D8; CCC-SPEC-02/06/07/08).
- TF-19 gets a driver reduce that honours the allowlist (§0.9, §1; CCC-SPEC-03).
- Completeness is now judged by accounted shards, generations are keyed by commit, and commit consistency is checked on every sharded reduce (§2, §6; CCC-SPEC-04).
- The full Hub population is covered, and the divergence of mirrors is raised as a decision (§0.10, §9, D9; CCC-SPEC-05).
- Parent-cycle omissions are added (§1; CCC-SPEC-09).
- The bundling annotation is emitted by the trainers instead of being hand-edited (§0.11; CCC-SPEC-11).
- Private-id leak paths are closed (§5, §11; CCC-SPEC-12).
- D1/D2 are made fair and complete (CCC-SPEC-13/14).
- Named tests are added (§5; CCC-SPEC-15).
- The release scope is enumerated, plus a version decision (§1, D10; CCC-SPEC-16).
- Review gates are added to §3 (CCC-SPEC-10).
- Two smaller fixes: a race between concurrent cache downloads (§0.12), and a gap in the research-provenance gate (§0.14).
**Supersedes:** the execution half of the combined-provenance-dgx cycle (F1b runbook + launcher-era runs, Sep 27 – Oct 1). The design intent of the das-native spec (`2026-09-26-das-native-design.md` §4/§7.2/§7.3/§10–§12), the F1b ADR-106, the pandas-2 reference-leg spec and the parallel-launcher spec stands; this doc corrects **how** the remaining work is executed and committed.
**Owner decisions recorded (2026-10-01).** These are settled; this revision does not re-litigate them:
- **Receiver:** accept the 30 → 327 widening, gated on held-out non-inferiority. The premise is corrected in §0; D7 asks the owner to re-confirm.
- **HF-only variants:** untouched and smoke-verified.
- **T10:** gets a sharding flag.
- **Commits:** absolute minimum, with provenance, merged non-squash (Phase B amendment 3: C1 is squash-merged as PR-1 before the wave, and every run executes at that merge commit on `main`).
- **Guard:** two layers, against corpus/variant mistakes.

**Owner decisions recorded (2026-10-02, §12):**

| Decision | Outcome |
|---|---|
| D1 | DAS benchmark, gold standard |
| D2 | four-cell parity |
| D3 | launcher fix + validations in this cycle |
| D3a | stratified serial re-run, not the full corpus |
| D3b | study-worker peak RSS measured on the small paired set |
| D4 | the original 17 |
| D5 | (c) public-only training, probe held out on GS |
| D6 | per-engine constant |
| D7 | (a) manifest-derived `restricted` label, re-fit at C1. The owner accepts that weights derived from the licensed statsbomb corpus ship in the public wheel, as the committed 30-match receiver already does. |
| D8 | combined rule, δ = 0.01 |
| D9 | revised: this cycle republishes the four mirrors and pushes the five Hub-only cards after the release, through a card-only seam added in C1; all relevant cards updated |
| D10 | 4.128.0, marked breaking |

---

## 0. Why this exists (verified facts, not narrative)

The first execution of this cycle re-fit models on the DGX for days and shipped nothing. A per-model pre-flight (archived artifact metadata vs the committed bundle metadata, run after the fact) shows:

| Model | Archived (anchor `3ca609f`) vs committed bundle | Status |
|---|---|---|
| ghost_gk ×4 | identical corpus (179 games; n_rows 36000 / 1039502 / 1048834 / 1048834; identical feature-contract probe sha) | **valid** — `default`, `sweeper`, `sweeper_position_only` reused; `position_only` re-fit at C1 (§0.11) |
| ghost_outfield ×2 | identical (179 games / 4170920 rows; identical feature-contract probe sha) | **valid — reuse** |
| gk_completion `default` | `--mode rebundle` reproduced (GS, `max_per_provider` 64, n_rows 3491; served `coef` byte-equal to committed) | **valid — reuse** |
| xshot ×2, xcross ×2 | shipped `sc_extended` (owner corpus, 980 matches; n_rows 8181977 / 4927917) vs committed `public` (the original 17 public matches) | **wrong corpus — re-fit** |
| gk_completion `skillcorner` | 64 owner SkillCorner matches (3324 rows) vs committed 10-match public arm (542 rows) → rebundle aborted | **wrong corpus — re-run** |
| receiver | committed 30 and archived 327 both labelled `corpus_visibility=public`, but the label is derived from the provider NAME (`train_receiver_model.py:529`, the rule ADR-038 deleted). The pining statsbomb manifest lists **327 matches, all `visibility: private`**; the public token lists **0**. ADR-062 calls this "a 30-match **licensed** StatsBomb 360 corpus"; the committed MODEL_CARD calls it "open-data". | **owner decision D7 (label + re-confirm the widening), D8 (gate rule)** |
| T10 (`measure_f1b_feature_delta.py`) | ran at `24ef308` — NOT reachable from `main`; `silly_kicks/tracking/` changed since | **re-run** |

Root cause of every "wrong corpus" row: the owner pining token was sourced for **every** run (`source ~/.pining_owner.env`), so the owner/NDA SkillCorner corpus was visible to every run.
- **xshot/xcross:** the paired path builds three candidate masks (`train_xshot_occurrence.py:431-440`). The fixed-sequence rule `_paired.fixed_sequence_ship` (`scripts/_paired.py:21`, called at `:534`) then shipped `sc_extended`.
- **gk_completion:** the skillcorner arm pulled 64 matches instead of the 10-match public arm (`_gk_completion_weights/skillcorner/MODEL_CARD.md:31`: "Always pass `--max-per-provider 10`").

**Code changes since `3ca609f`** (`3ca609f..HEAD`):
- Every change under `silly_kicks/` other than the new package is docstring or comment only: the Bekkers 2024 → 2025 citations, the gi.py import-boundary note, and one atomic docstring.
- The read-only MCP server `silly_kicks/mcp/` (#264, ~364 lines) is new runtime code, but no training, feature or tracking code imports it.
- So the valid artifacts still describe today's model code.

1. **The public corpus grew after the bundles were trained.**
   - **History.** `scripts/_corpus.py` `PUBLIC_CORPUS["skillcorner"]` went from 10 to 20 ids in `ce0401a`. The commit is dated 2026-09-10; the code comment records the upload as 2026-09-09. The comment also says the bundled xshot/xcross/ghost models "were trained on the ORIGINAL public 17".
   - **The original 17** are listed at `ce0401a^:scripts/_corpus.py`:
     - SkillCorner: `1886347 1899585 1925299 1953632 1996435 2006229 2011166 2013725 2015213 2017461`
     - IDSSE: `DFL-MAT-J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9`
   - **What the public token returns today:**
     - `select_match_ids(["skillcorner","idsse"])` returns **27** matches.
     - `select_match_ids(["skillcorner"], max_per_provider=10)` returns exactly the original 10, because the manifest lists them first.
   - That order is a property of the server, not a contract. So ids are pinned explicitly where the trainer allows it, and asserted where it does not.
1b. **The re-fit row counts are known in advance, and the `default` re-fits are not float32-only.** The archived `3ca609f` runs wrote one extraction shard per match. Counting the original 17 in those shards (this revision; Reviewer B measured the same):
   - **xshot:** 156106 rows / 34649 positives (both variants).
   - **xcross:** 91999 rows / 2849 positives (both variants).

   The committed bundles:
   - **xshot `position_only`** (`0ce2c21`): 156110 / 34650.
   - **xshot `default`** (`6e3a132`): 156110 / 34205.
   - **xcross:** 92001 / 2849.

   The extraction path is unchanged from `3ca609f` to HEAD (`_extract`, `_loader_pining`, `_driver`, `_corpus`, the model modules: no diff). What the re-fits therefore change:
   - **`position_only`:** moves by float32 selection flips only (−4 / −1 xshot; −2 / 0 xcross). That is the `selection_instability` T10 measures.
   - **xshot `default`:** also absorbs the label changes since `6e3a132` (+444 positives, positive rate 0.2191 → 0.2220). These changes include ADR-055 GoalMap, ADR-063 and the ADR-052 seam.
   - **xcross `default`:** the public candidate also predates `0ce2c21` (§0.2).
2. **The committed xcross `default` record is from a paired run.** `_xcross_weights/default/metrics.json`:
   - `providers` = gradientsports + idsse + skillcorner, `n_rows` 1209332;
   - `candidates.paired.why` reads "sc_extended failed the rule; the sequence stops";
   - the TF-19 probe ran on held-out GS matches 10502/10503, `probe_gated_on_held_out: false`.

   Its shipped `public` candidate saw the same 92001 public rows as xcross `position_only` (identical `positive_rate`). The xshot `default` record and both `position_only` records come from single-candidate public runs.
3. **A single-candidate run is one 50-trial study** (`assemble_studies` `else` branch, `train_xshot_occurrence.py:564-575`). The ~15 nested studies exist only on the paired path.
4. **`--mode rebundle` serves the committed coefficients** (`train_gk_completion.py:565-567`). It re-stamps the gate metrics and the provenance, and writes into the **checkout** it runs from (`_WEIGHTS_ROOT`, `:34`).
5. **Dependency graph of the remaining runs.**
   - The §7.3 drivers declare only `GhostGkModel` as a model input (`build_gkdv_arm_values.py:59`, `build_tf19_instrument_responsiveness.py:146`). Their imports confirm it: `gkdv._engine/_arms/_probe` (Reviewer B traced them).
   - `build_layer2_spells.py` passes `model_metadata={}`. `run_signoff_power.py` reads tables only.
   - T10 runs the shared feature **extractors** only.
   - The DAS parity driver loads no weights.

   So the xshot/xcross/gk_completion/receiver re-fits and the ghost `position_only` re-fit feed **no** other artifact of this cycle. The only model any downstream run consumes is the bundled ghost `default`.
6. **The DAS artifact as built cannot carry the das-native §4/§7.2 measurements.**
   - **Reference timing includes overhead.** `_reference_leg_subprocess` (`validate_das_native_parity.py:273-296`) spawns a fresh pandas-2 interpreter per match, so `ms_frame_ref` includes interpreter start, imports and parquet round-trips.
   - **Numba timing includes JIT.** `_time_leg` times numba with no warm-up (`:353-362`).
   - **Contention.** All timings are taken inside the concurrent map.
   - **Numba is recorded for one cell only.** `numba_minus_numpy_das` is written on team rows for DAS only (`:479`). Player rows and the AS output carry no numba column.
   - **Required fields are not emitted** (`reduce_parity`, `:564-604`):
     - GoalMap direction vs the reference's own inference;
     - finite counts on both sides;
     - the count of production frames D-KEY corrupts today.
7. **Never run yet:**
   - DAS corpus parity (§7.2) and the §7.3 downstream;
   - the `test_das_parity_artifact.py` gate (planned, not written);
   - the release commit (CHANGELOG has no `[Unreleased]`; the ADR-108 `COMMIT-2 PLACEHOLDER` at `:73` is unfilled).
8. **Unfinished commit-1 deliverables of the das-native cycle.** Found by checking every `Create:` path in the approved das-native plan against the tree:
   - `tests/tracking/test_das_benchmark.py` (Task 5) does not exist.
   - Task 5 Step 5's chunk-size table is not in ADR-107. ADR-107 states `chunk_size = 32` without it (`:61`).
   - ADR-107 `:67-68` still describes the old frame-count guardrail; the code uses a seconds budget (`_das.py:61`).
   - The pandas-2 reference-leg spec §7 required notes in ADR-108 and `docs/context/`. Only ADR-107 got one.
   - `estimate_das_cost` has a numba-serial constant plus `_PRANGE_EFFICIENCY` but **no numpy constant** (`_das.py:56-61`). Das-native §6.14 asks for one per engine.
9. **The §7.3 populations are recorded, and differ per driver.**
   - **Sign-off inputs** (`docs/research/tf19_signoff_power/README.md:5-6`): the layer-2 spells and the GKDV arm values, both over **64 GS** matches.
   - **TF-19 responsiveness** (`docs/research/tf19_instrument_responsiveness/metrics.json`): `n_matches = 179` = `{gradientsports 64, skillcorner 108, idsse 7}`. Its `reduce_mode` reads "manual-shard-read". The driver's own authoritative reduce runs only WITHOUT `--match-ids-json` (`build_tf19_instrument_responsiveness.py:559`). Run that way, it lists every provider's whole manifest — 980 matches with the owner token today.
   - **The DGX script in the private archive** (`corpus_lists/gkdv_tf19_179.sh`) is a later re-run of 179-match workers in a different checkout, not the committed artifact's run. Its timing (GKDV 11.6 h, TF-19 6.5 h on 4 workers) is still the best wall-clock estimate.
   - **Private archive** (owner-held, off the DGX, never committed; location supplied by the owner at execution): `corpus_lists/` holds the population list, the slices and the tc3 key listing. Byte-exact copies of the float64 `~/tc3-cache` (T10 input) and the float32 `~/tc3-cache-f32` (ghost training input) are in `tc3_cache_f64/` and `tc3_cache_f32/`, each verified per file against the DGX (`SHA256SUMS.dgx`).
10. **Hugging Face Hub population** (anonymous listing of org `silly-kicks`, 2026-10-02): 10 public repos, 9 of them frame-geometry (`xsuccess-v1` is event-only).

| Hub repo | `training_commit` | Relation to the wheel |
|---|---|---|
| `xshot-occurrence-v1`, `…-position-only-v1`, `xcross-attempt-v1`, `…-position-only-v1` | none recorded; `shipped_variant: sc_extended` | **HF-only** variants (the settled "untouched" decision) |
| `ghost-gk-v1` | `4bda048` | **HF-only** `full` variant: its `metrics.json` reads `variant: full`, 179 games, 1039502 samples; the wheel excludes `full` (`pyproject.toml:222`). It is not a mirror, so the settled "untouched" decision applies. |
| `ghost-gk-sweeper-v1`, `ghost-gk-sweeper-position-only-v1` | `adafb72` | already stale vs the bundled sweeper variants (`4bda048`) |
| `ghost-outfield-v1`, `ghost-outfield-position-only-v1` | `b68328a` | **mirrors** of today's bundled ghost_outfield — diverge after C1 |

11. **The ADR-067 `reproducibility` caveat is a bundling-time hand annotation.**
   - **Who requires it:** `test_position_only_reproducibility_caveat` (`tests/tracking/test_position_only_bundled.py:111-122`) requires it on the three `position_only` `metrics.json`.
   - **Who writes it:** no trainer emits it.
   - **Why carrying it forward fails:** the ghost note names `training_commit 4bda048`, which is false on a re-fit. Hand-editing a driver-stamped file adds content that the run did not produce — the rule `tf19_signoff_power/invalidation.json` `_about` states.
12. **Concurrent cache downloads race.** `_download_to_temp` writes every artifact through a fixed `<dest>.partial` name (`_loader_pining.py:160-167`). Two processes fetching the same artifact at the same moment interleave writes into one file. The one-wave plan runs exactly such processes.
13. **Unreleased since 4.127.0.** None of these has a CHANGELOG entry:
   - **F1b** (`9449461`): float32 frame coordinates and `team_id` → `category`.
   - **Native DAS** (#259): `[das]` extra deleted, `**kwargs`/`use_progress_bar` removed, new `ValueError`s, every DAS value moves.
   - **MCP server** (#264).
   - **Detection primitive** `detected_mask` (#260).

   Repo precedent ships breaking changes in a minor with `!` and a BREAKING section (4.80.0 `fix(tracking)!`, `feat(tracking)!` ghost re-fit). The lakehouse pins `silly-kicks<5`.
14. **Every `docs/research/**/*.json` must carry clean provenance.** `tests/scripts/test_artifact_provenance_output.py` requires `run_commit` + `run_tree_dirty is False`, unless the file is listed in `_UNPROVENANCED` with a reason. The existing exemption for `tf19_signoff_power/invalidation.json` must point at a file that exists.

## 1. Scope

**In:**
- **Re-fits:**
  - xshot ×2 and xcross ×2 on the original 17;
  - gk_completion skillcorner (rebundle, original 10);
  - ghost `position_only` (§0.11);
  - receiver per D7.
- **Reuse:** ghost `default`/`sweeper`/`sweeper_position_only`, ghost_outfield ×2, gk_completion `default` (and the receiver per D7).
- **Measurements:**
  - the receiver widening gate as a committed, tested driver (§7);
  - T10 re-run (sharded, §6);
  - DAS corpus parity with full numba coverage, the D-KEY figure, finite counts and GoalMap-vs-inference counts (§0.6, D2);
  - the DAS performance benchmark (D1);
  - §7.3 per das-native `:611-621`: GKDV arm values, layer-2 spells, TF-19 instrument responsiveness via the new allowlist reduce, and sign-off power.
- **Annotations:**
  - `docs/research/tf19_signoff_power/invalidation.json` retired, with its exemption removed;
  - a NEW `docs/research/tf24_stage2_refresh/invalidation.json` sibling annotation (no TF-24 re-run).
- **Code fixes:**
  - the two-layer guard (§5);
  - trainers emitting `reproducibility` (§0.11);
  - the receiver label derived from manifest visibility (D7);
  - the cache-download race fix (§0.12).
- **Hub smoke** over the full Hub population, as a committed driver (§9). It also records whether each Hub README equals its in-repo card.
- **Hub model cards (D9):**
  - the four mirror cards in `docs/huggingface/model-cards/` are updated to the bundled weights;
  - the five Hub-only cards get an "unchanged in 4.128.0, float64-trained" note wherever they describe wheel state;
  - the wheel `MODEL_CARD.md` files for the receiver and both gk_completion variants are updated.
- **Card-only push seam (D9, C1):** `scripts/publish_model_card.py`. It pushes one registered card as the README of an existing Hub repo, then reads it back (§9). Both publish seams stage the card with LF line endings.
- **Post-release Hub pushes (D9):** after the tag is on PyPI, this cycle republishes the four mirrors (weights + card) and pushes the five Hub-only cards. A post-push Hub smoke must then show every README equal to its card.
- **xcross held-out probe (D5c):** a probe-only flag. The TF-19 substitution probe runs on held-out GS 10502/10503, never on training matches, for both xcross variants.
- **Launcher (D3):**
  - **Wiring fixes in C1:**
    - a `{worker}` token;
    - DAS subsets written in the driver's list-matches shape, with done-keys via `join_key`;
    - trainer `--prep-only` / `--list-studies` / `--study-list` so study fan-out is drivable.
  - **Owner validations (launcher plan Tasks 7–9):**
    - drive the full DAS map through the launcher;
    - peak RSS per workload (study worker on the small paired set, D3b);
    - DAS launcher-vs-serial shard identity on a stratified ~30-match serial re-run, timing columns excluded (D3a);
    - xshot/xcross parallel-study == serial-study weights byte-identical, plus shared-mmap `X, y` == per-process `X, y`, on a small paired set.
- **Golden and anchors:** re-capture of the only bundled-output golden.
- **das-native leftovers** (§0.8, D6):
  - `test_das_benchmark.py`;
  - the chunk-size table;
  - ADR-107 corrected;
  - the ADR-108 and `docs/context/` reference-leg notes;
  - the per-engine constant;
  - `estimate_das_cost` constants re-derived.
- **Docs and release:**
  - an ADR-106 amendment: final anchors, the corpus incident, the receiver decision;
  - the multi-period reused-`frame_id` smoke (pandas-2 spec §8);
  - the complete release (§0.13 content, D10 number);
  - the lakehouse downstream notice handed to the owner (das-native §11.2).

**Out (explicit):**
- re-fitting or re-uploading any Hub-only artifact: the five Hub-only repos get a card push only, and their weights stay untouched;
- the owner-held ghost-GK disclosure remediation actions (privacy request, token downscope, historical wheels, audit bundle). A parameters-only upload and a card push are not part of that hold;
- widening the xshot/xcross/gk_completion public corpus 17 → 27 (D4);
- TF-65 (only its TODO row, already merged);
- lakehouse re-materialisation (consumer-side);
- any metric or model design change.

## 2. Provenance model

Every committed artifact names the exact code and weights that produced it, on a clean tree, at a commit **reachable from `main`** (merge non-squash — standing practice; after Phase B amendment 3 the run commit is itself the PR-1 merge commit on `main`). Per artifact:
- `run_commit` / `training_commit` is a SHA reachable from `main`;
- `run_tree_dirty == false`;
- for sharded runs, **completeness is by accounted keys**: every listed item has a shard or an exclusion marker in the single shard generation, and there are no unrecorded failures. Summed `n_attempted` is informative only, because `_driver` counts resumed skips as not attempted (`_driver.py:269`, `:456`);
- **Drivers this cycle changes (T10, DAS parity, TF-19):** the shard generation is keyed on the run commit, so shards are attributable to a commit even when a worker died before writing its manifest. Every manifest present names the same commit (`commits_seen == {C1}`).
- **GKDV arm values and layer-2 spells** are not changed here. They keep their existing provenance: per-worker manifests carry `run_commit` / `run_tree_dirty`, and `run_signoff_power.py:114-125` refuses a dirty or mixed-commit input table. The plan also checks that both tables' manifests name C1.

Anchors used in this cycle — exactly two:
- `3ca609f` — the reused artifacts (reachable via the #261 non-squash merge).
- **C1** — commit 1 on the cycle branch. Every new run of this cycle executes at C1:
  - the re-fits;
  - the receiver gate;
  - the Hub smoke;
  - T10;
  - DAS parity and the benchmark;
  - all §7.3 drivers.

The only work done on the pre-C1 tree is the velocity-golden re-capture. It is test-fixture content committed in C1, not a provenance artifact.

## 3. Commit structure — two commits is the minimum

**Commit 1 (C1) — "cycle code + reused F1b weights"** (no version bump):
- **Code.** All of the cycle's code, so every run anchors to it:
  - **Guard:** the G1 launch preflight, plus corpus-identity recording for all-public corpora, in the public-arm trainers; and the G2 registry gate in its C1 state (§5).
  - **Trainers:** emit the `reproducibility` caveat from the corpus taxonomy (§0.11). The receiver trainer derives its label from manifest visibility and accepts an allowlist (D7, §7).
  - **Receiver gate driver** (§7) and **Hub smoke driver** (§9), both committed.
  - **Card-only push seam** `scripts/publish_model_card.py`, and LF card staging in `scripts/_hub_publish.py` (§9, D9).
  - **T10:** sharding flags with commit-keyed generations, plus parity and resume tests (§6).
  - **TF-19:** the allowlist-honouring reduce (CCC-SPEC-03).
  - **DAS parity additions:** full numba coverage in an extended shard schema, the golden-bound counts, the D-KEY / finite / direction counts, completeness and commit checks, `providers_for_slice`, the `--benchmark` mode, in-process reference timing, and the old-path leg (D1, D2).
  - **Loader:** the cache-download race fix (§0.12).
  - **das-native leftovers:** `test_das_benchmark.py`, the ADR-107 chunk table and corrected guardrail sentence, the ADR-108 and `docs/context` reference-leg notes, and the per-engine cost constant (D6).
  - `BUNDLED_PUBLIC_ARM` in `scripts/_corpus.py`.
  - **xcross probe-only flag (D5c):** `--probe-match-ids-json`. The probe matches are extracted in a separate pass and never enter training rows.
  - **Launcher wiring (D3):** `{worker}` token; DAS list-matches subsets; trainer `--prep-only` / `--list-studies` / `--study-list`.
  - `.gitattributes`: a `diff` attribute so bundle `metrics.json` diffs stay readable.
- **Reused weights (anchor `3ca609f`):** ghost_gk `default`/`sweeper`/`sweeper_position_only`, ghost_outfield ×2, gk_completion `default`. Copied byte-for-byte from the local archive.
- **Golden re-capture.** `tests/tracking/data/ghost_velocity_path_baseline.npz` is re-captured per its own convention ("revisited, not absorbed"): its docstring records the measured move; it is the only committed golden pinning bundled-model output. `tests/tracking/test_position_only_bundled.py` gains a new history constant for `3ca609f`; the older constants stay.
- **Docs:** this spec + its plan; the ADR-106 amendment.

**One wave at C1** (§10). Every new run executes concurrently, except two:
- the receiver gate follows the receiver re-fit (D7a);
- the DAS benchmark runs alone at the end.

**Commit 2 (C2) — "F1b re-fits + provenance artifacts + release"**:
- **Re-fit weights (anchor C1):** xshot ×2, xcross ×2, gk_completion `skillcorner`, ghost `position_only`, and the receiver per D7/D8.
- **G2 tightened:** to the exact corpus ids and per-dir anchors (§5); new history constants in `test_position_only_bundled.py`.
- **Artifacts:**
  - `docs/research/das_native_parity/` (`metrics.json` + `performance.json` + a README labelling the map timings as contended and not for ratios);
  - `docs/research/f1b_float32/`: T10 `metrics.json` only — **no per-match table**, it carries owner-tier ids — plus `receiver_gate.json`, `hub_smoke.json` and `findings.md` (`findings.md` includes the launcher validation results, D3);
  - the four mirror Hub cards, the Hub-only card notes and the wheel model cards (D9);
  - the §7.3 outputs;
  - `tf19_signoff_power/invalidation.json` retired, with its `_UNPROVENANCED` entry removed;
  - the new `tf24_stage2_refresh/invalidation.json` with an `_UNPROVENANCED` entry. It is an annotation like the tf19 one; the owner approves this exemption with this spec.
- **Release:**
  - `tests/tracking/test_das_parity_artifact.py`;
  - the ADR-108 placeholder filled;
  - `estimate_das_cost` constants re-derived;
  - the CHANGELOG release entry with the full content of §0.13 plus this cycle (D10 number), and BREAKING/Hyrum blocks;
  - `TODO.md` groomed;
  - `silly_kicks/_version.py` bumped.

**Why not one commit.** Two independent reasons:
- `build_gkdv_arm_values.py` and `build_tf19_instrument_responsiveness.py` `load()` the **bundled** `GhostGkModel` (§0.5). The reused ghost `default` must therefore be committed before they run, and their outputs committed after.
- The guard and the fixed trainers must be inside the re-fits' `run_commit`.

**Why not three commits.** The re-fits feed no other artifact (§0.5). Every run shares the single anchor C1, and no integration commit is needed (§11.7).

**Review gates before each commit.** Each of C1 and C2 is preceded by:
1. full CI-faithful verification;
2. `/final-review`;
3. an external implementation review;
4. the owner's explicit approval for that specific commit.

C1 is a coherent, fully-tested state: the not-yet-re-fit bundles are the ones `main` already serves on float32 frames. Neither commit is a micro-commit.

Process rules:
- One feature branch, no worktree (Phase B amendment 3: a second branch for C2, because PR-1 merges before the wave).
- Merge non-squash (Phase B amendment 3: PR-1 squash; the run commit is its merge commit on `main`).
- Tag only after post-merge CI is green.
- **After the release** (tag pushed, PyPI live, wheel bundles verified): the D9 Hub pushes (§9). They are not a commit, but they publish to a public org and cannot be taken back, so each push batch is an **explicit OWNER GATE**, like the commit, push, PR, merge and tag gates. The session shows the exact invocations and their `--verify-only` output, then waits for an explicit yes before any networked push. A `--verify-only` run is a dry run, not the approval (A r4 A-SPEC-05).
- The owner publishes.
- **No commit, push, PR, merge or tag without the owner's explicit approval for that specific action. Each is a separate gate.**

## 4. Per-run corpus and token (the correction)

| Run | Pining token | Corpus definition | Anchor |
|---|---|---|---|
| xshot faithful + position_only | **public** | `--providers idsse,skillcorner --match-ids-json <original 17> --expect-variant public` → single `public` candidate | C1 |
| xcross faithful + position_only | **owner** — the held-out probe needs GS (D5c); training stays public by construction: the `--match-ids-json <original 17>` allowlist plus G1 `--expect-variant public`, which refuses unless every requested TRAINING match is public | same training corpus as xshot; `--probe-providers gradientsports --probe-match-ids-json {"gradientsports": ["10502","10503"]} --probe-comparison-providers ""` | C1 |
| gk_completion skillcorner | **public** | `--variant skillcorner --providers skillcorner --max-per-provider 10 --mode rebundle`; G1 refuses a non-public request; recorded `requested_match_ids` must equal the original 10 | C1 |
| ghost_gk `position_only` | none — LOCAL float32 tc3 (`--data-dir ~/tc3-cache-f32`, the archived runs' input; byte-exact local copy exists) | the same 179 games | C1 |
| receiver (D7a) | **owner** (statsbomb is manifest-private) | `--provider statsbomb --pool-provider gradientsports --feature-set public` (the committed run's configuration) | C1 |
| receiver gate | owner (statsbomb manifest order for Step 1; ids never written) | the receiver run's rows (D7a) or the archived rows (D7b) | C1 |
| Hub smoke | none (anonymous Hub download) | the org's frame-geometry repos | C1 |
| DAS corpus parity | **owner** (das-native §7.2; aggregate artifact) | full pining corpus, driven by `_parallel_launch.py --mode das`: list-matches subsets, `--worker-tag {worker}`, `providers_for_slice` (D3) | C1 |
| DAS launcher-vs-serial check (D3a) | owner | ~30 matches stratified by provider (including the multi-period one), run serially into a separate shard root; shards compared to the launcher's, timing columns excluded | C1 |
| xshot/xcross study fan-out check (D3, D3b) | owner | small paired set (a few GS + public + owner SkillCorner matches; low `--n-trials`): serial vs `--prep-only` + launcher `--mode f1b` + `--assemble`; peak RSS of a study worker recorded | C1 |
| DAS benchmark | owner | fixed seeded sample, 3 per velocity-bearing provider (D1) | C1 |
| T10 | none — LOCAL float64 tc3 (`--data-dir ~/tc3-cache`) | the same 179-game corpus as the archived T10; restored from the verified local copy if the DGX lost it | C1 |
| GKDV arm values (`--arm das`) | owner | `--providers gradientsports`, allowlist = the 64 GS ids (the sign-off population, §0.9) | C1 |
| layer-2 spells | owner | same 64 GS ids | C1 |
| TF-19 instrument responsiveness | owner | `--providers gradientsports,skillcorner,idsse`; workers on slices of `corpus179.json`, then the new `--reduce-only` with the full 179 as population | C1 |
| sign-off power | none (reads the two tables) | `--spells`/`--arm-values` from the two runs above, `--seed 0`, `--lock-commit 6b242cf` (the registered anchors in `gkdv/_validate.py` are unchanged since `6b242cf`; recorded in findings) | C1 |

Rules:
- The token is chosen **per run**, never sourced globally.
- A run whose committed bundle is public-tier uses the public token. The one exception is the xcross runs (D5c), which need GS for the probe only: their training corpus stays public through the allowlist plus G1.
- An owner-token wrapper **refuses an empty or unreadable token**: the loader would otherwise fall back silently to the public token (`_loader_pining.py:73`). Owner visibility is verified by manifest counts before launch.
- Every `--out` / `--output-dir` / shard root lies OUTSIDE every checkout.
- `--providers` lists only providers that have a key in the allowlist (`_wanted_for_provider` falls back to the whole manifest for a provider absent from it, `_loader_pining.py:210`). Partitioned drivers use `_partition.providers_for_slice`.

## 5. Guard (two layers)

**G1 — launch preflight + corpus identity (shift-left), C1 code.**
- **xshot/xcross** gain `--expect-variant {public,sc_extended,full}` (default off; existing behaviour unchanged).
  - With `public`, the trainer lists the **requested** corpus (`select_match_ids` + `match_visibility`) and refuses **before extraction** unless every requested match is public.
  - At ship time it refuses to write an artifact whose `shipped_variant` differs from the expectation. The ship decision moves before the study on the single-candidate path, so a refusal costs no fit.
  - The expectation rides the persisted study config, so `--assemble` workers enforce it too.
- **gk_completion `skillcorner`** refuses **before extraction** when the requested SkillCorner corpus is not all-public. Today nothing refuses it.
- **Corpus identity is recorded in the artifact only when the corpus is all-public:**
  - xshot/xcross `metrics.json` gain `corpus_match_ids` (sorted `[provider, match_id]` pairs actually extracted);
  - gk_completion `skillcorner` gains `requested_match_ids`;
  - for any corpus with a non-public match, a SHA-256 digest of the sorted pairs is recorded instead (`corpus_match_ids_sha256`). Hub publishes copy `metrics.json`, so owner/NDA ids must never appear in one.
- **Named tests** (each band from both sides):
  - pre-extraction refusal when a requested match is private;
  - pass when all are public;
  - ship-time refusal on a mismatch, and pass on a match;
  - the expectation surviving the persisted config (an `--assemble` call refuses);
  - gk_completion refusal before extraction;
  - `corpus_match_ids` content on a public corpus;
  - the digest (and no ids) on a restricted corpus;
  - `requested_match_ids` content.

**G2 — CI registry gate** `tests/test_bundled_weights_corpus_policy.py`.
- **Population.** It derives the population of variant dirs by globbing `silly_kicks/tracking/_*_weights/*/`, skipping the wheel-excluded `full` dirs (`pyproject.toml:222`). It asserts the population **exactly** against a declared policy registry (ADR-056: an unknown dir fails, a missing dir fails, `_UNDERIVABLE` is asserted empty).
- **Values.** Per dir, it asserts committed metadata against the declared corpus policy. Every value is copied from committed or archived metadata; none is invented.
- **C1 state:**
  - xshot/xcross `default` + `position_only`: `metadata.json` `shipped_variant == "public"`, `provider_list == ["idsse","skillcorner"]`;
  - ghost_gk ×4: `corpus_provenance.providers == ["gradientsports","skillcorner","sportec"]`, `n_games == 179`;
  - ghost_outfield ×2: `n_games == 179`, `variant == <dir>`;
  - gk_completion `default`: `providers == ["gradientsports"]`, `artifact_label == "full"` (owner decision 2026-08-02; the reused rebundle carries it);
  - gk_completion `skillcorner`: `variant == "skillcorner"`, `n_matches == 10`;
  - receiver: `providers_trained == ["statsbomb"]`, and `corpus_visibility` as committed today;
  - every `metrics.json` that records `run_tree_dirty`: `false`.
- **C2 tightening** (lands with the re-fit weights):
  - xshot/xcross: `corpus_match_ids` == `BUNDLED_PUBLIC_ARM` exactly, and `reproducibility == "public"`;
  - gk_completion `skillcorner`: `artifact_label == "public"`, `all_public is True`, `requested_match_ids` == the original 10;
  - receiver: `corpus_visibility` per D7;
  - ghost `position_only`: `reproducibility == "restricted"` with a note naming its own `training_commit`;
  - every frame-geometry dir: its training/run commit equals its declared anchor (`3ca609f` or C1). "Every frame-geometry bundle was re-fit on float32 frames" thereby becomes a CI fact.
- **Anti-rot:** a deliberately wrong metadata dict (`shipped_variant="sc_extended"`) must be caught.
- **Placement:** the test lives beside `tests/test_bundled_weights_classification.py` and reuses its discovery helper, so the two registries cannot drift apart.

## 6. T10 sharding (provenance-safe)

`measure_f1b_feature_delta.py` is a single-process `for_each` pass (~17.6 h serial) over local parquets. Add, mirroring the DAS driver's split:
- `--list-match-keys`;
- `--match-keys-json` (a worker's subset);
- `--shards-only` + `--worker-tag`. A worker writes its `manifest_<tag>.json` carrying `run_commit` / `run_tree_dirty`; the tag is required.
- `--reduce-only`. It:
  - reconciles the single shard generation;
  - requires every listed key to have a shard or an exclusion marker;
  - requires the generation to be the one this commit's token inputs produce;
  - requires every manifest present to name this commit;
  - writes `metrics.json` in the **same schema** as the serial run, plus an additive `n_accounted`.

The `for_each` token inputs gain the run commit and a corpus identity (a digest of the full sorted key list). Effects:
- shards are attributable even when a worker dies before writing its manifest;
- a worker resumed at another commit lands in another generation, which the reduce refuses.

The serial path is otherwise unchanged.

Required tests:
- a two-worker sharded run reduces to the **same** model block and table as the serial run;
- a worker killed mid-pass and relaunched reduces identically;
- a worker killed after its last shard but before its manifest still reduces;
- an unfinished corpus is refused;
- a mixed-commit corpus is refused;
- an unknown key is refused;
- `--shards-only` without a tag is refused. This test must FAIL before the change; argparse alone must not satisfy it.

The committed T10 artifact is `metrics.json` only. The per-match table carries 162 owner-tier match keys and stays in the private archive.

## 7. Receiver — accept 327, gated (pre-registered; owner re-confirmation D7, rule D8)

The 327-match model (CV top-1 0.4966) and the committed 30-match model (0.5098) were each scored on their own corpus, so the numbers are not comparable. The gate is a committed driver, `scripts/validate_receiver_widening.py`:
- tested and linted;
- `require_clean_tree`;
- registered in `ARTIFACT_DRIVERS`;
- writes `docs/research/f1b_float32/receiver_gate.json` with provenance;
- never records a match id.

It runs at C1 on the rows of the 327-match receiver run: the C1 re-fit under D7a, or the archived `3ca609f` rows under D7b.

**Step 1 — identify the committed model's 30 training matches.**
- **Candidate set.** At `08347cd` the trainer used the whole statsbomb manifest of the time (`load_statsbomb_matches`, uncapped). The candidate set is the first 30 ids of today's manifest (owner token; read in-process, never written out).
- **Pre-registered success criteria.** Refit `ReceiverModel("public")` on exactly those 30 games' rows. Identification succeeds iff all three hold:
  - all 30 ids are present;
  - the refit's `top1_cv` is within ±0.005 of 0.5098;
  - every refit parameter is within relative 1e-2 of the committed `model.json`.
- **Negative control** (AGENTS.md "every band tested from both sides"): 20 random 30-match subsets (seed 0) of the remaining games must ALL fail the same criteria. Otherwise the identification is declared non-discriminating.

**Step 2 — the gate.**
- **Split.** On the 327-match rows (namespaced as the trainer does), use the trainer's own deterministic split: `GroupKFold(5).split(rows, groups=rows["game_id"])`, `train_receiver_model.py:195-208`.
- **New vs old.** Per fold, **new** = a fresh fit on the fold's train split (exactly `cv_top1`'s procedure); **old** = the committed bundled model, unchanged.
- **Exclusions.** If Step 1 succeeded, the 30 identified games are removed from every **test** split. Otherwise they stay, the comparison favours the old model (conservative), and that is recorded.
- **Scoring.** Both models are scored per pass with the trainer's own argmax rule (`_top1_accuracy`, `:182-192`) on identical test passes.
- **Statistic.** Pass-weighted top-1 difference, new − old, with a match-level bootstrap (resample test matches, seed 0, 2000 resamples).
- **Decision rule (D8, owner-approved):** ship iff **both** hold:
  - the point estimate new − old ≥ 0 — the trainer's own data-earns-inclusion rule, `pooling_gate`, `:211-238`;
  - the 95 % bootstrap lower bound > −0.01.

  Rationale for δ = 0.01 (one percentage point of top-1): it is below the committed model's fold-to-fold SD (≈ 0.013 over its five folds) and close to the margin that rejected the GS pool (−0.009). The point rule keeps the repo's precedent; the bound stops a noisy pass.

**Pre-registered failure path.** If the gate fails and Step 1 succeeded:
- the receiver is re-fit at C1 on float32 frames on exactly the 30 identified matches (the trainer gains `--match-ids-json` in C1);
- acceptance: `n_matches == 30`, `corpus_visibility` per D7, `top1_cv` reported against 0.5098;
- it ships as the float32 re-fit of the committed corpus, so every frame-geometry bundle is still re-fit, as F1b §4.3 requires.

If Step 1 also failed, STOP and report to the owner. Rationale for everything in this section goes into `findings.md`.

## 8. Corrected re-fits — acceptance per run

- **xshot / xcross (public, the original 17)** — pre-registered from the archive's `3ca609f` extraction (§0.1b):
  - `shipped_variant == "public"`;
  - `corpus_match_ids` == the original 17;
  - xshot **156106 rows / 34649 positives**, xcross **91999 / 2849**, both variants, exactly. The C1 extraction path equals `3ca609f`'s; a different count STOPS the run and is reported;
  - the trainer's acceptance gates pass;
  - chirality + feature contract verify on `load()`;
  - **xcross (D5c):**
    - `probe_sample_matches` == GS 10502/10503;
    - `probe_sample_in_training_folds` all `false`;
    - no "NOT held-out" note in the log.

    The two GS probe ids are already committed in the current xcross `default` record and in driver help strings, so recording them adds no new exposure.
- **Metric movement disposition.**
  - `position_only` PR-AUC / Brier are expected within ±2·`pr_auc_std` of the committed candidate. A larger move is reported to the owner before C2.
  - The `default` re-fits absorb the label changes since `6e3a132` (§0.1b). Their PR-AUC/Brier deltas are recorded and reported, never gated against the old labels. The CHANGELOG and `findings.md` say so.
- **gk_completion skillcorner:**
  - G1 passes;
  - `requested_match_ids` == the original 10;
  - `n_rows == 542`, `n_matches == 10`;
  - `--mode rebundle` reproduces the committed weights.
  - **Pre-registered fallback if it aborts:** report to the owner first, then `--mode retrain --feature-space moved --probe-old <the same 10 matches extracted at 4b15365, row-aligned via --cache-features>`.
- **ghost `position_only` (C1):**
  - corpus provenance identical to the archived `3ca609f` run (179 games, 1039502 rows);
  - its `metrics.json` emits `reproducibility: restricted` with a generated note;
  - the archived re-fit's weights are the comparison: identical code and data predict byte-identical weights; any difference is reported.
- **receiver:** §7.
- **Before each full run:** a one-match smoke that prints `n_matches`, `providers`, the requested ids, `shipped_variant`/arm, and peak RSS. A mismatch stops the run.

## 9. Hub — verified at C1, pushed after the release

The Hub is versioned separately from the wheel. Nothing is pushed before the release. After the release, this cycle performs the D9 pushes. The owner-held disclosure remediation actions are not touched.

**Hub smoke driver** `scripts/validate_hub_variants.py` (committed, C1):
- derives the population from the anonymous org listing and asserts it exactly against a registry: 9 frame-geometry repos (5 `hf_only`, 4 `mirror`), plus `xsuccess-v1` classified as event-only;
- downloads each repo anonymously and loads it fail-closed (chirality + feature contract);
- records a load the library itself refuses (its fail-closed integrity checks) per repo under `load_refused`, instead of aborting the run, so one refused repo cannot hide every other repo's result. A refused `hf_only` repo always fails the run (no planned push replaces it); a refused mirror fails only under `--require-mirrors-match-wheel`, the post-push check, because the republish replaces it. Any other exception still propagates (owner-approved 2026-10-02, implementation review B M-1);
- scores the canonical float32 frame (`tests/test_bundled_models_load_on_float32_commit1.py`) to finite values;
- records per repo the resolved revision, the `training_commit` and the relation to the wheel (§0.10);
- records per repo whether the Hub `README.md` equals the registered in-repo card. `CARD_SOURCE` maps each of the 10 repos to its card: the 9 files in `docs/huggingface/model-cards/`, and `silly_kicks/xsuccess/weights/MODEL_CARD.md` for `xsuccess-v1`. Both sides are LF-normalized before the comparison. With `--require-cards-match`, any mismatch fails the run;
- records each mirror's Hub `metadata.json` `training_commit` against its wheel bundle's (`MIRROR_BUNDLE`). With `--require-mirrors-match-wheel`, any mismatch fails the run;
- downloads anonymously in fact: the implicit HF login is disabled for the run, since `from_hub` passes no token argument (B r4 CCC-PLAN-35).

Output: `docs/research/f1b_float32/hub_smoke.json`, with provenance, run at C1. At C1 the card comparison is recorded, not gated: the cards change in C2. The expected C1 state is the measured one: 8 READMEs match, the two sweeper repos differ. The mirror comparison is likewise recorded, not gated, at C1. All four mirrors are expected to differ: after C1 the wheel holds the reused `3ca609f` weights, while the Hub serves `adafb72` (sweepers) and `b68328a` (outfield) until the post-release republish (B r5 CCC-PLAN-43). Likewise `load_refused` is pre-registered as exactly the two sweeper mirrors: their Hub weights (`adafb72`, pre-ADR-089) fail the chirality check (measured 2026-10-02). Any other refused repo stops the run for a report. `findings.md` states all three expected sets, so the committed C1 `hub_smoke.json` does not read as a defect.

**Card-only seam** `scripts/publish_model_card.py` (C1). The repo has a card-required seam for model publishes (`publish_model_with_card`, ADR-088). Card-only pushes were ad-hoc README uploads (`1b56ad8`, `aae6fdb`), with no guard and no read-back. The new seam closes that gap, as an ADR-088 amendment:
- the repo must be registered in `HUB_REGISTRY`, and the card is taken from `CARD_SOURCE` (never a free path), so a card cannot reach the wrong repo;
- the card must exist, and is uploaded with LF line endings (a Windows checkout with `core.autocrlf=true` holds CRLF). YAML frontmatter is not required: the live `xsuccess-v1` card has none (measured 2026-10-02);
- the repo must already exist: a card-only push never creates a repo;
- it uploads `README.md` only; an unchanged card is not re-uploaded;
- it reads the README back and fails unless the bytes are identical;
- `--verify-only` reports changed / unchanged without uploading.

`publish_model_with_card` stages its card through the same LF normalizer, so both seams publish identical bytes for the same card.

**Cards (D9).** C2 updates:
- the four mirror cards in `docs/huggingface/model-cards/`: training commit, corpus, metrics and a float32-frame note, taken from the bundled artifacts;
- the five Hub-only cards: a short "unchanged in 4.128.0; trained on float64 frames" note wherever they describe wheel state;
- the wheel `MODEL_CARD.md` for the receiver (rewritten; label per D7) and both gk_completion variants (provenance lines).

A test ties every card to the bundle it describes (§11).

**Post-release pushes (D9), performed by this cycle after the tag is on PyPI.** D9 authorizes the session to run them. It does not pre-approve any single push: each batch is an explicit OWNER GATE (§3, A r4 A-SPEC-05).
1. Confirm the HF identity has write access to the `silly-kicks` org.
2. **OWNER GATE — mirrors.** Run all four mirror `publish_*.py` invocations with `--verify-only` from the release-tag checkout (weights + card, `publish_model_with_card`). Show the owner the exact four invocations and their dry-run output, and wait for an explicit yes. Only then republish. A mirror card is pushed only with its weights: a card describing weights the repo does not serve would be false.
3. **OWNER GATE — Hub-only cards.** Run `publish_model_card.py --verify-only` for the five Hub-only repos. Show the owner the exact five invocations and each result (`changed`, `card_sha256`, `hub_sha256_before`), and wait for an explicit yes. Only then push.
4. Run the Hub smoke with `--require-cards-match --require-mirrors-match-wheel`. All 10 READMEs must equal their cards, no repo may be refused (`load_refused` empty), the four mirrors' Hub `training_commit` must equal the wheel bundle's (the driver computes and gates this), and every score must be finite.

Approval of one gate never extends to the other, or to a re-push after a failure.

The CHANGELOG states:
- the five HF-only variants (xshot ×2, xcross ×2 `sc_extended`, ghost `full`) keep their float64-trained weights; only their cards are refreshed;
- the four mirrors (the two ghost sweeper repos, stale since `adafb72`; the two outfield repos) are republished from the release to match the wheel.

## 10. Execution and parallelism

DGX: 20 cores, 119 GiB, aarch64. **Nothing on the DGX is assumed to survive** (owner, 2026-10-01). Every prerequisite has a check and a rebuild step:
- checkouts and venvs;
- the `py2ref` pandas-2 reference env;
- the 4.127.0 old-path env and a pandas-2 env of C1 for the new path (D1);
- the pining cache;
- both tc3 corpora — restored from the verified local copies (§0.9) if the DGX lost them.

**Checkouts and outputs.** Two checkouts of C1:
- `run`, shared by every read-only workload;
- `gkc`, because gk_completion writes into its own tree (§0.4).

Each checkout has its own venv. All outputs go outside both trees. User-scope memory caps need lingering (`loginctl show-user`); the plan checks it.

**The wave, at C1.** Concurrent:
- xshot ×2 and xcross ×2 (single-study processes);
- gk_completion skillcorner;
- ghost `position_only`;
- receiver (D7a);
- DAS parity map, driven by the launcher (D3);
- T10 map;
- GKDV, spells and TF-19 workers;
- the D3 validations: the stratified serial DAS re-run, and the small paired study fan-out check.

Then, in order:
1. the receiver gate (and its fallback if pre-registered);
2. the reduces — DAS `--reduce-only`, T10 `--reduce-only`, the GKDV/spells final pass, TF-19 `--reduce-only`;
3. `run_signoff_power.py`;
4. the Hub smoke (any time);
5. the DAS benchmark **alone** on the quiet box.

**Sizing and safety.**
- A one-match smoke per workload measures peak RSS. One of the DAS smokes must be a multi-period match whose `frame_id` restarts per period (pandas-2 spec §8). That match is found directly from the frames (per-period `frame_id` ranges overlap), never from the driver's own D-KEY count.
- The DAS map now runs the reference library twice per match: once with the shared direction, once with the library's own inferred direction (§0.6). The reference leg dominates the map, so sizing assumes about twice the per-match DAS time of the first run's estimate (≈ 28 CPU-h instead of ≈ 14).
- Workers are sized against one global budget (119 GiB minus 20 GiB headroom) and 20 cores.
- Every worker runs under a per-process memory cap.
- BLAS/OpenMP threads are pinned to 1 per worker.
- The download race is fixed in code (§0.12), so concurrent workers may share one cache.

**Monitoring.** Use `ps -eo pid,etime,rss,args | grep python | grep -v grep` or a completion sentinel — never `pgrep -f <pattern>`.

What ran serially last time, and what changes:

| Last time | Now |
|---|---|
| xshot/xcross ran the 15-study paired path over a 980-match extraction | one study each over 17 matches |
| ghost variants ran one at a time | 3 reused, 1 re-fit concurrent |
| T10 ran in a single process, ~17.6 h | sharded |
| DAS, layer-2 and §7.3 never started, though independent | in the same wave |
| re-fits and downstream runs were sequenced, though independent | one wave |

## 11. Acceptance (cycle done when all hold)

1. **Bundles and CI.** All 13 bundle dirs pass G2 (C1 state at C1, tightened state at C2), plus the golden / chirality / float32-load / position-only tests. Full CI is green on C1 and on C2.
2. **Provenance.** Every committed artifact has a `run_commit` reachable from `main` after merge and `run_tree_dirty == false`. Sharded artifacts satisfy the §2 completeness and commit rules. `test_artifact_provenance_output.py` passes with only owner-approved exemptions.
3. **DAS gate.** `test_das_parity_artifact.py` passes:
   - parity bounds over all four cells (D2), each cell non-vacuous (a non-zero count of finite numba comparisons);
   - zero finite-mask mismatches outside the reason classes;
   - the §4.2 speed targets from `performance.json` (D1);
   - the D-KEY, finite and direction counts present;
   - provenance.
4. **Recorded results.**
   - The receiver gate passed, or the pre-registered failure path completed.
   - The Hub smoke passed.
   - The re-fit acceptance (§8) is recorded.
   - The D3 launcher validations passed and are recorded in `findings.md`:
     - stratified shards identical, timing columns excluded;
     - parallel-study == serial-study weights byte-identical;
     - mmap `X, y` == per-process;
     - peak RSS per workload;
     - no OOM under load;
     - a speedup vs serial.
5. **DAS figures quoted.** The ADR-108 placeholder and the CHANGELOG Hyrum block carry the DAS-shift figures from the committed parity artifact. The `estimate_das_cost` constants are re-derived from `performance.json`.
6. **Version.** The version (D10) appears only in C2. The built wheel is verified to bundle every changed weights dir byte-exact.
7. **Commits.** Exactly two commits on the branch. Main is integrated only by the non-squash PR merge, never by a merge into the branch. (Phase B amendment 3: C1 is rebased onto `main` and squash-merged as PR-1; C2 is PR-2.) NEXT-FREE numbers are derived read-only from `origin/main`.
8. **Private data.** No committed file adds an owner-tier match id. The only exception is the two GS probe ids (10502/10503): they are already committed in the current xcross `default` record and in driver help strings. A test guards `docs/research/f1b_float32/` and `das_native_parity/` against id-bearing columns.
9. **Cards (D9).**
   - In C2, `tests/test_model_cards_match_bundles.py` passes: every mirror card states its bundle's `training_commit` and metrics, every Hub-only card names its bundled sibling's commit, and each wheel card states its `run_commit`.
   - The card-only seam's tests pass with a fake Hub API: unregistered repo refused, missing repo refused before any upload, LF upload, unchanged card not re-uploaded, read-back mismatch fails, `--verify-only` never uploads.
   - Both post-push gates are tested offline both ways (pass when all match; fail on one stale README; fail on one stale mirror). The load-refusal handling is tested both ways too: a refused mirror is recorded at C1 and fails under `--require-mirrors-match-wheel`; a refused Hub-only repo always fails; a non-refusal exception propagates.
   - After the release, the post-push Hub smoke (`--require-cards-match --require-mirrors-match-wheel`) shows all 10 READMEs equal to their cards, no repo refused, and the four mirrors' `training_commit` equal to the wheel's.

## 12. Owner decisions (decided 2026-10-02; options kept for the record)

Outcomes: **D1** recommended; **D2** recommended; **D3** both parts in this cycle, with **D3a** the stratified ~30-match serial check and **D3b** study-worker RSS on the small paired set; **D4** (a); **D5** (c); **D6** recommended; **D7** (a), including acceptance of licensed-derived weights in the wheel; **D8** combined rule, δ = 0.01 (§7); **D9** revised by the owner on 2026-10-02: this cycle performs the post-release Hub pushes (four mirror republishes and five Hub-only card pushes) through a card-only seam added in C1, with the corrected count of four mirrors and all cards updated; **D10** (a) 4.128.0, breaking. The options below are as presented to the owner. Where they differ from the outcome, the outcome above wins.

- **D1 — DAS performance measurement.**
  - **Recommended (gold standard).** A `--benchmark` mode, run **alone** after every other process, over a fixed seeded sample (3 matches per velocity-bearing provider):
    - **Warm-up and best-of-3** for every leg.
    - **Reference:** timed **inside** the pandas-2 process (the two library calls only).
    - **Native legs:** numpy and numba serial; `compute_das` numba at n_threads ∈ {1, 2, 4, 8, 16, 20}.
    - **`add_das` and `das_xfns`, old vs new in the SAME pandas major.** The old path is the released 4.127.0 wheel in a pandas-2 + accessible-space 2.0.15 env. The new path is C1 installed in its own pandas-2 env (the new code supports pandas 2), so the ratios measure the engine, not pandas.
    - Paired vs independent legs.
    - Peak RSS including subprocesses.
    - **Contention evidence:** the CPU time used by other processes over the whole benchmark, from `/proc/stat` minus the run's own `getrusage`. The gate requires it below 5 % of the machine; before/after load averages are recorded for information only.
    - **Outputs and gate:** writes `performance.json`; the map's contended `timings_ms_per_frame` are labelled "not for ratios" in the artifact README; the gate asserts the §4.2 targets from `performance.json`.
  - **Alternative:** gate only what the map records, and record the missing targets as an owner-approved reduction.
- **D2 — corpus parity numeric bound.**
  - **Recommended:** extend the shard schema (`das-native-parity-3`) so numba is recorded for team and player, for both DAS and AS. The reduce then counts, per provider × grain × output, the rows outside `|Δ| ≤ atol + rtol·|ref|` for numpy vs reference (1e-12) and numba vs reference (1e-10), over `reason == OK` rows. The gate asserts every count is 0. A non-zero count is surfaced, never relaxed.
  - **Alternative:** no schema change — numba is then bounded for team DAS only, which covers 1 of 4 cells.
- **D3 — launcher.** The launcher has a wiring gap (`{subset}` is a JSON path, but `--study` takes one tag; DAS mode would put a path into `--worker-tag`). Its owner-run validations (launcher plan Tasks 7–9: peak RSS, speedup, cgroup-v2 preflight, OOM, byte-identical parity) are also still outstanding. This cycle does not use the launcher.
  - **Recommended:** a separate cycle that fixes the launcher and runs Tasks 7–9.
  - **Alternative:** add both to C1 and to the wave.
- **D4 — public corpus for the re-fits.**
  - **Recommended:** the original 17 (10 for gk_completion skillcorner), matching the model cards and the `_corpus.py` comment. Corrected premise: `position_only` moves by float32 only; the `default` re-fits also absorb the label changes since `6e3a132` (§0.1b).
  - **Alternative:** the current 27 / 20, a corpus widening that would need its own non-inferiority gate.
- **D5 — xcross `default` record.**
  - **(a) Recommended:** public-only like the other three. The record loses the paired block, and its TF-19 probe runs on SkillCorner matches that are in training; the trainer flags this (`train_xcross_attempt.py:711-716`). Nothing outside `_xcross_eval` reads `tf19_ready`.
  - **(b)** Reproduce the paired shape: owner token, GS + the original 17, `--ship-variant public`, run without `--expect-variant` (the G1 preflight would refuse a corpus with GS). Cost: 15 nested studies over ~1.2M rows, and a degenerate comparison (`sc_extended` == `public`).
  - **(c)** Public-only training plus the substitution probe on held-out GS 10502/10503 (owner token, probe only). This needs a small trainer flag in C1.
- **D6 — `estimate_das_cost` per engine.**
  - **Recommended:** add the numpy constant and select it by the engine `auto` resolves (`_das_engine._numba_available`). C1 values come from the local `test_das_benchmark.py` run; C2 values are re-derived from `performance.json`. Advisory only.
  - **Alternative:** record the §6.14 deviation as owner-approved.
- **D7 — receiver corpus label, and re-confirming the widening.** Fact (§0): the 327 statsbomb matches are manifest-`private` and licensed (ADR-062). The committed 30 came from the same licensed corpus. Both bundles say `public` only because the label is derived from the provider name. The weights themselves are derived and non-reversible (three standardized logistic coefficients — redistribution policy: derived work shareable, data not).
  - **(a) Recommended:**
    - re-confirm the 327 widening on the corrected fact;
    - the trainer derives `corpus_visibility` from manifest visibility (`restricted`) and emits an ADR-067-style `reproducibility: restricted` note;
    - the receiver is re-fit at C1 under the fixed trainer, since the archived 327 run carries the old label and must not be hand-edited;
    - the MODEL_CARD's "open-data" wording is corrected.
  - **(b)** Keep `public`, with a recorded rationale (SB360 treated as StatsBomb open data, the pining flag being a hosting artifact). The archived 327 is reused as-is.
  - **(c)** Revoke the widening and re-fit on the identified 30 (§7 failure path).
- **D8 — receiver gate rule.**
  - **(a) Recommended:** the trainer's own Q3 rule — point estimate new − old ≥ 0, CI reported.
  - **(b)** Non-inferiority, 95 % LB > −0.01 (rev 2; looser than Q3, never approved).
  - **(c)** Superiority, 95 % LB > 0.
- **D9 — Hub divergence.** (As presented in rev 3, before the §0.10 correction. The outcome line above wins: four mirrors, and this cycle performs the pushes.) After C1, `ghost-gk-v1` and both outfield mirrors serve different weights from the wheel under the same names, and the sweeper mirrors are already stale.
  - **(a) Recommended:** the owner republishes the five repos after the release (owner-held action; this cycle only lists the exact files).
  - **(b)** Leave them, and document the divergence in the CHANGELOG and the model cards.
- **D10 — version number.**
  - **(a) Recommended:** 4.128.0 with `!` and BREAKING sections, per repo precedent; it keeps the lakehouse's `<5` pin valid.
  - **(b)** 5.0.0 (semver major for the deleted `[das]` extra, removed kwargs, float32 frame dtypes and the DAS value shift). The lakehouse must then widen its pin, as das-native §11.2 notes.

## 13. Review instructions

Round 3 is a `/re-review` by both reviewers against their round-2 reports:
- confirm each disposition in Appendix A with fresh evidence;
- review only changed content (the rev-4 list in the header).

Round 3 results are in Appendix A. Rev 5 / plan rev 4 then changed two things, and round 4 reviews only these:
- **the D9 revision** (header list): spec §1, §3, §9, §11.9, §12 (D9); plan Task 6b (new), Task 7 (README check, `ghost-gk-v1` re-classified `hf_only`), Task 15 Steps 3/6, Task 19 Step 4, Task 20 Step 2 (Hub smoke), Task 21 Step 10 (CHANGELOG Hub text), Task 22 Steps 10–14 (two explicit OWNER GATES, A r4 A-SPEC-05 / A-PLAN-02), and the Global Constraints D9 row;
- **B r3 CCC-PLAN-24..28** (Appendix A): plan Tasks 10, 11, 12b, 17 Step 4b, 18 Step 1, and the Global Constraints in-task check.

Reviewer B also covers the plan tasks it could not read at its round-3 pin: Tasks 2b, 6, 7, 8 and 13–22.

Reviewer A already reviewed the D9 revision (its round 4, Appendix A). A's next pass covers only:
- the fix for A-SPEC-05 / A-PLAN-02: spec §3, §9 and the header; plan header, Global Constraints D9 row, Task 22 Steps 10–14;
- B r3 CCC-PLAN-24..28 as applied, which A has not seen.

Round 6 (A) and round 5 (B) reviewed the B round-4 fixes (Appendix A).

Round 7 (A) and round 6 (B) reviewed the B round-5 fixes (Appendix A).

**Final round (spec rev 8 / plan rev 7; both APPROVE, Appendix A): the B round-6 fix only** (Appendix A, round-6 table): plan Task 2b, test `test_a_missing_or_stale_cached_probe_sample_is_refused` and its fixture. Everything else is settled.

**Previous round (spec rev 7 / plan rev 6): the B round-5 fixes only** (Appendix A, round-5 table): spec header and §9 (C1 expectations); plan Global Constraints (script-gate rule), Task 2b (hermetic refusal tests; absent-meta guard and its test), Task 7 (Step 3b Rule C exemption; Step 4 command; S105 noqa), Tasks 15 Step 3, 19 Step 4, 20 Step 2 and 21 Step 5 (C1 `mirrors_mismatched`), Task 21 Step 1b (outfield card anchor). Everything else is settled.

Each pass records its model id and skill version.

---

## Appendix A — round-1 finding dispositions

| ID | Disposition |
|---|---|
| A-SPEC-01 / CCC-SPEC CONSIDER 2 | Accepted: §0, mcp/ noted |
| A-SPEC CONSIDER (anchor, date) / CCC-SPEC CONSIDER 15 | Accepted: §0, §0.1 |
| A-PLAN-01 / CCC-PLAN CONSIDER 9 | Accepted: one-match old-path and benchmark smokes before the wave (plan) |
| CCC-SPEC-01 / CCC-PLAN-05 | Accepted: §0.1b measured; §8 pre-registers the exact counts and a disposition; D4 premise corrected |
| CCC-SPEC-02 / CCC-PLAN-06 | Accepted: §0 corrected; D7 |
| CCC-SPEC-03 / CCC-PLAN-01 | Accepted: TF-19 `--reduce-only` honouring the allowlist, with a sharded==serial test (§1, §4) |
| CCC-SPEC-04 / CCC-PLAN-11 | Accepted: §2/§6 accounted-key completeness, commit-keyed generation, resume tests |
| CCC-SPEC-05 / CCC-PLAN-15 | Accepted: §0.10, §9 committed Hub driver over the derived population; D9 |
| CCC-SPEC-06 | Accepted: D8 |
| CCC-SPEC-07 / CCC-PLAN-19 | Accepted: §7 failure path; reused-dir count corrected (6 at C1) |
| CCC-SPEC-08 / CCC-PLAN-16 | Accepted: the gate is a committed, tested driver run at C1 |
| CCC-SPEC-09 | Accepted: (a)–(e) in scope (§1, §0.6, §10); (f) folded into D3 |
| CCC-SPEC-10 | Accepted: §3 review gates |
| CCC-SPEC-11 / CCC-PLAN-12 | Accepted: trainers emit `reproducibility`; ghost `position_only` re-fit at C1 (§0.11, §8) |
| CCC-SPEC-12 / CCC-PLAN-02 / CCC-PLAN-14 | Accepted: ids recorded only for all-public corpora (digest otherwise); T10 table not committed; leak test (§11.8) |
| CCC-SPEC-13 | Accepted: D2 schema extension |
| CCC-SPEC-14 / CCC-PLAN-13 | Accepted: D1 same pandas major; foreign-CPU contention gate; map timings labelled |
| CCC-SPEC-15 | Accepted: §5 named tests; §7 negative control |
| CCC-SPEC-16 / CCC-PLAN-18 | Accepted: §0.13, §3 C2 release content; D10 |
| CCC-PLAN-03 | Accepted: §11.7 no merge into the branch |
| CCC-PLAN-04 | Accepted: §3 separate push and PR gates |
| CCC-PLAN-07 | Accepted: §3 C2 provenance via committed drivers; tf24 exemption owner-approved with this spec; tf19 exemption removed |
| CCC-PLAN-08 | Accepted: `providers_for_slice` in the DAS driver (§4) |
| CCC-PLAN-09 | Accepted: §4 fail-closed owner wrapper + count verification |
| CCC-PLAN-10 | Accepted: §2 commit check on every sharded reduce |
| CCC-PLAN-17 | Accepted: D5(b) defined without `--expect-variant`; the plan's alternative paths corrected |
| CCC-SPEC CONSIDER 1, 3, 5, 6, 8–14; CCC-PLAN CONSIDER 1–8, 10–14 | Accepted (spec text or plan steps) |
| CCC-SPEC CONSIDER 4 | Accepted: the one-wave argument no longer cites the sign-off check |
| CCC-SPEC CONSIDER 7 | Accepted as D5(c) |

**Round 2 (B r2):**

| ID | Disposition |
|---|---|
| B r2 spec N1 | Accepted: §2 scopes commit-keyed generations; GKDV/spells keep manifest + sign-off provenance |
| B r2 spec N2 | Accepted: §10 sizing |
| CCC-PLAN-20 | Accepted: Task 9 tests import `pytest` and build full-schema shards (plan) |
| CCC-PLAN-21 | Accepted: the DAS tests assert the refusal that fires (generation check); the `commits_seen` check is kept as defence in depth with a planted-manifest test (plan) |
| CCC-PLAN-22 | Accepted: D-KEY counted by a pure helper with an exact two-period fixture; the smoke match is found from the frames (plan, §10) |
| CCC-PLAN-23 | Accepted: per-cell finite numba comparison counts recorded; the gate requires them non-zero (plan, §11.3) |
| B r2 plan CONSIDER C1–C9 | Accepted (plan) |

Round 3: A APPROVE / APPROVE (rev 4 / plan rev 3). B APPROVE (spec rev 4) / APPROVE scoped (plan rev 3: Tasks 2b, 6, 7 beyond C1, 8 and 13–22 were not reviewed, because the plan changed mid-round). B's five new CONSIDERs were each verified against the code, then accepted into plan rev 4:

| Finding | Disposition |
|---|---|
| CCC-PLAN-24 | Accepted: Task 10 (g) keeps the live `nbmax` (it computes `numba_vs_numpy_das_max_abs`, `validate_das_native_parity.py:590/596`) |
| CCC-PLAN-25 | Accepted: `best_of` typed generically in both modules; `compute_s` returned as a 0-d array; the in-task check now includes `pyright` on touched files (Global Constraints) |
| CCC-PLAN-26 | Accepted: the generation refusal names the token, the commit and `direction_col`; a same-commit `direction_col` mismatch test is added |
| CCC-PLAN-27 | Accepted: a `--prep-only` test on the paired study fixture (a public-only corpus enumerates no studies, so the check would be vacuous) |
| CCC-PLAN-28 | Accepted, both ways: the DAS done-marker is keyed on the generation the driver writes (`--print-generation` → launcher `--das-generation`, required in das mode), and the wave asserts a fresh shard root |

Round 4 (A, D9 delta, on spec `60d9820f` / plan `8c965209`): REQUEST CHANGES on one BLOCKING finding, which is accepted:

| Finding | Disposition |
|---|---|
| A-SPEC-05 / A-PLAN-02 (BLOCKING) | Accepted: each post-release Hub push batch is an explicit OWNER GATE (§3, §9; plan Task 22 Steps 11 and 12). The session shows the exact invocations and their `--verify-only` output, then waits for an explicit yes before any networked push. Approval of one batch never covers the other, or a re-push. The pre-push weight blob ids are recorded in Step 10 so Step 13's "Hub-only weights unchanged" check has a real before-state. |

Round 5 (A, on spec `7e0e10cc` / plan `dc98b175`): APPROVE / APPROVE; A-SPEC-05 / A-PLAN-02 resolved. Round 4 (B, same pins): APPROVE (spec) / REQUEST CHANGES, narrow (plan). Executed in scratch clones, red to green. Each finding was verified against the tree before being accepted into spec rev 6 / plan rev 5:

| Finding | Disposition |
|---|---|
| N3 (spec, CONSIDER) | Accepted: the status line names no round number; Appendix A holds the verdicts; plan Task 15 Step 6 sets the final status before C1 |
| CCC-PLAN-27 PARTIAL → CCC-PLAN-38 | Accepted: the `--prep-only` test stubs only `_corpus.assert_public_corpus` (the synthetic fixture ids are not in `PUBLIC_CORPUS`; G1 and the registry are tested in Task 2), so it stays non-vacuous |
| CCC-PLAN-29 | Accepted: Task 6 `main(argv)` calls `require_clean_tree` itself and passes `prov` into `run` |
| CCC-PLAN-30 | Accepted: tests for a non-discriminating control (identified False, every control tried), a non-reproducing candidate (stubbed and real), D8 at LB = −0.01 exactly, the nearest values on both sides, NaN, and `MARGIN` pinned exactly |
| CCC-PLAN-31 | Accepted: a manifest-less worker at commit A is refused by a reduce at commit B, and the same reduce at A succeeds |
| CCC-PLAN-32 | Accepted: the Hub smoke is an injectable `run()`; `--require-cards-match` and a new `--require-mirrors-match-wheel` (the driver now computes mirror Hub vs wheel `training_commit`) are each tested pass and fail; `main(argv)` flag threading is tested |
| CCC-PLAN-33 | Accepted: the card test ties each value to its one labelled table row (number list exact, one row per label), adds a `Training corpus` row, pins the exact F1b paragraph head, the exact Hub-only note and one exact provenance line per wheel card; an anti-vacuity test shows a stale cell fails with the right number elsewhere |
| CCC-PLAN-34 | Accepted: seam tests for no Hub read before refusal (unregistered, missing card), `hub_sha256_before` against an independent hash, no `create_repo`, and the CLI `--verify-only` path |
| CCC-PLAN-35 | Accepted: the live smoke disables the implicit HF token before importing `huggingface_hub` |
| CCC-PLAN-36 | Accepted: stale `--first30-json` removed from the Task 6 interface |
| CCC-PLAN-37 | Accepted: the id scan adds a `"game_id"` key check (key-only: `_FRAME_KEYS` names the column) and a `<provider>__` joined-key check, and fails on an empty dir |
| CCC-PLAN-39 | Accepted: the probe-provider test asserts its message; a probe id the training pass could load is refused up front (count only, no id in the error), with a test |
| CCC-PLAN-40 | Accepted: `tests/test_bundled_probe_ids_are_committed_ids.py` (C1) pins every bundle's probe ids to the public arm plus GS 10502/10503 |
| B r4 outside-round (Task 5) | Accepted: the receiver test seam stubs `match_visibility` (and the owner-rows test the listing), a `main()`-level D7 visibility test is added, Step 5 runs with the pining network sandboxed, and `Path` → `pathlib.Path` (F821) |

Round 6 (A, on spec `1aed32ef` / plan `3efdae39`): APPROVE / APPROVE. Round 5 (B, same pins; executed sandboxed, mutations re-run): APPROVE (spec; N3 resolved) / REQUEST CHANGES, narrow (plan: one SHOULD FIX introduced by the CCC-PLAN-32 fix). Every round-4 finding was resolved under execution. Verified and accepted into spec rev 7 / plan rev 6:

| Finding | Disposition |
|---|---|
| CCC-PLAN-41 (SHOULD FIX) | Accepted: the injectable Hub-smoke `run(load_model=…)` is a Rule C false positive (it loads Hub MODELS, not corpus matches). It is resolved by a documented `_UNSHARDED_LOOP_EXEMPT` entry, not by a rename that would hide the loop. Task 7's PASS command now runs `test_corpus_driver_resilience.py`, and a Global Constraint runs the repo-wide script gates in every task that touches `scripts/` |
| CCC-PLAN-42 (CONSIDER) | Accepted: the Task 2b refusal tests stub every pining entry point to raise, so a regressed refusal fails the test instead of reaching the live API |
| CCC-PLAN-43 (CONSIDER) | Accepted: the C1 `mirrors_mismatched` is pre-registered (all four mirrors: wheel `3ca609f` vs Hub `adafb72` / `b68328a`) in §9, plan Tasks 15, 19, 20 and `findings.md` |
| B r5 outside-round | Accepted: (a) the Task 2b cache-hit probe guard refuses an ABSENT `_probe_sample/meta.json` with a `SystemExit` (was `FileNotFoundError`), with an absent / stale test; (b) the F1b paragraph's anchor in the two outfield cards (they have no "Both-axes" paragraph) is directly above the metrics table |

Round 7 (A, on spec `edaf7d31` / plan `b9b2578f`): APPROVE / APPROVE. Round 6 (B, same pins; executed and mutated): APPROVE (spec) / REQUEST CHANGES, narrow, one item (plan). CCC-PLAN-41..43 and both round-5 outside-round notes were resolved under execution; the guard itself is correct.

| Finding | Disposition |
|---|---|
| CCC-PLAN-44 (SHOULD FIX) | Accepted: the cached-probe test's fixture saved `labels.npy` as an object array, which the trainer's plain `np.load` (`train_xcross_attempt.py:880`) rejects before the probe check. `labels.npy` is now numeric (the other three arrays stay object), and the plan adds a remove-the-guard check that `[absent]` then fails with `FileNotFoundError` |

Round 8 (A, on spec `94368a7b` / plan `e111bfe9`): APPROVE / APPROVE. Round 7 (B, same pins): APPROVE / APPROVE. Nothing open: CCC-PLAN-01..44, A-SPEC-01..05 and A-PLAN-01..02 are all closed. Edits after approval, during execution: this Status line set to its final state (plan Task 15 Step 6); the plan's one private host replaced by the owner-supplied `$DGX` variable (Global Constraints), per the no-private-locations rule. Implementation review of C1 (A: APPROVE; B: ready after listed fixes), owner-approved 2026-10-02: (M-1) the Hub smoke records fail-closed load refusals as `load_refused` (§9, §11.9; plan Tasks 7, 15, 19, 20, 21, 22), with the C1 set pre-registered as the two sweeper mirrors and the post-push set as empty; (m-1) the Rule C exemption for `validate_das_native_parity.run_benchmark` (plan Task 11); (m-2) the plan's file table names `silly_kicks/tracking/_das_engine.py` (numpy default chunk 16, from the ADR-107 table) and `tests/tracking/test_das.py` (cost-guardrail tests made independent of the constant and the engine). The NIT on the gk_completion `metrics.json` line endings is left as is: every bundle loader normalizes CRLF before hashing, as for the other unpinned bundle files.

Phase B amendment (owner-approved 2026-10-02: option A, amend C1 and force-push the draft PR). The Task 17 smokes found three driver defects, all present before this cycle, and fixed test-first in C1: (F1) the DAS reference leg cast ids with `int()`, failing every IDSSE match; ids now pass through as strings and both legs key players by one rule; (F2) the benchmark's old/new path legs received frames without `team_in_possession`, which `add_das` refuses; possession is now derived first, untimed; (F3) the launcher's cgroup cap ran `systemd-run --scope` for a non-root user, which the system manager refuses; it now uses `--user` and refuses once, before any worker, when a capped no-op cannot start. Two more reference-leg defects then surfaced on real data (owner-approved 2026-10-03, F6 option (i), same amend): (F5) accessible-space returns team results for the rows with possession only but player results for every row, and the leg read both with the possession-row counter, scrambling the player grain (one SkillCorner match: 20022 of 20097 player rows wrong); player values are now read by input row, and any result off that measured shape raises; (F6) native DAS excludes the ball carrier from offside, but the leg never told the library who the carrier was, so a carrier beyond the defensive line counted as offside for the library alone; the carrier is now forwarded as `player_in_possession_col` whenever the frames carry `ball_carrier_player_id`, which certifies the configuration users run (the 4.127.0 path forwarded it too). With both, one SkillCorner and one IDSSE match reproduce the library exactly (0 of 977 and 0 of 895 frames, 0 player rows, differ). The golden frames carry no carrier and have possession on every row, so the golden recipe gate and the frozen generator are unchanged. (F7, owner decision 2026-10-03: option (a)) The OLD path's `das_xfns` (4.127.0 + accessible-space) peaked at 113.4 GiB on an IDSSE match and was OOM-killed inside a 110G cap on a GS match, where the new path needed 2.6 GB; each path run now has a memory ceiling (default 100 GiB), above which it records what finished plus `over_memory` instead of crashing. The §4.2 speedups are medians over the matches where both paths finished that call, with their counts and the per-provider non-fits in `performance.json`; every other benchmark leg runs on every sample match. The ceiling stays 100 GiB (owner decision 2026-10-03), so IDSSE's old `das_xfns` (113.4 GiB) is recorded as not fitting too. Re-review round 3 of the amend: A APPROVE; B APPROVE with m-1..m-4, all resolved (m-1 a test that `_main` arms the ceiling; m-2 an unmeasurable reading is recorded as such, and the page size comes from `mmap.PAGESIZE` so the module type-checks on every host (round 4: A and B APPROVE); m-3 a missing or hung scope probe is a clean refusal; m-4, owner decision 2026-10-03, a `das-reference-contract` CI job runs the real accessible-space 2.0.15 against native on synthetic scenes and cannot pass by skipping). The plan records F4 (the `--prep-only` JSON is the last stdout line) and the smoke and sizing gaps (plan Phase B header). No spec requirement changes.

Phase B amendment 2 (owner decisions 2026-10-04: amend C1 again and re-run the whole wave; fix the manifest double count in the amend; add a gate-passing D3b pair). The first wave, at the first amended C1, failed two acceptance checks and mis-stated two corpus manifests; all three are driver defects, fixed test-first in C1, and every wave run repeats at the new C1. (F8) 216 of 895 SkillCorner matches carried DAS frames outside the golden bound (1197 team and 25715 player rows, max 28.4; GS and IDSSE exact): accessible-space 2.0.15 takes each frame's carrier in the caller's row order but builds the positions from a frame-sorted copy, and the corpus scored rows are not frame-sorted, so frames were paired with other frames' carriers. The reference leg now sorts its rows by frame key before the call; on match 2003157 the out-of-bound frames fall from 18 to 0 (max 0.0), and the contract job gains an unsorted carrier scene with a negative control. (F9) T10 never measured ghost_gk: since `9449461` its adapter unpacked the default `(features, labels)` return as `(features, meta)`, so every match raised and was recorded as a per-model `error:KeyError` status while `n_failed` stayed 0. The adapter now requests `return_meta=True`, a model that errors on every match now refuses the artifact (owner-approved 2026-10-04), and the acceptance requires ghost_gk to be measured. (F10) The GKDV and spells corpus manifests reported 128 matches for the 64-match corpus: a full-population pass replayed every resumed match's counters into a third manifest, summed with the wave workers'. Worker manifests that are summed now record the keys they cover, the aggregate refuses overlapping coverage, and GKDV and spells gain a `--reduce-only` combine that counts each match once and writes no manifest. (F11, found by `/final-review`) The DAS reduce counts exclusions from the per-match markers, because a launcher relaunch, which carries only a killed worker's remaining items, dropped the killed attempt's exclusions from the summed manifests. (D3b) The small paired set refuses at the xcross Brier gate, so its runs write no `model.json`; the comparison moves to a serial vs fan-out pair on the 17 public matches. The `--lock-commit` diff of `_validate.py` is comments plus one additive helper, not comments only; the constants the sign-off reads are unchanged. No spec requirement changes.

Phase B amendment 3 (owner decisions 2026-10-04: merge before the wave; A/B re-reviews before that merge; the TODO "Last updated" block replaced in it). C1 (the cycle code, the reused F1b weights, amendments 1–3, and a brief TODO summary of what the merge provides) is rebased onto `main`, verified CI-faithfully on the exact rebased tree, `/final-review`ed without versioning, re-reviewed, and squash-merged as PR-1, with no version bump and no CHANGELOG entry. Its merge commit on `main`, M, is the run commit for the whole wave: every requirement here that names C1 as the run commit is met by M, which is on `main` by construction, so provenance no longer needs a non-squash merge. Fixes found during the wave land as new commits on `main` through their own owner-gated PR; the owner decides which runs repeat. C2 follows as PR-2 on a new branch off `main` with the release (version, CHANGELOG, TODO, cards); its merge may squash. This changes the commit structure (§3, §11.7) and nothing else.
