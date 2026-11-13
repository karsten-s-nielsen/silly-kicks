# Corpus widening and the visibility-flag split — design

Status: r8 APPROVED by reviewers A and B (r8 re-reviews, no open finding) and by the owner on 2026-11-12
("Approve both"). r8 answers the r7 re-reviews (A: CW-SPEC-16 to 18; B: CW-SB-19,
-20): the provider source, the validator path and the named key-set pins (section 5.7, 7.2, test 15). r7 carried the owner's ruling r7 (the per-provider gate, section 5.7) and
completes the list of edited gates (section 7.2; the plan reviews found the r6 list incomplete). r6 was APPROVED by
reviewers A and B and by the owner ("Approve, write plan"); one requirement line was added after that approval
(section 7 Refit row: `mode = "refit"`), which both plan reviews accepted. r6 answered the r5 re-reviews (A and B: the tier read in `floor.certify`; B: the parity
reference and the order of steps), in section 7.1. r5 answered the r4 re-reviews (A: CW-SPEC-13 PARTIAL and a note on non-paired subsets; B:
CW-SB-16): the run-to-manifest key (section 7.1). r4 answered the r3 re-reviews (A: CW-SPEC-13 to 15; B: CW-SB-13 to
15) and carried the owner's ruling on the release size check. r3 answered the r2 re-reviews (A: CW-SPEC-09 to 12; B: CW-SB-10 to 12) and carried
the rulings on per-job arms and the wheel size limit. r2 answered the r1 reviews (A: CW-SPEC-01 to 08; B: CW-SB-01 to 09)
and carried the probe ruling (tables at the end). Branch `feat/corpus-widening` on
`origin/main` `e5573a40` (silly-kicks 4.142.0). No code until PR #302 has merged (owner's release sequence: #303 +
tag, #302, this cycle, dataset restage, ONE refit, adoption release).

Owner rulings of 2026-11-12 that set this cycle:
- "Widen before refit": the 10 newer public SkillCorner matches join training before the next xShot/xCross refit.
- "Yes: split flag, public arm 91": GS is public data whose raw files we do not redistribute; the cycle separates the
  two meanings of the visibility flag and widens the public arm to SkillCorner 20 + IDSSE 7 + GS 64.
- Why pining marks GS `private` (owner): "we kept them private in pining since the re-distribution rights were
  unclear/undeclared". Pining's flag stays as it is.
- Q-A to Q-E (section 9).
- r2: "Hold them out": the two xCross TF-19 probe matches (GS 10502, 10503; ruling D5 (c)) stay out of every training
  arm. The public training arm is therefore 89 matches (SkillCorner 20 + IDSSE 7 + GS 62) and the extended arm 187.
- r3: "Rule picks the wheel model": the wheel jobs get both arms and ship what the section 5.2 rule selects; the Hub
  jobs get only `sc_extended` (section 5.5).
- r3 (owner note): "there is a size limit on what can go in the wheel due to pypi" (section 5.6).
- r4: "Both files, fail at 80 MB": the release workflow checks the wheel AND the sdist before upload (section 5.6).
- r7: "Per-provider gate": GS would be about 70% of the public arm's matches (62 of 89), and the pooled gates and
  the pooled ship rule could pass on GS rows while a SkillCorner or IDSSE user gets a model worse than the base rate.
  The shipped arm must show skill on each provider's held-out rows; the refit reports each provider's share, skill
  and rule deltas; no reweighting unless a gate fails (section 5.7).

## 1. Intent

`scripts/_corpus.py` reads one pining flag, `visibility`, for two different questions:

1. **May the raw data be redistributed?** For GS: no (rights unclear). Pining's `visibility` answers this.
2. **Can anyone reproduce an artifact from data they can obtain themselves?** For GS: yes (anyone can download it).

Because both questions use the one flag, GS can never enter the `public` arm, never serves as a public held-out fold,
and every GS-containing run is labelled `full`. This cycle gives each question its own registry, widens the
xShot/xCross public arm, and stages the data for the one refit.

What the label is NOT: a redistribution verdict for derived artifacts. Under the reversibility rule (ADR-116), a
non-reversible derived artifact is publishable whatever its corpus (precedent: the ghost-GK wheel, trained on 179
matches including GS and the 98 licensed SkillCorner matches). The `public` label states provenance: the artifact
can be reproduced from publicly obtainable data.

## 2. Facts (read-only survey, 2026-11-12)

**The 10 newer public SkillCorner matches** (registered in `PUBLIC_CORPUS` since 2026-09-10; held out of training by
combined-cycle ruling D4; added with a hackathon): `1874553 1927964 1959846 1986691 1996436 2006363 2007448 2007721
2010085 2016236`. All are `visibility: public` in pining, A-League 2024/2025, with the same artifact types as the
original 10 (`dynamic_events`, `match`, `phases_of_play`, `tracking_extrapolated`). None is in any HF dataset.

Their `dynamic_events` carry the game-interruption markers the ADR-130 stoppage gate needs, in counts comparable with
the original 10 (median per match; the newer 10 have about 7% fewer `game_interruption_after`;
`D:\Temp\widening\markers.py`, output `~/tf46-diag/widening_markers.csv` on the DGX):

| Group | Possessions | `game_interruption_after` | `_before` | Dead spans | Unknown spans |
|---|---|---|---|---|---|
| original 10 | 950.5 | 96.5 | 86.0 | 86.0 | 97.5 |
| newer 10 | 952.5 | 89.5 | 81.5 | 81.5 | 89.5 |

Not yet run on them: the S1 tracking-geometry gate (it runs at staging and can exclude a match; section 6.4).

**GS:** 64 matches, FIFA Men's World Cup 2022, all `visibility: private` in pining (owner-token listing). Two of them
(10502, 10503) are the `xcross_probe` dataset: the held-out TF-19 probe (D5 (c);
`tests/test_bundled_probe_ids_are_committed_ids.py:14-17`).

**Storage on the Hub** (all `silly-kicks/hf-refit-*` datasets are private, ungated repos; sizes from `dataset_info`):

| Dataset | SkillCorner | IDSSE (sportec) | GS | Total |
|---|---|---|---|---|
| `hf-refit-tc3` | 3.83 GB / 108 | 0.92 GB / 7 | 10.95 GB / 64 | 15.70 GB |
| `hf-refit-xshot115` | 3.83 GB / 108 | 0.92 GB / 7 | — | 4.75 GB |
| `hf-refit-xshot17` | 0.35 GB / 10 | 0.92 GB / 7 | — | 1.27 GB |

About 35 MB per SkillCorner match and 171 MB per GS match.

**Consumers** (`git grep` on `e5573a40`; r1's map missed several, see section 10):

| Consumer | Today | Section |
|---|---|---|
| `scripts/hf_jobs/arm_policy.py:45-69` `compute_arms` (off-HF `--arms-json`) | `public` / `sc_extended` / `full` from `PUBLIC_CORPUS` | 5, 6 |
| `train_xshot_occurrence.py:619`, `train_xcross_attempt.py:774` `run_paired` | `("sc_extended" in arms) and ("full" in arms)` | 5 |
| `scripts/_paired.py:12-38` `clears_rule`, `fixed_sequence_ship` (three arms) | the paired ship rule | 5 |
| `train_xcross_attempt.py:79-109,1163` `_gated_probe_matches`, the M6 gate | the probe's held-out status | 5.4 |
| `scripts/_corpus.py:100-164` `BUNDLED_PUBLIC_ARM`, `bundled_public_arm_pairs`, `rehydrate_public_corpus_ids` | the pinned original 17 (and gk_completion's 10 SkillCorner) | 7 |
| `scripts/_corpus.py` `artifact_label`, `assert_public_corpus`, `is_public_row`, `requested_is_all_public` | pining `visibility` for both questions | 3, 4 |
| `_coordination_corpus.py:258-268`, `validate_cover_shadow_rq1.py:178`, `validate_pass_risk_calibration.py:95` | `artifact_label` | 4 |
| `datasets.toml` `visibility`; `datasets.py:281` `public_ids`; `campaign.py:202-207` `--corpus-visibility` | per dataset | 6 |
| `manifest.py:174-181`, `adopt.py:920` (`corpus = "public_ids"`); `docs/artifacts/manifest.toml` | a public bundle's ids vs the lock | 7 |
| Gates: `test_corpus_guard.py:24-32,122-127`, `test_datasets_lock.py:66`, `test_bundled_weights_corpus_policy.py:58-63`, `test_visibility_blind_gate.py:13`, `test_bundled_probe_ids_are_committed_ids.py` | the pins | 7, 8 |

`scripts/_corpus_visibility.py` is unrelated (SkillCorner per-player DETECTION `visibility`); out of scope.

## 3. The two registries

In `scripts/_corpus.py`:

- **`RAW_REDISTRIBUTABLE`** = today's `PUBLIC_CORPUS` (SkillCorner 20 + IDSSE 7), renamed for its meaning. Pining's
  `visibility` stays its live check: `assert_public_corpus` keeps refusing an unregistered match that claims `public`
  (the licensing control). It governs raw data only; nothing in this cycle publishes raw data.
- **`PUBLICLY_OBTAINABLE`** (new) = `RAW_REDISTRIBUTABLE` ∪ the GS matches. It governs provenance (the `public` arm and
  label) and is fail-closed (an unregistered match is not obtainable). It is not read from pining, which answers
  question 1.
- **`HELD_OUT_PROBE`** (new) = GS 10502, 10503 (owner ruling r2). A held-out probe match is in no training arm.

**Derived sources (ADR-056: a registry gate derives its population).** Each registry has one derived source checked by
a test, not a second hand-list:
- `RAW_REDISTRIBUTABLE` keeps today's protection unchanged: the live pining check at run time
  (`assert_public_corpus`, with `expect_full_public_arm` on a maintainer run) and today's CI gate
  (`test_corpus_guard.py`). No committed file lists all 27 ids, so no second CI source is invented;
- the GS part of `PUBLICLY_OBTAINABLE` against the GS units of the committed lock (`datasets.lock.json`, dataset
  `tc3`: 64 games);
- `HELD_OUT_PROBE` against the committed lock's `xcross_probe` units.

`test_visibility_blind_gate.py:13` bans `PUBLIC_CORPUS` by name in-job; it bans all three new names instead.

## 4. Labels (provenance)

`artifact_label(providers, all_obtainable)` keeps its vocabulary for reading (Hyrum: the Hub repo names, the wheel
variant `public`, `metrics.json` `shipped_variant`, the model cards). New artifacts get:

- `public` — every training row is publicly obtainable.
- `sc_extended` — the rows include licensed SkillCorner matches.
- `full` is retired for new artifacts (Q-A): with GS publicly obtainable it equals `sc_extended`. The reading code
  still accepts `full` on committed artifacts; Hub repo names do not change.

The new meaning applies to every new record that uses `artifact_label` (Q-D: xShot/xCross, coordination, cover-shadow,
pass-risk); no committed record is regenerated. The label input changes from "all rows pining-public" to "all rows
publicly obtainable".

## 5. The xShot/xCross arms and the ship rule

### 5.1 The arms

`compute_arms` (off-HF) builds, from the registries and a staged dataset's units:

- `public` = the units in `PUBLICLY_OBTAINABLE` and not in `HELD_OUT_PROBE` (89 when every match is admitted);
- `sc_extended` = `public` + the licensed SkillCorner units (187 when every match is admitted).

It emits no `full` arm. A probe match in the dataset is in no arm; the new dataset excludes them anyway (section 6.3).

### 5.2 The two-arm paired path (owner ruling Q-B, every constant pre-registered here)

Both trainers change the paired predicate to `run_paired = ("public" in arms) and ("sc_extended" in arms)` and
replace the three-arm `fixed_sequence_ship` with a two-arm decision in `scripts/_paired.py`:

- **Held-out folds:** today's `_public_folds` unchanged: `StratifiedGroupKFold(n_splits = max(2, min(5, n_public_matches)),
  shuffle=True, random_state=42)` over the `public` arm's rows (so GS public matches can be held out); fold k's
  public matches are removed from every arm's training rows.
- **K:** the number of USABLE folds: a fold whose delta is NaN (a degenerate fold) is dropped, as `_paired_data_effect`
  does today; 5 folds are planned with 89 public matches.
- **Delta per fold:** PR-AUC on fold k's public held-out rows of the `sc_extended` candidate minus the `public`
  candidate, each fitted at its OWN nested study params (`{arm}_f{k}`, today's nested protocol, `_paired_data_effect`).
- **Rule:** `clears_rule(deltas)` unchanged: positive in at least K−1 of K folds and a positive mean
  (`scripts/_paired.py:12-18`).
- **Outcome:** ship `sc_extended` if the rule clears, else ship `public`. The operator override `--ship-variant`
  stays as today (xCross).
- The ADR-131 gates then apply to the shipped arm, unchanged.

The non-paired branch (a dataset whose `compute_arms` yields one arm) ships that arm, as today.

### 5.3 No separate non-inferiority gate (Q-E)

The ADR-131 gates require every arm to show skill against an honest reference; this supersedes D4's note. The
original 17-match numbers are reported beside the new ones for context only.

### 5.4 The probe stays held out (owner ruling r2)

`_gated_probe_matches` and the M6 gate keep their logic; because no probe match is in any arm (5.1) or in the training
dataset (6.3), the probe is held out by construction. A test fails if a probe match is a training row while the
record reports the probe as held out.

### 5.5 Per-job arms (owner ruling r3)

Today the tier comes from the dataset (`tf46_refit.toml`: public jobs on `xshot17`, `sc_extended` jobs on `xshot115`),
and `campaign.py:202-203` passes `compute_arms(staged_units(j.dataset))` to every job. On one dataset that would give
every job both arms. The fix:

- A campaign job on a visibility-gated trainer declares `arms`, a non-empty subset of `["public", "sc_extended"]`.
  `campaign.py` passes `compute_arms(...)` restricted to the job's `arms` (and `ship_order` restricted the same way).
  A job declaring an arm the dataset does not produce is refused at preflight. A visibility-gated job without `arms`
  is refused (no silent default).
- **Wheel jobs** (`xshot_occurrence` / `xcross_attempt`, variants `default` and `position_only`): `arms = ["public",
  "sc_extended"]`. They run the section 5.2 rule and ship its choice: `public` unless `sc_extended` clears.
  Licensed-derived weights in the wheel are allowed (ruling D7; reversibility, ADR-116); the bundle's label then reads
  `sc_extended`. The manifest entries (`shipped_variant`) move with the bundles in the adoption PR (section 7).
- **Hub jobs** (the `sc_extended` channel): `arms = ["sc_extended"]`, one arm, the non-paired branch; they always ship
  `sc_extended`.

A job can ship only an arm it was given (test 11).

**A single-arm job must cover every staged unit.** In the non-paired branch the study (`single`) and the gate metrics
use all dataset rows (`train_xshot_occurrence.py:629-630, 895-898`), while the fit uses the arm's rows
(`_fit_work`, `:925-931`). So preflight refuses a job with ONE declared arm whose arm is a strict subset of the
dataset's units. Every planned and committed single-arm job satisfies this: `xshot17` → `public` (all 17),
`xshot115` → `sc_extended` (all 115), the Hub jobs on `xshot187` → `sc_extended` (all 187). A paired job (two arms)
is unaffected: its studies and folds are per arm.

### 5.6 The release size limit (owner note r3, ruling r4)

PyPI caps each uploaded file at 100 MB by default. Measured on PyPI (reviewer A, 2026-11-12): at 4.142.0 the wheel is
14.08 MB and the sdist 43.08 MB; at 4.128.0, 12.42 MB and 36.59 MB. The sdist is the larger file and grew 6.5 MB over
those 14 releases. The wheel xShot/xCross `public` bundles are 0.20 to 0.37 MB each and the Hub `sc_extended` boosters
1.06 to 1.82 MB each (`docs/artifacts/observed.json`); if the rule ships `sc_extended` for all four wheel bundles, the
wheel (and the sdist, which contains the same files) grows by about 4.9 MB. This cycle adds:

- **A per-model gate** `artifact_size_lt_5mb` on the xShot/xCross bundles (fail-closed, beside the ADR-131 gates; the
  value the TF-16 spec pre-registered for these boosters, `2026-06-02-tf16-weights-run-design.md:158`).
- **A release check** (owner ruling r4): a new step in `.github/workflows/publish.yml`'s `build` job, after
  `python -m build` and before the upload, beside the tag-version step. It prints the size of every file in `dist/` and
  fails if any file (the wheel or the sdist) is 80 MB or larger: 20 MB headroom below the 100 MB default. The wiring is
  guarded by `tests/test_ci_publish_guard_wired.py`.

### 5.7 The per-provider gate (owner ruling r7, every constant pre-registered here)

**Why.** Counted by match, GS is 62 of 89 public-arm matches (70%) and 62 of 187 extended-arm matches (33%); row
shares are expected to be close (shots and crosses per match are comparable across providers; not measured). The
ADR-131 gates pool every held-out row, and their reference is the pooled training-fold rate. So a model could pass
mostly on GS rows and be worse than the base rate on SkillCorner or IDSSE rows, and a model could beat the pooled
reference partly by recognising the provider when provider base rates differ.

**The gate.** Two acceptance gates join the ADR-131 block, for every candidate, on the same out-of-fold rows and the
same usable folds as `forecast_skill`:

- **Provider of a row:** the row-aligned provider column the design matrix already carries (`_provider`, read as
  `inp.providers`; `train_xshot_occurrence.py:554, :613`, `train_xcross_attempt.py:608`), normalised to the licensing
  identity by `_CONVERTER_TO_IDENTITY` (`scripts/_corpus_identity.py:22`; `sportec` → `idsse`), the same names as an
  artifact's `provider_list`. It is NOT derived from `groups`: `groups` holds bare game ids
  (`_xshot_occurrence.py:831, :946`; `_xcross_attempt.py:431, :559`). The cv fold unit persists the held-out rows'
  identities (`providers.npy`, beside `groups.npy`); an empty provider string refuses the unit (fail loud).
- **Matches of a provider:** the distinct `(provider, group)` pairs of its held-out rows.
- **Reference:** the provider's OWN training-fold true positive rate: for a held-out row of provider P in gate fold
  k, the positive rate of P's rows among fold k's training rows `tr` (the rows behind today's `train_rate =
  y[tr].mean()`). It is computed in the cv fold unit and recorded there (`fold.json`: `train_rate_by_provider`). This
  is stricter than the pooled reference: provider recognition alone earns no skill.
- **Per provider P:** `brier_skill_P = 1 - mean((y-p)^2) / mean((y-r_P)^2)` and `log_loss_skill_P` with the same
  `SKILL_LOG_CLIP = 1e-15`, both over P's held-out rows; point estimates (no bootstrap: IDSSE has 7 matches, and a
  match-bootstrap bound over 7 clusters is too wide to be useful).
- **Gates:** `provider_brier_skill_gt_0` and `provider_log_loss_skill_gt_0` are True iff the candidate has held-out
  rows of at least one provider and EVERY provider with held-out rows has the skill strictly above 0. A NaN skill
  (no training rows of P in a fold, or a zero reference loss) fails the gate (fail-closed, `_gt0`).
- **Home:** `silly_kicks/tracking/_artifact_integrity.py`, beside `forecast_skill` (the one home of the skill math);
  `skill_gates` adds the two keys from `skill["by_provider"]`; a missing `by_provider` fails both (fail-closed, as a
  missing `skill` fails the six ADR-131 gates). The fold-unit computation is `cv_skill` (`scripts/_study_shared.py:301-317`),
  which builds `skill["by_provider"]`. The integrity validator calls `skill_gates` on the run's fresh metrics
  (`validate_trained_artifact_integrity.py:87`) and folds the two provider gates into `calibration_ok` (`:103-108`),
  so they enter its GO verdict (`:117`). A run whose metrics lack `by_provider` (every run before this cycle) fails
  them; this cycle re-validates no run and regenerates no committed verdict file.
- **Scope:** as the ADR-131 gates: the shipped arm's acceptance decides the run (both trainers, wheel and Hub jobs).
  A shipped arm that fails a provider gate fails the run; the run does not fall back to the other arm (that would be
  a choice made after seeing the result).

**Reported, not gating** (in `metrics.json`):

- `skill["by_provider"][P]`: `n_matches`, `n_oof_rows`, `n_positive`, `row_share`, `brier_skill`, `log_loss_skill`,
  and `pr_auc_lift` (within-fold, P's rows only; NaN allowed);
- for the two-arm rule (5.2): the per-fold delta restricted to each provider's public held-out rows,
  `paired["by_provider"][P]` (the decision still uses the pooled deltas, Q-B unchanged).

**No reweighting** in this cycle. If a provider gate fails in the refit, the owner decides the next step with the
reported numbers.

**Wheel change.** `_artifact_integrity.py` is in the wheel, so this cycle changes the wheel and carries a minor
version bump (the number is set at commit prep). The release sequence gains this release before the adoption release.

## 6. Datasets (Part Deux's three points)

### 6.1 Private repos

Every repo this repo STAGES (a `repo` or a `snapshot_repo` under `silly-kicks/`) is and must stay a private Hub repo:
staging GS frames there is not redistribution. `datasets record` and a campaign mount refuse a staged repo that is not
private. All staged repos are private today (`hf-refit-tc3`, `-xshot115`, `-xshot17`, `-xcross-probe`, `-sb360`,
`-sb-open`; checked 2026-11-12). An upstream source we do not own (`sb_open`'s `repo = statsbomb/open-data`, `access =
"open"`) is exempt; its staged copy `snapshot_repo = silly-kicks/hf-refit-sb-open` is not.

### 6.2 Fields

`datasets.toml` `visibility` is replaced by two required keys on every dataset:

| Dataset | `raw` (question 1) | `provenance` (question 2) |
|---|---|---|
| `tc3` | `restricted` | `extended` |
| `xshot17` | `redistributable` | `public` |
| `xshot115` | `restricted` | `extended` |
| `xcross_probe` | `restricted` | `public` (held-out probe; in no arm) |
| `sb360` | `restricted` | `extended` (licensed StatsBomb 360) |
| `sb_open` | `redistributable` | `public` |
| the new dataset (6.3) | `restricted` | `extended` |

`provenance` is declared by hand but CHECKED at `datasets record`: for a tracking dataset, `public` requires every
staged unit to be in `PUBLICLY_OBTAINABLE` (held-out probe units are obtainable too); otherwise it must be `extended`;
a mismatch refuses the record. The StatsBomb datasets keep their declared values; they are outside the tracking
registries.

`--corpus-visibility` for `receiver_model` keeps its CLI vocabulary (`public` / `restricted`,
`train_receiver_model.py:401`): campaign passes `public` iff `provenance = public`, else `restricted` (`sb360` →
`restricted` and `sb_open` → `public`, exactly today's values).

`public_ids` (`datasets.py:281`) is recorded, at the NEXT `datasets record` of a dataset, for every dataset that
contains `public`-arm units: the sorted ids of the units `compute_arms` puts in `public`, before any run (INV-9).
Committed lock entries are NOT re-recorded in this cycle: `tc3` and `xshot115` keep `public_ids: null` (no public
bundle reads them; `xshot115` is deleted after adoption), and `xshot17` keeps its 17. `xshot187` gets its
`public_ids` when Part Deux records it (restage).

### 6.3 Layout and storage (Q-C)

ONE new dataset holds every usable training match: SkillCorner 118 (20 public + 98 licensed) + IDSSE 7 + GS 62 (the 64
minus the two probe matches) = 187 matches, about 15.7 GB, private repo, name `xshot187`. `--arms-json` selects each
arm. `xcross_probe` stays the separate held-out probe dataset. After the refit is adopted, `hf-refit-xshot17`
(1.27 GB) and `hf-refit-xshot115` (4.75 GB) are deleted (owner go), together with their `datasets.toml` entries,
lock entries and campaign references.

### 6.4 A match the S1 gate excludes

The registries list REGISTERED matches. Staging runs S1; a match S1 excludes is not staged. The arms are built from
the STAGED units, so the counts 89 and 187 are expected values, not pins. `datasets record` writes the staged unit
list and `public_ids` (6.2) into the lock before any run; every later check (arms, `rehydrate_public_corpus_ids`'s
n-guard, `adopt`) compares against that recorded list, not against 89. The spec's tests use the recorded list. An S1
exclusion is reported to the owner with the restage.

## 7. Pins by stage (the committed 17-match bundles stay green)

| Stage | Bundled pin (`BUNDLED_PUBLIC_ARM`) | A public bundle's expected `corpus_match_ids` | Gates |
|---|---|---|---|
| This cycle | unchanged: the original 17 (and gk_completion's 10 SkillCorner) | `xshot17`'s locked `public_ids` (17) | the committed bundles' gates unchanged and green; the edited gates are listed below; the new registries get their own tests |
| Restage (Part Deux) | unchanged | — | `xshot187` is declared in `datasets.toml` TOGETHER with its first `datasets record` (one commit), so `set(lock) == set(declared())` (`test_datasets_lock.py:57-59`) never breaks; the record writes its unit list and `public_ids` |
| Refit | unchanged | — | the run reads `xshot187`; `compute_arms` builds the two arms; the refit campaign declares `mode = "refit"` (ADR-130), the mode section 7.1's certification reasoning assumes |
| Adoption release | per bundle, by the rule's outcome (section 7.1); gk_completion keeps its own pin | per section 7.1 | `test_corpus_guard`, `test_datasets_lock`, `test_bundled_weights_corpus_policy`, `test_bundled_probe_ids_are_committed_ids` and `docs/artifacts/manifest.toml` move with the bundles in the adoption PR; `xshot17`/`xshot115` deleted after |

### 7.1 The adoption contract for a wheel bundle, by the rule's outcome

Two different things were both called `tier`; this cycle separates them.

- **The channel** — which destination a campaign job serves — stays the campaign job's `tier`, declared BEFORE the
  run and never changed by it: the wheel jobs declare `tier = "public"`, the Hub jobs `tier = "sc_extended"`, as today.
  It keys the run: the store key `{campaign}-{commit7}-{job}-{tier}` (`campaign.py:147-150`) and `Instance.id`, and
  `adopt.artifacts_for` still matches `(job, variant, tier)` (`adopt.py:81-83`). The wheel job and the Hub job of one
  `(job, variant)` therefore have different store keys and match different manifest entries in every outcome.
- **The shipped arm** — what the run's rule chose — is a new manifest field `shipped_arm` (`public` / `sc_extended`).
  The adoption PR writes it from the run's recorded `shipped_variant`, so the expectation follows a recorded decision.
  For a committed entry without the field, `shipped_arm` reads as its `tier` (today's meaning, unchanged).
- **Every read of `tier` in adopt, and its value after the change** (exhaustive; `rg "tier" scripts/hf_jobs/adopt.py
  scripts/hf_jobs/floor.py` on `e5573a40`):

  | Read | Purpose | Value after the change |
  |---|---|---|
  | `adopt.artifacts_for` (`adopt.py:81-83`) | match an instance to its manifest entry | `tier` (the channel) |
  | the store key / `Instance.id` (`campaign.py:100-102, 147-150`) | key the run | `tier` (the channel) |
  | the `shipped_variant` check (`adopt.py:351-353`) | the run shipped what the entry expects | `shipped_arm` |
  | `spec.certification_for(...)` (`adopt.py:130`) | public head-to-head eligibility | `shipped_arm` |
  | `spec.certification_for(...)` (`adopt.py:648`) | choose the certification branch | `shipped_arm` |
  | `floor.certify(spec, tier, ...)` (`adopt.py:684`, `:697`) → `basis = spec.certification_for(tier)` (`floor.py:100`) | the certification basis inside floor | `shipped_arm` passed as the argument |
  | the run record's `"tier"` field (`adopt.py:281`) | record which channel ran | `tier` (the channel) |
  | the adoption report line (`adopt.py:976`) | describe the entry | `tier`, plus `shipped_arm` printed beside it |
  | `_staging_run_key` (`adopt.py:1285-1293`) | find the run's staging folder | `tier` (the channel) |
  | `_commit_file` (`adopt.py:1297-1308`) | find the entry's commit field | `tier` (the channel) |
  | the cross-commit diff keys (`adopt.py:1321-1354`) | pair the new and old runs | `tier` (the channel) |

  So an `sc_extended` wheel outcome takes the owner branch, and floor's `basis = spec.certification_for(shipped_arm)` is
  `"owner"` (the registry map `{"public": "public", "sc_extended": "owner"}`, `registry.py:268, 297`); floor then runs
  the CV-parity regression check and reports `detail["basis"] == "cv_parity"` (`floor.py:107-116`).
- **The reference and what decides.** The one refit runs in `refit` mode (ADR-130): an arm passes on its OWN acceptance
  gates (ADR-131), and the floor comparison is reported, not gating (`adopt.py:782-790`). For an `sc_extended` wheel
  outcome the reported owner-parity comparison uses the committed entry of the same destination (the committed wheel
  bundle, today the public 17-match model): a context number across a corpus change, never a pass/fail input.
- **The order of steps in the adoption PR:** (1) read each run's recorded `shipped_variant`; (2) write `shipped_arm` (and
  the outcome's fields, table below) into each wheel manifest entry; (3) run `adopt verify` / certify / adopt. Verify
  never runs on an entry whose `shipped_arm` is missing for a wheel `sc_extended` outcome: a wheel entry reads as its
  `tier` (`public`) when the field is absent, so a run that shipped `sc_extended` is refused at the `shipped_variant`
  check until step 2 is done.

Precedents exist for both outcomes: the current wheel xShot/xCross entries (public) and the Hub `sc_extended` entries
(`manifest.toml:362-427`).

| Field | Rule ships `public` | Rule ships `sc_extended` |
|---|---|---|
| `tier` (the channel: matching and the store key) | `public` | `public` (unchanged: the wheel channel) |
| `shipped_arm` (new; `adopt.py:351-353` checks `shipped_variant == shipped_arm`) | `public` | `sc_extended` |
| `certification` | `public` | `owner` |
| `dataset` | `xshot187` | `xshot187` |
| `corpus` | `public_ids` | `digest` |
| expected corpus | `xshot187`'s locked `public_ids` (the admitted public arm, 6.4) | `expected_identity("xshot187", naming)` (the lock digest of the `sc_extended` arm) |
| rehydrate (`adopt.py:920`, `rehydrate_public_corpus_ids`) | yes; its n-guard compares with the locked `public_ids` (not 17, not 89) | no (only `corpus = public_ids` rehydrates) |
| `expect` keys | `shipped_variant = public`, `reproducibility = public`, `provider_list` = the admitted public providers | `shipped_variant = sc_extended`, `reproducibility = sc_extended`, `provider_list` = the admitted providers |
| `test_bundled_weights_corpus_policy` | the public wheel rule (explicit `corpus_match_ids`) | the owner-tier wheel rule the ghost-GK wheel bundles already use (`tier = owner`, corpus provenance) applied to `sc_extended` |

The two outcomes may differ between the four wheel bundles (each job's rule decides). Test 14 goes through the real
campaign instances, both outcomes.

### 7.2 What this cycle edits

This cycle changes no committed bundle, pin, lock entry or manifest entry, and does not declare `xshot187`. It adds
the registries, the label rule, the two-arm path, the per-job arms, the size gates, and the dataset fields and checks.

Committed campaigns get `arms` in this cycle (section 5.5 refuses a gated job without them), keeping today's behaviour:
in `scripts/hf_jobs/campaigns/tf46_refit.toml` and `cycle1.toml`, the wheel jobs on `xshot17` declare `arms =
["public"]` (the only arm `xshot17` produces) and the Hub jobs on `xshot115` declare `arms = ["sc_extended"]`
(today's `ship_order[-1]`). The refit's own campaign on `xshot187` is written at the refit step, with the section 5.5
arms.

Gates and files this cycle EDITS (all others stay unchanged). Each is a direct consequence of a decision above; none
adds behaviour beyond that decision:

- the rename (section 3): `tests/scripts/test_corpus_guard.py` (name only), `tests/scripts/test_corpus_taxonomy.py`,
  `tests/scripts/test_corpus_identity_split.py:6`, `tests/scripts/test_xshot_visibility_blind.py:21-22`,
  `tests/scripts/test_xcross_visibility_blind.py:21-22`, `tests/scripts/hf_jobs/test_visibility_blind_gate.py:13`,
  `tests/scripts/test_coordination_corpus_visibility.py:13`, comments in `tests/_skillcorner_sample.py:18` and
  `tests/invariants/test_input_convention_detector.py:331`, `docs/context/conventions-core.md:38`, and the `AGENTS.md`
  bullet that cites ADR-038 for `artifact_label` (gains ADR-132). The frozen snapshot
  `tests/fixtures/claude_md_at_77286f4.md` is NOT edited;
- the label (section 4, Q-A): `tests/scripts/test_corpus_taxonomy.py:34`, `tests/scripts/test_coordination_corpus_visibility.py:37`
  and `tests/scripts/test_derive_coordination_params.py:499` asserted `full`; each asserts the new rule's value for its
  pairs;
- the arms (5.1, the probe in no arm): `tests/scripts/hf_jobs/test_arm_policy.py:41-53` (F29) restates its invariant
  as "every NON-probe unit is in an arm and the probe units are in none" (`tc3` holds 10502 and 10503);
- the two-arm rule (5.2): `tests/scripts/test_paired_rules.py` (`fixed_sequence_ship` removed),
  `tests/scripts/test_nested_gate_folds.py:38-79, 317-349`, `tests/scripts/test_expect_variant_wiring.py:105-124`, and
  the toy payload `scripts/_toy_corpora.py:524-534` (used at `:1118`, `:1128` and by the parity tests) emit two arms;
- the dataset fields (6.2): `tests/scripts/hf_jobs/test_datasets_lock.py:14` (`visibility` → `raw` + `provenance`),
  `tests/scripts/test_train_receiver_model.py:409` (reads `provenance`);
- per-job arms (5.5): the campaign tests that build `--arms-json`; `tf46_refit.toml` and `cycle1.toml` (the `arms`);
- the size check (5.6): `.github/workflows/publish.yml` and `tests/test_ci_publish_guard_wired.py`;
- the per-provider gate (5.7) and the size gate (5.6): the key-set pins `tests/tracking/test_forecast_skill.py:26-34`
  (`GATE_KEYS`) and `:240-243`, `tests/scripts/test_nested_gate_folds.py:203-211`; the validator fixtures
  `tests/scripts/test_validate_trained_artifact_integrity.py:48-58` and `:87-94` (they gain `by_provider`);
- the cv fold unit (5.7): the fold-unit round-trip tests gain `providers.npy` and `train_rate_by_provider`.

Unchanged and green (checked): `tests/scripts/hf_jobs/test_hf_arms.py:32` (the generic `ArmSpec` parser; the `full`
refusal sits in trainer prep, not in `arms.parse_arms`), `tests/tracking/test_xshot_occurrence_integration.py:694-744`
(the generic `_paired_data_effect`), the ghost-GK `full` variant, the coordination `sc_extended` tests,
`corpus_identity` / `check_expected_variant` and their `all_public=` callers (`scripts/_corpus_identity.py:68-112`,
`scripts/hf_jobs/datasets.py:194-195`, xshot `:971`, xcross `:1225`: the in-job digest path, not the label), and the
SB360 registry's own `entry.visibility` (`tests/sb360/_registry.py`, `test_axis_locks.py`, `test_registry_surface.py`,
`render_sb360_matrix.py`: a different field).

## 8. Tests (TDD; each fails first)

1. Registry derivation: each registry equals its derived source (section 3), not a hand-list. Ids compare as
   (identity provider, game), with the provider normalised by `_CONVERTER_TO_IDENTITY` (`scripts/_corpus_identity.py`,
   the map `arm_policy` uses). A flat unit's `game` is the game id (`tc3`: `10502`). A pergame unit's `game` carries
   the provider prefix (`xcross_probe`: `game = stem = "gradientsports-10502"`); the test strips the `<provider>-`
   prefix and asserts the committed `xcross_probe` entry yields (`gradientsports`, `10502`) and (`gradientsports`,
   `10503`).
2. Fail-closed: an unregistered match is neither redistributable nor obtainable; a pining-`public` match missing from
   `RAW_REDISTRIBUTABLE` still refuses (today's licensing control).
3. Labels: GS-only → `public`; GS + public SkillCorner → `public`; any licensed SkillCorner row → `sc_extended`; no
   input yields `full`; a committed `metrics.json` carrying `full` still loads and verifies (Q-A).
4. `compute_arms`: `public` = obtainable minus probe, `sc_extended` = `public` + licensed SkillCorner, no `full`; a
   probe unit is in no arm; arms are built from the given units (an excluded match changes the counts, not the code).
5. Two-arm paired path (both trainers): with arms `public` and `sc_extended`, `run_paired` is True and the decision
   uses `clears_rule` on the per-fold deltas; a two-arm run that ships `sc_extended` without the rule clearing FAILS
   the test; a cleared rule ships `sc_extended`; an uncleared one ships `public`.
6. Probe hold-out: a probe match as a training row while the record calls the probe held out fails.
7. Private repos: `datasets record` / mount refuse a non-private staged repo; `sb_open` (`access = "open"`) passes.
8. Fields: `raw` and `provenance` are required on every dataset; the old `visibility` key is refused (no fallback);
   `--corpus-visibility` maps `provenance` to `public` / `restricted` (sb360 → restricted, sb_open → public).
9. `datasets record` (on a fixture dataset, not the committed lock) writes `public_ids` = `compute_arms`' public arm
   for a dataset with public-arm units; the committed lock is not re-recorded.
10. This cycle's no-change guarantee: the committed bundles' gates (section 7 row 1) pass unchanged.
11. Per-job arms: a job gets only its declared arms; a job without `arms`, with an arm the dataset lacks, or with ONE
    arm that is a strict subset of the dataset's units (a public-only job on `xshot187`) is refused at preflight; the
    wheel jobs (both arms) run the 5.2 rule; the committed single-arm jobs (`xshot17` → `public`, `xshot115` →
    `sc_extended`) pass.
12. Provenance check: `datasets record` refuses `provenance = public` when a unit is not publicly obtainable.
13. Size: `artifact_size_lt_5mb` fails a bundle of 5 MB or more (both trainers); the release step fails when any
    `dist/` file (wheel or sdist) is 80 MB or larger and passes below (exercised on a stub `dist/`), and
    `test_ci_publish_guard_wired` fails if the step is missing from `publish.yml`'s `build` job before the upload.
14. Adoption contract, through the campaign code: build the wheel instance (`tier = public`, both arms) and the Hub
    instance (`tier = sc_extended`) of one `(job, variant)` on `xshot187` from a campaign fixture; assert distinct store
    keys and instance ids; then, for each outcome of the wheel run (`shipped_variant` = `public`, then `sc_extended`),
    run `adopt` over both instances and assert: each instance matches exactly its own manifest entry (wheel → the wheel
    destination, Hub → the Hub destination, never crossed); the wheel entry's `shipped_arm` and certification follow the
    outcome (public: `public_ids`, rehydrate with the locked list, certification basis `public`; `sc_extended`: `digest`,
    no rehydrate, `floor.certify`'s `detail["basis"] == "cv_parity"`); a `shipped_variant` that disagrees with
    `shipped_arm` is refused (incl. an `sc_extended` run before step 2 wrote the field).
15. Per-provider gate (5.7), on fixtures with the REAL shapes (bare game-id `groups`, a separate row-aligned
    provider column with `sportec` rows): one provider's skill <= 0 against its own training-fold rate while pooled
    skill is positive → `provider_brier_skill_gt_0` (or `provider_log_loss_skill_gt_0`) False and the run fails; a
    model that predicts only each provider's training-fold rate beats the pooled reference but has provider skill 0
    and fails; a NaN provider skill and a missing `by_provider` fail; every provider skilled passes; `by_provider` is
    keyed `idsse`, never `sportec`, and never by game id; an empty provider string refuses the fold unit; the
    validator's `calibration_ok` is False when a provider gate fails.
16. Per-provider reporting (5.7): `skill["by_provider"]` carries the listed fields for every provider; the paired
    payload carries `paired["by_provider"]`; the ship decision is unchanged by them (pooled deltas decide).

## 9. Owner rulings (2026-11-12)

| Question | Ruling |
|---|---|
| Q-A: the `full` label | "Retire for new artifacts" (section 4) |
| Q-B: the paired ship rule | "public vs sc_extended", `clears_rule` unchanged (section 5.2) |
| Q-C: dataset layout | "One 189-match dataset"; with the r2 probe ruling it holds 187 (section 6.3) |
| Q-D: scope of the label change | "All new records"; no committed record regenerated (section 4) |
| Q-E: D4's non-inferiority gate | "No extra gate" (section 5.3) |
| r2: the probe matches | "Hold them out": public 89, extended 187 (sections 5.1, 5.4, 6.3) |
| r3: who runs the ship rule | "Rule picks the wheel model": wheel jobs get both arms and ship the rule's choice; Hub jobs get `sc_extended` only (section 5.5) |
| r3: the wheel size | owner note on the PyPI limit (the note named the limit only; the 5 MB value is TF-16's) (section 5.6) |
| r4: the release size check | "Both files, fail at 80 MB": wheel and sdist, in `publish.yml` before upload (section 5.6) |
| r7: GS dominance (62 of 89 public matches) | "Per-provider gate": brier and log-loss skill > 0 on each provider's held-out rows against its own training-fold rate; per-provider reporting; no reweighting (section 5.7) |

## 10. r1 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SPEC-01, CW-SB-08 (probe matches enter training) | Owner ruling "Hold them out"; `HELD_OUT_PROBE`; out of every arm and of `xshot187`; test 6 |
| CW-SPEC-02, CW-SB-01 (no two-arm paired path; constants) | Section 5.2: the predicate, the decision, folds, K, delta, rule, outcome; test 5 |
| CW-SPEC-03, CW-SB-02, CW-SB-03 (pins and `public_ids` break) | Section 7: pins by stage; this cycle changes no committed pin; `public_ids` for every dataset with public-arm units (6.2) |
| CW-SPEC-04, CW-SB-04, CW-SB-05 (StatsBomb datasets) | Section 6.2 table incl. sb360/sb_open/xcross_probe; `--corpus-visibility` mapping; 6.1 exempts `access = "open"` |
| CW-SPEC-05, CW-SB-06 (S1 exclusion vs pinned 91) | Section 6.4: counts are expected values; checks use the recorded unit list |
| CW-SB-07 (tautological registry test) | Section 3: each registry against a derived source |
| CW-SB-09, A's CONSIDER (line refs, `test_visibility_blind_gate`, "rate") | Line refs corrected; section 3 last paragraph; section 2 says "counts comparable", with the 7% figure |

## 11. r2 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SPEC-09 (no per-job arm set on one dataset) | Owner ruling r3; section 5.5 (job `arms`, preflight refusals); test 11 |
| CW-SPEC-10 (probe id normalisation) | Test 1: (identity provider, `game`) via `_CONVERTER_TO_IDENTITY` |
| CW-SPEC-11, CW-SB-12 (provenance unchecked) | Section 6.2: checked at `datasets record`; test 12 |
| CW-SPEC-12, CW-SB-10(b) (`public_ids` vs the committed lock) | Section 6.2: next record only, no re-record now; test 9 on a fixture |
| CW-SB-10(a) (declaring `xshot187` breaks the lock test) | Section 7: declared with its first record, at restage |
| CW-SB-10 (gates edited) | Section 7: the list of edited gates |
| CW-SB-11 (`sb_open` snapshot) | Section 6.1: a staged `snapshot_repo` must be private (it is); only the upstream repo is exempt |
| CW-SB-12 (K, storage) | Section 5.2: K counts usable folds; section 6.3: 15.7 GB |

## 12. r3 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SB-13, CW-SPEC-13 (an `sc_extended` wheel bundle cannot be adopted) | Section 7.1: the adoption contract for both outcomes; test 14 |
| CW-SB-14 (pergame `game` carries the provider prefix) | Test 1: strip `<provider>-`; asserts the committed `xcross_probe` entry |
| CW-SB-15 (committed campaigns have no `arms`) | Section 7: `tf46_refit.toml` / `cycle1.toml` get today's arms; in the edit list; row 1 fixed |
| CW-SPEC-14 (the sdist is the larger upload) | Owner ruling r4; section 5.6: wheel and sdist checked, current figures; test 13 |
| CW-SPEC-15 (the 50 MB figure; where the check runs) | Section 9 marks the note's scope; the check runs in `publish.yml`'s `build` job (5.6) |

## 13. r4 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SPEC-13 PARTIAL, CW-SB-16 (the wheel `sc_extended` outcome and the instance key) | Section 7.1: `tier` = the channel (fixed before the run; store key, `Instance.id`, matching); new `shipped_arm` (from the run record; the `shipped_variant` check and certification read it); test 14 through both real instances, both outcomes |
| A's note (a single-arm job on a strict subset fits on its arm while its study and gate use all rows) | Section 5.5: preflight refuses a one-arm job whose arm is a strict subset of the units; test 11 |

## 14. r5 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SPEC-13 PARTIAL, CW-SB-17 (the `floor.certify` tier read) | Section 7.1: the exhaustive table of `tier` reads with their values; `floor.certify` receives `shipped_arm`; test 14 asserts `basis == "cv_parity"` |
| CW-SB-18 (the parity reference; the order) | Section 7.1: refit mode gates on the arm's own acceptance, the comparison is reported against the committed entry of the destination; the adoption PR writes `shipped_arm` before verify |

## 15. r7 changes (owner ruling r7; plan reviews A and B, round 1)

| Source | Change |
|---|---|
| Owner question on GS dominance; ruling r7 | Section 5.7: per-provider gate, per-provider reporting, wheel change and version bump; tests 15, 16 |
| Plan review A CWP-PLAN-01, B CW-PB-01 (the r6 edit list was incomplete) | Section 7.2: the complete list, grouped by the decision that causes each edit; F29 restated |
| Plan review A CONSIDER (docs and comments outside the list) | Section 7.2: `conventions-core.md:38`, the two comments, the `AGENTS.md` bullet; the frozen snapshot excluded |
| Plan review B CW-PB-03 (where the `full` refusal sits) | Section 7.2: in trainer prep, so `test_hf_arms.py:32` stays green |

## 16. r7 findings and their resolution

| Finding | Resolution |
|---|---|
| CW-SPEC-16, CW-SB-19 (BLOCKING: `groups` are bare game ids, no provider prefix) | Section 5.7: the provider is the row-aligned `_provider` / `inp.providers`, normalised by `_CONVERTER_TO_IDENTITY`; the cv unit persists `providers.npy`; an empty provider refuses; test 15 uses the real shapes |
| CW-SPEC-17, CW-SB-20 (the validator path) | Section 5.7: `cv_skill` computes; the validator applies `skill_gates` to the fresh metrics and folds the provider gates into `calibration_ok` and GO |
| CW-SPEC-18, CW-SB-20 (unnamed key-set pins; validator fixtures) | Section 7.2 names `test_forecast_skill.py:26-34, 240-243`, `test_nested_gate_folds.py:203-211`, `test_validate_trained_artifact_integrity.py:48-58, 87-94` |
| CWP-PLAN-07 (grep hits that stay) | Section 7.2 unchanged list: `corpus_identity` `all_public=` callers; the SB360 `entry.visibility` |
