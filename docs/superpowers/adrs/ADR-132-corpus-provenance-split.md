# ADR-132: Split the corpus visibility flag into raw redistribution and provenance; widen the xShot/xCross arms

| Field | Value |
|---|---|
| **Date** | 2026-11-12 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner rulings 2026-11-12: Q-A to Q-E, r2, r3, r4, r7); reviewers A and B (spec r1 to r8, plan r1 to r3) |
| **Supersedes / amends** | amends ADR-038 (the label and the public registry); amends ADR-131 (two per-provider acceptance gates); supersedes the combined-cycle D4 note (no separate non-inferiority gate) |
| **Source spec** | `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (r8) |

## Context

`scripts/_corpus.py` read one pining flag, `visibility`, for two different questions:

1. May the raw data be redistributed? For the Gradient Sports (GS) FIFA Men's World Cup 2022 matches: no. The
   redistribution rights are unclear, so pining keeps them `private`.
2. Can anyone reproduce an artifact from data they can obtain themselves? For GS: yes. Anyone can download it.

Because one flag answered both questions, GS could never enter the `public` training arm or serve as a public held-out
fold, and every GS-containing run was labelled `full`. The public xShot/xCross arm stayed at 17 matches; ADR-131 showed
that the public xCross `position_only` arm is underpowered at that size. Ten newer public SkillCorner matches
(registered 2026-09-10) were also held out of training by D4.

## Decision

1. **Two registries and a probe.** `RAW_REDISTRIBUTABLE` (the old `PUBLIC_CORPUS`, SkillCorner 20 + IDSSE 7) answers
   question 1, together with pining's live `visibility` check (`assert_public_corpus`, unchanged). `PUBLICLY_OBTAINABLE`
   (`RAW_REDISTRIBUTABLE` + GS `3812`-`3859` and `10502`-`10517`, 64 matches) answers question 2; it is fail-closed and
   not read from pining. `HELD_OUT_PROBE` (GS `10502`, `10503`, the xCross TF-19 probe; ruling r2) is in no training
   arm. Each registry has one derived source checked by a test: the GS part against the committed `tc3` lock, the probe
   against the committed `xcross_probe` lock.
2. **The label is provenance.** `artifact_label(all_obtainable=...)` returns `public` when every training row is
   publicly obtainable, else `sc_extended`. `full` is retired for new artifacts (Q-A); committed records keep it, and
   its readers accept it (Q-D: no committed record is regenerated).
3. **The arms.** `compute_arms` builds, from a staged dataset's units: `public` = obtainable minus the probe;
   `sc_extended` = `public` + the licensed SkillCorner units. It emits no `full` arm, puts a probe unit in no arm, and
   refuses a non-probe unit that has no arm. With every match admitted, `public` holds 89 matches (SkillCorner 20 +
   IDSSE 7 + GS 62) and `sc_extended` 187 (dataset `xshot187`, staged by the restage step).
4. **The two-arm rule** (Q-B). With both arms, the trainers run the paired path on the public held-out folds and ship
   `sc_extended` iff its per-fold nested PR-AUC deltas over `public` clear `clears_rule`, else `public`
   (`_paired.two_arm_ship`; `fixed_sequence_ship` is removed). A legacy `--arms-json` with a `full` arm is refused in
   trainer prep. The xCross `--ship-variant` override drops `full`. No separate non-inferiority gate (Q-E).
5. **The per-provider gate** (ruling r7). GS is 62 of the 89 public matches, so a pooled gate could pass on GS rows
   while SkillCorner or IDSSE users get a model worse than the base rate. Two acceptance gates join the ADR-131 block:
   `provider_brier_skill_gt_0` and `provider_log_loss_skill_gt_0`. Each is True iff every provider with held-out rows
   has Brier (log-loss) skill strictly above 0 against that provider's OWN training-fold true rate (point estimates,
   `SKILL_LOG_CLIP = 1e-15`, no bootstrap: a 7-match provider's cluster bound is too wide). The provider is the row's
   `_provider` normalised by `_CONVERTER_TO_IDENTITY` (`sportec` -> `idsse`), never `groups` (bare game ids). The cv
   fold unit records `providers.npy` and `train_rate_by_provider`; an empty provider refuses the unit. A NaN skill or
   a missing `by_provider` fails (fail-closed). The integrity validator folds both gates into `calibration_ok`. A
   shipped arm that fails a provider gate fails the run; no fallback to the other arm, no reweighting. Reported, not
   gating: `skill["by_provider"]` and `paired["by_provider"]`.
6. **Per-job arms** (ruling r3). A visibility-gated campaign job declares `arms` (a non-empty subset of `public`,
   `sc_extended`). The wheel jobs get both arms and ship the rule's choice; the Hub jobs get `sc_extended` only.
   Preflight refuses a gated job without `arms`, an arm the dataset lacks, and one arm that is a strict subset of the
   dataset's units, held-out probe units included (a single-arm job studies and gates on every row but fits on its
   arm's rows, and a probe row in that study would break the probe's held-out status).
7. **Dataset fields and private repos.** `datasets.toml` replaces `visibility` with `raw` (`redistributable` /
   `restricted`) and `provenance` (`public` / `extended`). `datasets record` checks `provenance` for tracking data
   (`public` iff every staged unit is publicly obtainable) and records `public_ids` as the units of the `public` arm.
   A dataset is tracking when its providers are in the registries and non-tracking when they are known non-tracking
   sources (`arm_policy._NON_TRACKING_PROVIDERS`: `snapshot`, the SB360 shards); a provider in neither set refuses.
   Every repo this repo stages must be a private Hub repo; `datasets record` and the campaign preflight refuse
   otherwise, fail-closed on an API error. `receiver_model`'s `--corpus-visibility` maps from `provenance`.
8. **The adoption contract.** `tier` stays the channel (store key, `Instance.id`, manifest matching). The new manifest
   field `shipped_arm` holds what the rule chose; an entry without it reads as its `tier`. The `shipped_variant`
   check, the certification branch and `floor.certify` read `shipped_arm`. A public bundle's `corpus_match_ids` are
   rehydrated from its dataset's locked `public_ids`.
9. **Size gates.** `artifact_size_lt_5mb` (5_000_000 bytes, the TF-16 value) fails a shipped xShot/xCross bundle
   closed. The release workflow checks the wheel and the sdist before upload and fails at 80_000_000 bytes or more
   (`scripts/check_dist_sizes.py`; ruling r4).

## Alternatives considered

- **Keep one flag.** Rejected: GS stays out of every public arm, and the public xCross arm stays underpowered.
- **Three arms (`full` kept).** Rejected (Q-A): with GS obtainable, `full` equals `sc_extended`.
- **Reweight providers.** Not in this cycle (r7): the gate decides; the owner decides the next step with the reported
  numbers if a provider gate fails in the refit.
- **A bootstrap bound per provider.** Rejected: over 7 IDSSE clusters the bound is too wide to be useful.

## Consequences

- The wheel changes (`_artifact_integrity.py`), so this cycle is its own minor release before the refit.
- Every run before this ADR fails the provider gates (no `by_provider`); this ADR re-validates no committed run.
- The release sequence: this release, then the `xshot187` restage, then ONE refit (`mode = "refit"`), then the
  adoption release (manifest `shipped_arm`, cards, pins). The committed 17-match bundles, pins, lock entries and
  manifest entries do not change in this cycle.
- `hf-refit-xshot17` and `hf-refit-xshot115` can be deleted after the refit is adopted (owner go).

## Related

- **Specs:** `docs/superpowers/specs/2026-11-12-corpus-widening-design.md`
- **Plans:** `docs/superpowers/plans/2026-11-12-corpus-widening-plan.md`
- **ADRs:** amends ADR-038 and ADR-131; builds on ADR-116 (reversibility), ADR-123 Amendment B (visibility-blind
  trainers), ADR-128 (stage contract) and ADR-130 (refit mode).
