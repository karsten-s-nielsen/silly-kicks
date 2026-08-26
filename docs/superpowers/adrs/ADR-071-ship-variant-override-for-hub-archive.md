# ADR-071: `--ship-variant` operator override — the Hub `sc_extended` repo is the owner-tier ARCHIVE, not the bundle-gate winner

| Field | Value |
|---|---|
| **Date** | 2026-08-26 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The xShot / xCross trainers run a fixed-sequence *bundle-selection* paired test (`scripts/_paired.py::fixed_sequence_ship`) that decides which single variant — `public`, `sc_extended`, or `full` — to ship as THE wheel-bundled artifact. `sc_extended` ships only if it clears `clears_rule` (positive in ≥ K-1 of K folds AND a positive mean); otherwise the sequence stops and `public` ships. That rule decides what goes in the WHEEL.

The HuggingFace Hub repos `silly-kicks/xshot-occurrence-v1` and `silly-kicks/xcross-attempt-v1` are a *different* thing: the owner-tier **archive**, Hub-only for licensing, holding the `sc_extended` variant (both carry `shipped_variant: sc_extended`, `provider_list: [idsse, skillcorner]`). They exist so a licensed consumer can `from_variant("sc_extended")`. Their identity is "the owner-tier model", independent of whether that model beats the public baseline this training run.

The forcing function: during the 4.94.0 Hub re-fit (to repair artifacts unloadable since v4.74.0), the faithful xCross run's `sc_extended` candidate **missed the fold-consistency bar by one fold** — deltas `[-0.006, +0.026, -0.013, +0.029, +0.024]` → 3/5 folds positive (needs 4/5), **mean +0.012 (positive)** — so the gate shipped `public`. The original archived artifact was the *same* 3-provider run with `sc_extended` winning; this is a marginal, noise-sensitive flip, not a finding that the owner-tier model is worse. A `public` model is redistributable and belongs in the wheel, **not** in a Hub-only archive — publishing it there would corrupt both the repo's semantics and its licensing rationale.

The obvious workaround — drop `gradientsports` so the corpus goes single-candidate and the trainer labels it `sc_extended` directly — **does not work for xCross**: the xCross trainer mandates the TF-19 substitution probe as a headline deliverable and raises `SystemExit` (before writing `metrics.json`) when the probe cohort is empty, which it is for a `gradientsports`-free corpus. (xShot has no such probe, so its 2-provider path completes cleanly — see Consequences.)

## Decision

Add `--ship-variant {public,sc_extended,full}` (default `None` = unchanged gate behavior) to the xCross trainer. When set, it **overrides the shipped variant AFTER the paired test runs** — the gate's verdict and per-fold deltas are still recorded in `metrics.json` under `candidates.paired`, so the override is fully auditable — and it ships that variant's model fit on the **same corpus mask the gate would have used**, so the model is identical to a gate-selected ship of that variant. The Hub `sc_extended` archive is populated via this override whenever the bundle gate selects `public`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Accept the gate verdict — publish `public` to the Hub `sc_extended` repo | no code | corrupts repo semantics (public = redistributable/bundled, not Hub-only); breaks the licensing rationale | the archive is defined by tier, not by beating the baseline |
| B. 2-provider single-candidate run (drop `gradientsports`) | no code; deterministic `sc_extended` | **xCross `SystemExit`s on the mandatory TF-19 probe (empty cohort → no `metrics.json`)**; also diverges `metrics.providers` | incomplete artifact; the probe is a required deliverable for xCross |
| C. Copy the 3-provider probe sample into a 2-provider cache and re-run | no code | hacky; the probe diagnostic then runs on a corpus the model wasn't trained on; inconsistent provenance | a shortcut that launders diagnostic provenance |
| D. `--ship-variant` override on the 3-provider run (chosen) | complete + blessed artifact (probe present); auditable (records the overridden verdict); reuses cached Optuna studies (fast); reproduces the original's provenance exactly | one new operator flag to maintain | — |

## Consequences

### Positive

- The Hub `sc_extended` archive can be regenerated deterministically regardless of a noise-sensitive gate flip, with a complete artifact (model + metadata + `metrics.json` + TF-19 probe) that reproduces the original's `provider_list: [idsse, skillcorner]` / `providers: [all 3]` provenance.
- The override is self-documenting: `metrics.json` records BOTH the gate's actual verdict/deltas and the operator override, so a reader can always see what the bundle gate would have chosen.
- Reusable for any future Hub re-fit where the owner-tier variant marginally misses the bundle bar.

### Negative

- A new operator flag on a trained-artifact driver. It can produce a Hub artifact that does not beat `public` — but that is the archive's purpose (tier, not superiority), and the recorded gate deltas keep that honest.
- The flag bypasses a statistical gate; misuse (e.g. forcing `full` to the wheel bundle) would defeat the fixed-sequence error-rate control. It is scoped to the *archive* production path (a `scripts/` driver), never the wheel-bundle decision, and the guard raises on a single-candidate corpus where the masks/probe do not exist.

### Neutral

- Added to the xCross trainer only. xShot does not need it: xShot has no TF-19 probe, so its 2-provider single-candidate path completes cleanly and already ships `sc_extended` (the 4.94.0 xShot Hub artifacts were produced that way). The asymmetry is intentional — the flag lives where the probe requirement makes the 2-provider path unusable.
- The xShot 2-provider Hub artifacts carry `metrics.providers: [idsse, skillcorner]` (only two providers were loaded) vs the xCross 3-provider `[idsse, skillcorner, gradientsports]`; the `metadata.provider_list` (`[idsse, skillcorner]`) and the fitted model are identical in both cases.

## Related

- **Issues / PRs:** `#217` (silly-kicks 4.94.0)
- **ADRs:** builds on ADR-038 (public-label gate at training time), ADR-052 (artifact-driver provenance), ADR-067 (position-only variants), ADR-070 (position-only Hub variant)
- **Code:** `scripts/train_xcross_attempt.py` (`--ship-variant`), `scripts/_paired.py::fixed_sequence_ship`

## Notes

The override deliberately runs *after* `fixed_sequence_ship` rather than skipping the paired test, so the gate's verdict and deltas are computed and recorded even when overridden. Reusing the existing 3-provider feature cache + Optuna study databases makes the re-run fast (the studies resume rather than re-searching).
