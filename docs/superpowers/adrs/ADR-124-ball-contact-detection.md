# ADR-124: Ball-contact detection primitive (TF-79)

| Field | Value |
|---|---|
| **Date** | 2026-10-14 |
| **Status** | Accepted |
| **Deciders** | Karsten |

> **ADR number coordination (RESOLVED, owner ruling 2026-10-15).** TF-79 is **ADR-124** — the next free ADR
> number after the part-deux HF-jobs cycle (`origin/feat/hf-jobs-training-harness @ 6f6508a`, PR #284, reviewed,
> CI-green, merge-ready), which holds the two numbers below it. That commit is frozen and TF-79 was still
> force-amending its initial commit, so TF-79 took the next free number rather than disturb it. Release versions
> are independent: TF-79 = ADR-124 / v4.136.0; part-deux releases after TF-79 at v4.137.0. No part-deux-side edits.

## Context

Several features need the **first ball contact** (who, when) derived from tracking rather than taken from a
provider's event feed: TF-46 corner analysis needs the first contact after a corner restart; TF-46 Arm 2(c)
needs the GK position at first contact and the kick→contact displacement; the deferred PathCRF "missing
ball-touch" item needs a raw candidate-touch list to align against the event stream. A feasibility pilot found
that SkillCorner GI first-contact labels are possession-based and agreed with a tracking contact in only 55% of
a 29-corner sample, and a single-rule detector found a contact in only 49% of broadcast corners — so the
primitive must be tracking-derived, honest about non-detection, and shared (not re-implemented per consumer).

Owner ruling (2026-10-13): build ONE shared candidate-touch primitive (the single-implementation rule, cf.
ADR-055), not a detector private to TF-46. This is the candidate-touch stage only.

## Decision

Add a pure tracking primitive `silly_kicks/tracking/_ball_contact.py` that turns a ball trajectory (with `z`)
plus player positions into a per-contact table (`detect_ball_contacts`), with an additive action-grain enricher
for the first contact in an action window (`add_ball_contacts`) and a window query (`first_contact_in_window`).
Detection is a hybrid of RDP structural breaks and kinematic velocity breaks, with z-aware 3D proximity
attribution; "unresolved" is a kept NaN row (E7), never a guess.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Per-frame annotation on frames | simple attach | contacts are sparse; wasteful; re-derivation for window queries | lossy for the known consumers |
| B. Per-action enrichment only | one surface | no raw candidate list for PathCRF; no row for a corner window | too lossy for a shared primitive |
| C. **Per-contact table + action enricher (chosen)** | serves TF-46 window query AND PathCRF alignment without re-derivation | two public surfaces | — |
| velocity-break-only / RDP-only detection | single signal | each misses a motion signature (speed-only cushion vs. no-bend) | hybrid union covers both |
| lift PathCRF code | ready-made | **MPL-2.0 into an MIT library** | license-incompatible; reimplement the concept only |

## Consequences

### Positive
- A single first-contact source for TF-46 (both arms) and the deferred PathCRF item; no duplicate detectors.
- Honest-NaN throughout: unresolved contacts and `is_aerial` on absent/estimated `z` are explicit, not guessed.
- Pure, additive, no retrain; composes into the existing tracking-feature gates.

### Negative
- A new public surface (4 names) to maintain; a new `contact_source` vocabulary.
- RDP is implemented fresh (there was none in the library) — a small amount of new numeric code to own.

### Neutral
- Geometry columns register for reflection; the primitive itself does not reflect (TF-46 reflect-normalizes downstream).
- Validation is in-cycle but **reported, not gated** (no accuracy threshold), matching TF-46's "benchmarks not gates" stance.

## Related
- **Specs:** `docs/superpowers/specs/2026-10-14-tf79-ball-contact-detection-design.md`
- **Plans:** `docs/superpowers/plans/2026-10-14-tf79-ball-contact-detection.md`
- **ADRs:** depends on ADR-068 (no-rescan `group_rows`), ADR-073 (scale guard), ADR-078 (call convention),
  ADR-033 (purity), ADR-042 (keep-the-NaN-row), ADR-045/028 (reflection/mirror), ADR-053 (SB360 verdict),
  ADR-005 (NOTICE attribution), ADR-056 (liveness precondition), ADR-106 (float32 storage / float64 compute).
  Reuses the TF-48 (`_shot_goalmouth`) contact-radius/height priors; TF-48 itself is untouched (Chesterton's Fence).
- **External references:** Ramer 1972 / Douglas & Peucker 1973 (RDP, classical, reimplemented); PathCRF
  (KDD 2026, arXiv:2602.12080, MPL-2.0 — inspiration only, no code lifted). See `NOTICE`.

## Notes

Design decisions recorded here and enforced in code/tests:
1. Output shape = per-contact table + action enricher; **no `*_xfns`** (producer primitive, like TF-5/TF-48).
2. **Not a `METRIC_CONTRACTS` family** — it is a tracking enricher, not a `compute_*` aggregator mart. A later
   contacts-based aggregator mart would register there; this primitive does not. (Corrects spec §6, which first
   mis-placed `contact_source` in `TRACKING_CATEGORICAL_DOMAINS`, the frame-column domain registry; the gate
   `test_tracking_categorical_domains_keys_subset_of_columns` requires frame columns, so `contact_source` lives
   as a plain `category` dtype on the contacts table/actions instead.)
3. Detection parameters AND the validation match tolerance are FIXED/pre-registered (derived from physics +
   the TF-48 priors), checked only against the synthetic-recovery fixture — never selected against the accuracy
   corpus (a no-leakage rule).
4. `is_aerial` is honest-NaN where `z` is absent or provider-estimated (the measured-vs-estimated `z` question
   per provider is an open investigation recorded in the validation memo; IDSSE `z` confirmed MEASURED
   (continuous, var ≈ 3.2); SkillCorner `z` remains the open sub-item, handled by honest-NaN).
5. **Empirical revision (2026-10-15 real-corpus E2E; owner-directed before commit, gold standard).**
   Candidates are RANKED, never hard-dropped — a dropped real touch is unrecoverable downstream (NW alignment
   can't realign what isn't there), so recall is paramount at this candidate stage. The table carries
   `velocity_change_ms` (|Δv| touch-vs-flight signal: touch p50 ≈ 4.85 m/s vs flight-noise p90 ≈ 3.25 on IDSSE)
   and a `confidence` monotone in it; `min_velocity_change_ms` is an opt-in floor defaulting to 0 (no loss).
   Validation reframed from tight-window precision/recall to characterisation (signed offset distribution,
   agreement-vs-tolerance, confidence calibration, |Δv| separation), because SPADL event timestamps LAG the
   physical contact (~0.4 s median on IDSSE) — the event-feed misalignment TF-79/PathCRF exists to expose.
   **Provider caveat:** ball-contact detection is degraded on Gradient Sports — GS delivers a pre-smoothed
   (blunted) ball trajectory (contact |Δv| p50 ≈ 0.97 m/s), confirmed to be the provider's data, not
   silly-kicks preprocess (`smooth_frames` is additive; detect reads raw `x`/`y`). Validated on IDSSE
   (optical) + SkillCorner (broadcast).
