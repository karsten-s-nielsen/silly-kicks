# ADR-060: TF-24 recommends an indistinguishable set under prefer-incumbent selection; `tolerance_m` is a held constant

| Field | Value |
|---|---|
| **Date** | 2026-08-14 |
| **Status** | Proposed |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8); cross-reviewed by a second session |

> The ADR number is a placeholder until commit-prep — confirm it is still free against `origin/main`
> before numbering ([[no-version-number-until-commit-prep]]).

## Context

TF-24 Stage 1 calibrates ball-carrier inference (`tolerance_m`, `beta`, `gamma`) by maximizing
carrier-actor accuracy. Two measured properties make "report one best point" misleading:

1. **`beta`/`gamma` are non-identifiable.** The Phase-1 confirmation (`abe9f94`) found the six-point
   candidate spread ≈ 1/40 of one CV standard error, three points winning across five folds,
   `argmax_moved = False`. An argmax over indistinguishable points recommends noise and churns.
2. **`tolerance_m` is under-determined.** The objective's labels are on-ball moments only, with no
   loose-ball negatives, so a larger radius only helps the metric and the sweep presses `tolerance_m`
   to its upper bound (a re-sweep store landed `7.999`). The radius the sweep "finds" is an artifact of
   the label design. The shipped default already holds `3.0` for this reason
   (`_ball_carrier.py:358–362`); the search space was never updated to match (it has swept `tolerance_m`
   since TF-24's first commit `0a76f52`, when the TF-5 spec intended to calibrate all three).

Downstream, the raw sweep record `carrier_best.json` can assert `tolerance_m ≈ 7.999`, and Stage 2
consumes `{tolerance_m, beta, gamma}` from it into a held-out Brier objective
(`calibrate_tracking_defaults.py:327–348`).

TF-24's standing rule (ADR-009) — recommend, never change library constants — is preserved.

## Decision

TF-24's Stage-1 output is the **indistinguishable set plus a `prefer-incumbent` selection over
`beta`/`gamma`**: the recommendation stays the shipped default (`beta=0, gamma=0.25`) unless a
candidate clears **both** a practical-significance **effect-size floor** and a **paired-difference SE**
significance test. **`tolerance_m` is a held constant, enforced so a wrong radius is unrepresentable
rather than merely refused:** it is removed from `stage1_config`'s search space and
`CarrierAccuracyObjective` (a sweep never produces it), excluded from the selection artifact, and
sourced by Stage 2 from `DEFAULT_CARRIER_PARAMS`. The selection artifact
is committed, provenance-stamped, and validated by Stage 2 (missing manifest == dirty → refuse). There
is **no `tolerance_m` re-sweep** and the loose-ball-negatives work that could identify the radius is
**not pursued**.

## Alternatives considered

### Selection policy

| Option | Why rejected / chosen |
|---|---|
| Argmax over the set | recommends noise; churns run-to-run — the defect this removes |
| Most-central point | the centre is arbitrary; no principled tie-break |
| **Prefer-incumbent (chosen)** | moves only on evidence past both bars; train/serve stability; same "clears the floor" philosophy as `tf25_gate_fires` |

### The "move" criterion (SE)

| Option | Why rejected / chosen |
|---|---|
| Marginal SE (today's `moved_beyond_noise`) | statistically non-standard for a paired design (all points scored on the same folds); conservative by accident, not by design |
| Paired-difference SE **alone** | correct estimator, but its SE is tiny, so a practically-trivial (≈4e-4) gain clears it and the recommendation churns — the opposite of the intent |
| **Effect-size floor + paired SE, both required (chosen)** | the two answer different questions ("big enough to matter?" vs "real?"); for a non-identifiable parameter only the effect-size floor prevents churn, and the paired SE guards a large-but-noisy gain. The biostatistics "practical *and* statistical significance" standard, right for a standing diagnostic |

### `tolerance_m` at the Stage-2 boundary

| Option | Why rejected / chosen |
|---|---|
| Fail-closed assertion (`tolerance_m == 3.0`) | still lets a wrong radius be *represented* then refused |
| Hard-hold + warn | silently corrects a wrong input |
| **Exclude it from the artifact; source from the constant (chosen)** | a field that does not exist cannot be wrong — unrepresentable > refused > silently-corrected |

### `tolerance_m` upstream

| Option | Why rejected / chosen |
|---|---|
| Writer-pin only; track the search-space removal | closes the downstream trap but leaves the sweep still producing a meaningless best — a decision-in-disguise once breaking changes are free |
| Leave it entirely | `carrier_best.json` keeps asserting a meaningless radius |
| **Remove from the search space + objective in item C (chosen)** | breaking changes are not a concern; safe because item C's recommendation does not depend on the change (reuses the existing store; prefer-incumbent keeps the shipped default), so no fresh sweep is needed to land it |

## Consequences

### Positive

- The recommendation stops churning on noise and cannot smuggle an under-determined `tolerance_m` into
  Stage 2 (it is unrepresentable there).
- One "clears the noise floor" definition (`exceeds_noise_floor`) shared by the selection and
  `tf25_gate_fires`.
- The move criterion states what it means: practical significance guarded by statistical significance.
- A standing fold-stability diagnostic makes "no discriminating evidence" a reported verdict.

### Negative

- Prefer-incumbent keeps the shipped default even when a candidate is nominally higher within noise —
  intended, but a reader unaware of the non-identifiability may read it as ignoring a better point.
- `tolerance_m` remains uncalibrated; the honest fix (loose-ball negatives) is deferred indefinitely.
- `min_effect_size` (δ) is a new constant requiring a justified, **frozen** value (derived from Stage-2
  Brier sensitivity; a pre-land item). **On current data δ is the binding bar:** `beta`/`gamma` barely
  move the metric, so `paired_se ≈ 0` and the paired-SE test is near-decorative today — it protects the
  *future* large-but-noisy regime, not the current keep-incumbent result. The landed result is
  de-risked by asserting invariance to δ across a plausible range (spec §7).

### Neutral

- Two carrier artifacts: `carrier_best.json` (raw sweep record, now `{beta, gamma}` — `tolerance_m` no
  longer swept) and `carrier_selected.json` (the `{beta, gamma}` recommendation of record, provenanced).

## Related

- **Specs:** `docs/superpowers/specs/2026-08-14-tf24-item-c-recommendation-honesty-design.md`
- **ADRs:** refreshes ADR-009 (recommend-only preserved); builds on ADR-052 (provenance-stamped
  drivers), ADR-056 (artifact-population gate).
- **Code:** `silly_kicks/tracking/_ball_carrier.py:358–362`; `silly_kicks/calibration/_spaces.py:41`
  (introduced `0a76f52`); `scripts/check_stage1_argmax.py`;
  `scripts/calibrate_tracking_defaults.py:327–348`.
- **In scope (this ADR):** `tolerance_m` removed from `stage1_config`'s search space +
  `CarrierAccuracyObjective`. A future fresh sweep will re-confirm `beta`/`gamma` at the held radius,
  but the recommendation is robust to it (prefer-incumbent + non-identifiability).

## Notes

The Phase-1 confirmation at `abe9f94` recorded `argmax_moved=False` and the six-point spread of ≈ 1/40
of one CV SE. That run is the evidence this ADR rests on; §7 of the spec requires confirming *which
store* it read before this ADR moves from Proposed, since the headline evidence is a property of that
store.
