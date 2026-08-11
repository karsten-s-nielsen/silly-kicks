# ADR-059: A detector must require DISCRIMINATING evidence, not merely absent counter-evidence

| Field | Value |
|---|---|
| **Date** | 2026-08-11 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

> **Numbering note.** 056 is Cycle B. 057, 058 and this one all ship together in 4.79.0, so no
> renumbering hazard applies — the parallel session that forced two renumbers in two days has
> merged, and this repo now has a single writer.

## Context

`detect_input_convention` infers a provider's coordinate convention from the per-`(match, team,
period)` distribution of shot x-positions. Rule 1:

> Every reliable group has mean x > x_max/2 → `POSSESSION_PERSPECTIVE`.

`reliable` is the subset of groups with `n >= min_shots_per_group_medium` (5).

**Measured on real Gradient Sports data, this returns a CONFIDENTLY WRONG verdict on 2 of 36
matches.** Match 10502 (raw `ball_x` + 52.5, shots only):

| team | P1 | P2 |
|---|---|---|
| 51 | 13.3 LOW (n=3) | 94.0 HIGH (n=8) |
| 366 | 95.8 HIGH (n=7) | 9.8 LOW (n=3) |

Both teams shoot at opposite ends within a period and SWAP between periods — textbook
`PER_PERIOD_ABSOLUTE`, exactly what `gradientsports.py` declares, and consistent with the
independent finding that its output is correct to ~0.2 m. But the `>= 5` filter drops **both LOW
groups**, leaving an all-HIGH survivor set, and rule 1 concludes `POSSESSION_PERSPECTIVE` with
`confidence="medium"`.

### The prescribed fix was wrong, and looked right

The `TODO.md` row that recorded this defect diagnosed it as *"the rule fires on effectively ONE
team's data"* and prescribed: **require ≥2 distinct TEAMS among the reliable groups.**

**Measured: the survivor set spans TWO teams** (51 in P2, 366 in P1), so that guard permits the
misfire unchanged. The prescription would have shipped, reviewed clean against its own stated
rationale, and left the defect live.

The symptom was miscounted because the failure is not about *how many teams* survive. Under
`PER_PERIOD_ABSOLUTE`, "team 51 attacks high in P2" and "team 366 attacks high in P1" are exactly
what you expect — each team swaps, and the observations that would reveal it are the two the filter
removed. **The evidence does not discriminate between the hypotheses.**

## Decision

**Rule 1 may fire only when the reliable set contains a configuration that an absolute convention
could not have produced.** Two such configurations exist, and either suffices:

- **(a)** two distinct teams reliable in the SAME period, both high — impossible under an absolute
  convention, where they attack opposite ends;
- **(b)** one team reliable in ≥2 periods, high in both — impossible under a swapping convention.

When all groups are high but neither holds, return `convention=None` with a diagnostic naming the
reason, rather than guessing.

**Deferral is the safe direction, and the asymmetry is the whole argument.**
`validate_input_convention` reads `None` as *"signal too weak — keep the caller's declared
convention"*:

| Error | Consequence |
|---|---|
| **False positive** (the defect) | Contradicts a CORRECT declaration → warns, or **raises** under `on_mismatch="raise"`. Rejects good data. |
| **False ambiguous** (risk of the fix) | Defers to the correct declaration. **Output unchanged and correct**; only a cross-check is lost. |

A false ambiguous cannot produce wrong geometry. A false positive already does.

### One spelling of the shared predicate, not two

Clause (b) **is** the guard TF-22 added inline to the ABSOLUTE branch in 3.0.1 — `per_team_periods
>= 2` — for the same reason, against the same class of sparse asymmetric data (IDSSE J03WMX). Rule
1 never got it. Both branches now call `_a_team_spans_periods`.

Clause (a) is deliberately **NOT** given to the ABSOLUTE branch. Separating `ABSOLUTE` from
`PER_PERIOD` requires observing a team *across* periods; two teams inside one period says nothing
about swapping. Extracting a single "is this evidence discriminating?" helper for both branches was
the first design considered and is wrong — it would silently loosen TF-22. The primitives are
shared; the composition is per-branch, because the hypotheses being separated differ.

## Consequences

**Behaviour.** `POSSESSION_PERSPECTIVE` is returned in strictly fewer cases. The ABSOLUTE branch is
unchanged (same predicate, one spelling). Gradient Sports stops being contradicted on the 2 affected
matches; every provider whose data carries a discriminating configuration is unaffected.

**The coverage risk, and what actually protects against it.** Rule 1 is the branch validating
StatsBomb and SkillCorner as `POSSESSION_PERSPECTIVE`, so tightening it risks a silent downgrade to
ambiguous — a loss that shows as a gate quietly not checking rather than a red test.

`tests/invariants/test_input_convention_detector.py::test_statsbomb_raw_detected_as_possession_perspective`
already asserts exactly this on **3 real StatsBomb matches**, and it passes. That is the standing
detector, and it is worth more than a one-time corpus run, which confirms today and rots tomorrow.

**SkillCorner is guarded too**, by `test_skillcorner_raw_detected_as_possession_perspective` on a
real public match.

That gate was very nearly not built. A first reading concluded SkillCorner was owner-tier and
therefore non-redistributable, so the risk was recorded as an unclosable gap. **That was wrong, and
wrong in the manner `scripts/_corpus.py` exists to prevent:** its docstring states visibility is
keyed per-match, *"NEVER on the provider name"*, and describes deleting an allowlist that used
`skillcorner` as a tier proxy. The conclusion was drawn from that very docstring. `PUBLIC_CORPUS`
registers **ten** redistributable SkillCorner matches, and each pining record carries its own
`visibility` field reading `public` — two lookups, either of which would have settled it.

The fixture (match `1886347`, 5079 event rows, five columns, no player identifiers) is deliberately
a HARD case rather than a convenient one: `(team 1805, period 1)` has only 3 shots and is dropped by
the `>= min_shots_per_group_medium` filter — the exact sparse-drop shape that made rule 1 misfire on
Gradient Sports. It still classifies, because the survivors discriminate (1805 and 4177 are both
reliable in period 2, and 4177 spans both periods). So the gate demonstrates the new guard does not
over-tighten on real data carrying the defect's own shape.

`is_shot` is re-derived in the test from `end_type == "shot"` — the converter's own rule — rather
than baked into the committed file, so the fixture cannot silently disagree with the converter about
what a shot is. A companion test asserts the fixture still contains a below-threshold group, because
a fixture swap that removed it would leave the gate passing while no longer exercising the filter
behaviour it exists to cover.

**Mutation-verified:** with both discriminating predicates forced false — the silent-downgrade
regression this section names — real SkillCorner data detects as `None` and the gate fails.

## The durable lesson

**"No counter-evidence" is not evidence.** A rule of the form *"every observed X has property P,
therefore convention C"* is only sound if the observations COULD have shown otherwise. When a
filter removes the disconfirming cases, an all-confirming sample is an artifact of the filter, not
a measurement — and the rule reports high confidence precisely because it saw no dissent.

Two review-level corollaries:

1. **When a filter precedes a universal claim, ask what the filter removed.** Here `>= 5` shots
   removed exactly the groups that would have falsified the conclusion, and did so *more often on
   sparse matches* — which is why it presented as noise (2 of 36) rather than as a systematic bug.
2. **A diagnosed mechanism in a bug report is a hypothesis, not a finding.** This row carried real
   measured data and still misidentified the mechanism. Reproducing it before implementing cost ten
   minutes and was the only reason the prescribed guard was not shipped.
