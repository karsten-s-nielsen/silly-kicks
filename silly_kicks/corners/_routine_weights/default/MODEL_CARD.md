---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - corner-kicks
  - tracking-data
library_name: silly-kicks
---

# Corner Routine Model v1 &mdash; TF-46 Arm 1

`CornerRoutineModel` is a non-negative matrix factorization (NMF) of corner-kick run vectors into a
basis of co-occurring run combinations (Shaw & Gopaladesikan playbook, TF-46 Arm 1). It is **bundled in
the `silly-kicks` wheel** (params-only, redistributable); there is no Hub repo.

## Model Description

Factors the per-corner `42`-column run vectors (from `RunZoneModel`) into a rank-**15** basis
(reconstruction-elbow selected over ranks 2&ndash;30) with `alpha = 0.0`. `transform` projects a
corner's run vector onto the frozen basis via a pure-numpy non-negative multiplicative-update solve, so
`load()` imports no sklearn. sklearn `NMF` at **fit only**.

Unlike `RunZoneModel`, this model carries **no chirality fingerprint**: it factors
orientation-neutral run vectors (the reflection / y-handedness is already resolved upstream in
`RunZoneModel`), so the ADR-037 y-mirror class does not apply.

## Training corpus

Pooled multi-provider: SkillCorner + IDSSE/Sportec + StatsBomb SB360. 2702 direct corners. Only the
learned basis is stored &mdash; no raw provider tracking data. Reconstruction error decreases 52.9 to
1.86 across ranks 2&ndash;15 (the elbow), confirming a non-degenerate basis.

## Provenance

**Provenance.** `training_commit=6ee7e62`. Fit on a clean tree, enforced by the fit driver's
`require_clean_tree` (ADR-037).

## Construct validity

The construct-validity memo is `docs/research/corner_routines/` (README + metrics.json). The routine
features participate in the net-of-team discrimination check (routine fixed-effect leg). This is an
**instrument of corner routine structure, not a which-routine-wins ranking** (that is Arm 3).

## Integrity and load-time guards

`load()` is fail-closed on **SHA256SUMS**, verified before the basis is parsed; a tamper raises
`IntegrityError`.

## Usage

```python
from silly_kicks.corners import CornerRoutineModel

model = CornerRoutineModel.load("silly_kicks/corners/_routine_weights/default")
coeffs = model.transform(term_matrix)   # (n_corners, rank), non-negative
```

## Limitations

- Descriptive / unsupervised: the basis is a mechanism decomposition, carrying no accuracy claim.
- Archetype naming / which-routine-beats-which ranking stay consumer-side (ADR-110).

## Model Files

| File | Purpose |
|---|---|
| `model.npz` | NMF basis H (pickle-free) |
| `metadata.json` | rank, run count, alpha, selection curve, provenance |
| `SHA256SUMS` | integrity manifest, verified by `load()` |
| `MODEL_CARD.md` | this card |

## References

See the `NOTICE` file in the silly-kicks repository. Attribution: Shaw & Gopaladesikan, "Routine
Inspection" (42 Analytics). Decisions: ADR-011 (trained-model lifecycle), ADR-110 (raw primitives).

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
