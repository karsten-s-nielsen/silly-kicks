# TF-19 xS-arm GK-substitution probe — v1 and v2

This directory holds **two runs of the same registered protocol**. Both are kept: the first is the
result cited in ADR-037 and TODO.md; the second exists because that first artifact's provenance
cannot be verified.

| File | Run | Provenance |
|---|---|---|
| `metrics.json` / `report.md` | 4.60.0 (PR-S131), 2026-07-26 | `run_commit 78ffc70`, **no `run_tree_dirty` field** |
| `metrics_rerun_clean_provenance.json` / `report_rerun_clean_provenance.md` | 2026-07-28 | `lock_commit 78ffc70`, `run_commit d1fc18d`, **`run_tree_dirty: false`** |

## Why the re-run

The original stamped its commit with a bare `git rev-parse HEAD`, which returns the **same SHA
whether or not the working tree is modified**. So the recorded commit need not describe the code
that produced the numbers — verifiable-looking and false, and unrecoverable after the fact. Owner
approved the re-run on 2026-07-27; `validate_xs_probe.py` was wired to the fail-closed provenance
guard in 4.65.0.

`lock_commit` stays **`78ffc70`** in both. That is the load-bearing field for the blindness claim:
the v2 placebo pool and its constants were frozen on 2026-07-23, before any v2 data was seen. The
run commit is merely where it executed.

## Result — the re-run REPRODUCES the original

```
v1: no_valid_placebo → unmeasurable_at_dose   placebo_p95 0.0        (66.5% zero)
v2: pass             → joins_with_caveat      placebo_p95 0.00057    (44.7% zero)
```

64 GradientSports matches (GS-only by the GKDV measurement rule — `_PROVIDERS` is hard-coded, not a
CLI knob), 123,430 frames used from 123,430 targets (`targets_to_used_drop_frac: 0.0`, i.e. no
silent band shrinkage), 10,180 frames in the gated 2 m band, 64 distinct games.

**v1 reproduces the degenerate-placebo finding** that PR-3b hit: a random-outfielder null with
`placebo_p95 = 0.0` cannot certify anything, and that gate short-circuits before the dose-response
ever runs. **v2's relevance-matched defender pool clears the instrument-validity gate** and the
verdict stands.

## What this changes for TF-19

Nothing about the verdict; everything about its basis. `joins_with_caveat` now rests on two
**measured** inputs rather than one measured and one defaulted:

- the xS probe verdict: `pass` (here, clean provenance)
- `entanglement`: `inside_band` (see `../tf19_entanglement/`, previously a registered default — ADR-037 F6)
