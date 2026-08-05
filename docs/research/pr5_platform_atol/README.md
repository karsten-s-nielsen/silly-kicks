# Feature-contract fingerprint: x86 vs aarch64

Is `_feature_contract._CONTRACT_ATOL = 1e-6` / `_CONTRACT_RTOL = 0.0` — the tolerance on the
fail-closed guard for **every** bundled model (ADR-050) — wide enough to survive a cross-platform
recompute?

**Measured: yes, with room to spare.**

    max_abs_delta_overall  0.0        status  ok
    69 features across all three bundled contracts
    run_commit  35a66679c7bd   run_tree_dirty  false

| leg | machine | system | python |
|---|---|---|---|
| a | AMD64 | Windows | 3.14 |
| b | aarch64 | Linux | 3.12 |

Per model: ghost-GK 26 features, xShot 27, xCross 16 — **all zero delta**.

## Why this replaces a hand-run predecessor

The earlier version of this directory was produced by hand. It covered **27 of 69** features (xShot
only), and carried **no `run_commit` and no `run_tree_dirty`** — so it could not be cited by the very
rule it was meant to support, and its scope did not match what it was quoted for. It was removed
rather than retro-stamped, because adding a `run_commit` to a hand-run file is the same failure as
restamping one without re-running.

## Scope is an AND, and that mattered

All three contracts are probed, not just xShot. Ghost-GK and xShot were already *empirically*
aarch64-clean — `validate_xs_probe.py` constructs both, routing through `load()` ->
`verify_feature_contract`, and completed a 64-match DGX run. **Nothing else in this cycle loads
`XCrossAttemptModel` on aarch64**: `validate_xcross_causal.py` reads `metadata.json` directly and
never constructs the model, while `tracking/features.py` loads it via `from_variant("default")` for
`xcross_attempt_xfns`. It was the one contract-bearing artifact never loaded on aarch64 and it is
reachable from a live public path, so an xShot-only measurement would have left exactly the unverified
surface unverified.

## `--compare` refuses, it does not merely report

A delta between legs run at different commits would confound **platform** with **code** — the one
thing this artifact exists to separate. The comparison refuses unless both legs agree on `run_commit`,
`geometry_version` and probe identity, and both carry `run_tree_dirty: false`.

## Caveats that travel with the number

**The two legs confound architecture with interpreter** (AMD64/Windows/3.14 vs aarch64/Linux/3.12).
No third leg disentangles them. Read `0.0` as "this pair agrees", not "architecture is irrelevant".

**`atol` cannot transfer to the quantized xCross features even in principle.** `space_controlled` is
`cell_count / 805 * 7140`, quantized at ~8.8696 m² per cell — about 8.87e6 x `atol` — so its
cross-platform error is exactly `0.0` or `>= 8.87`, and the tolerance degenerates to an equality test
on it. `box_off_def_ratio` is likewise an integer ratio. The clean result here means those features
landed on the same cell on both platforms, not that they were compared within a tolerance.

**This measures the contract fingerprint only.** It says nothing about the ghost-GK re-fit's own
acceptance, which spec §6 makes PR 7's item. Do not read a clean result here as discharging it — the
predecessor's README made exactly that over-claim, asserting that "PR 7's ghost re-fit can stamp on
x86 and verify anywhere" on the strength of an xShot-only probe.

## Consequence

`atol=1e-6` / `rtol=0` **stands**, and fingerprints do **not** need to become platform-scoped. The
branch ADR-050 §1 warned about — a covering tolerance that would also swallow a real 1 cm geometry
change — is not the branch we are on.
