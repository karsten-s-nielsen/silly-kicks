# ADR-108: Periodic angular quadrature for native DAS

| Field | Value |
|---|---|
| **Date** | 2026-09-26 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

The native DAS engine (ADR-107) integrates a per-(angle, radial) density field over a polar grid. The
`accessible-space` 2.0.15 reference builds the angular grid with
`phi = np.linspace(phi_offset, 2π + phi_offset, n_angles, endpoint=False)` and then integrates each
ray over an angular wedge whose bounds are the **midpoints to its neighbours** — but it computes those
midpoints with a plain (non-wrapping) neighbour difference. The first and last rays therefore get a
**half-width wedge** (there is no neighbour "before" ray 0 or "after" ray n−1 in the array), so a
fixed slice of the circle is systematically under-weighted. Because that under-weighted slice sits at a
fixed absolute orientation, the resulting DAS is **direction-biased**: mirroring a scene (a 180° point
reflection, which must leave DAS invariant) changes the value.

Measured (probe, synthetic): the reference's mirror gap on S01↔M01 is **~441 m²**, versus **~4e-12** for
a periodic grid. The per-value shift from fixing it is a **median ~1.6% / p90 ~20%** on synthetic
scenes. This is a genuine correctness defect, not a numerical nicety.

## Decision

The shipped native quadrature is **periodic**: the two end rays' wedge bounds wrap around the circle.

- Interior bounds are the reference's neighbour midpoints (so interior `d_area` is **bitwise identical**
  across modes).
- End rays use the wrap-around midpoints: `phi_lower[0] = (phi[-1] − 2π + phi[0]) / 2`,
  `phi_upper[-1] = (phi[-1] + phi[0] + 2π) / 2` — restoring the missing half-wedges so the full circle
  is covered exactly once.
- `dr` and `d_area` are kept **separate** (the engine multiplies `(field · dr) · dA`; a premultiplied
  weight would change the association and is not what the reference does).

The **reference quadrature survives only as an internal parity-gate mode** (`quadrature="reference"` on
`PassSimParams`): it lets `test_das_engine_parity.py` prove the native engine reproduces the library
byte-for-byte (Δ=0 at `rtol=atol=1e-12`) under the same (defective) weights, isolating "did we
reimplement the physics correctly?" from "did we fix the quadrature?". It is **refused on the public
surface** (`get_das`/`get_individual_das`/`get_xc` raise `ValueError` on `quadrature="reference"`).

The exact-relation gate (`test_das_quadrature.py`) pins that the periodic-vs-reference difference equals
exactly the end-ray weight difference applied to the quadrature-independent integrand, and that periodic
DAS is mirror-invariant (`rtol=1e-6`, the `arccos` float floor — F1, §6.4) where the reference is not
(`> 1.0` relative gap).

## Alternatives considered

| Option | Why rejected |
|---|---|
| Ship the reference quadrature (reproduce the library exactly) | ships a direction-biased metric; mirroring a scene changes DAS by hundreds of m² |
| Premultiply `dr · d_area` into one weight | changes the `(field · dr) · dA` association and diverges from the reference's own multiplication order used by the parity gate |
| Drop the reference mode entirely | loses the byte-for-byte parity proof that the physics reimplementation is faithful — the reference mode is the oracle for that |
| **(chosen) periodic end-ray wrap-around; reference mode parity-only, refused publicly** | correct (mirror-invariant), and still fully parity-proven against the library |

## Consequences

### Positive

- DAS is mirror-invariant: a 180° point reflection leaves the value unchanged (to `rtol=1e-6`, the
  `arccos` float floor — F1, §6.4), as a spatial area must be.
- The physics reimplementation stays byte-for-byte verifiable against the library via the reference
  parity mode.

### Negative

- Every DAS value changes versus the old `accessible-space` output (periodic quadrature + the ADR-107
  GoalMap direction together). The **owner-corpus** shift (median / p90 / max on real matches) is
  measured by `scripts/validate_das_native_parity.py` and **quoted here at commit 2**, with the
  downstream re-materialize notice (calibration `das_*` features, gkdv ΔDAS arm).

<!-- COMMIT-2 PLACEHOLDER: owner-corpus DAS shift from docs/research/das_native_parity/metrics.json
     (median / p90 / max, per provider); the CHANGELOG Hyrum block quotes the same figures. -->

### Neutral

- The mirror-registry tolerance for `add_das` becomes a RELATIVE `rtol=1e-6` (F1, §6.4;
  `relative_tolerance=True` on the entry + the Gate A rtol branch in `_mirror_registry.py`), replacing
  the old absolute `_DAS_MIRROR_TOL=200.0` that sized around `accessible-space`'s non-equivariant polar
  grid. The residual is the `arccos`-limited float floor (~1e-7 relative), which scales with the DAS
  magnitude, not a quadrature asymmetry. `add_das` also joins Gate C now (`call_with_map` /
  `gate_c_must_move`), since direction comes from a `GoalMap` (ADR-055) the entry can vary.
- **The parity oracle runs in a pinned pandas-2 subprocess, and must keep doing so.** The owner-corpus
  shift above is measured against `accessible-space==2.0.15`, which is a pandas-2-era library: under
  pandas-3 Copy-on-Write its internal `PLAYER_POS` array is read-only, so its in-place offside step raises
  a `ValueError` the library catches ("Ignoring offside"), silently keeping offside attackers and inflating
  team DAS; and it pivots on `frame_id` alone, conflating frames that reuse a `frame_id` across periods.
  `scripts/_das_reference_leg.py` therefore runs the library under a separately provisioned pandas-2
  interpreter (`SK_DAS_REFERENCE_PYTHON`, the dev-only `das-reference` extra pins `pandas<3`) and feeds it a
  collision-free frame key (a dense rank over `game_id`/`period_id`/`frame_id`). The native engine is immune
  to both. Do not "simplify" the subprocess away: that silently restores the broken comparison
  (pandas-2 reference spec §7). The leg also reads player results by input row, forwards the ball
  carrier, as native uses it, and hands the library frame-sorted rows so each frame keeps its own carrier
  (ADR-107).

## References

Spec: `docs/superpowers/specs/2026-09-26-das-native-design.md` §6.4. Engine: ADR-107. Grid construction:
plan Appendix A.1. Mirror invariance: ADR-045 (reflection), the `test_mirror_registry` Gate.
