# ADR-031: kloppy tracking-gateway y-axis inversion (CS-pin) + DFL parse-port single-sourcing

| Field | Value |
|---|---|
| **Date** | 2026-06-16 |
| **Status** | Accepted (PR-S94 / T1 = CS-pin, shipped here; PR-S95 / T3 = parse port, planned; T2 = no-op) |
| **Deciders** | Karsten (with Claude); lakehouse session (cross-session review rounds 2–6) |

## Context

The silly-kicks kloppy **tracking** gateway (`tracking/kloppy.py`) produced frames with a y-axis
**inverted** relative to the SPADL action y-axis: for the same physical point, `action_y == 68 − frame_y`,
for every kloppy-based provider (SkillCorner, Metrica, and the IDSSE dev-harness path). It is a
**single-axis y mirror**, NOT an orientation issue — ADR-028 (per-action LTR reprojection), ADR-029
(`orient_frames_to_ltr`), and `play_left_to_right` are all 180° point reflections or identity, so
re-orientation cannot rescue it. The error magnitude is `|68 − 2y|`: zero at pitch centre (y=34),
growing to ~full pitch width at the touchlines — which is why it hid (the liveness gate is non-null
only; the GK-roster e2e is x-based; synthetic mirror-invariance fixtures are self-consistent).

**Root cause:** the **event** gateway (`spadl/kloppy.py`) transforms with
`to_coordinate_system=_SoccerActionCoordinateSystem(...)` (origin `BOTTOM_LEFT`, vertical
`BOTTOM_TO_TOP`) → canonical; the **tracking** gateway passed only `to_pitch_dimensions` +
`to_orientation=HOME_AWAY` and **never** pinned the coordinate system, so frames retained each
provider's kloppy-native vertical. The un-pinned transform dates to PR-S19 (v2.7.0) — an
asymmetry/oversight, not a deliberate guard.

Found during TF-48 (PR-S93). Spec: `docs/superpowers/specs/2026-06-15-kloppy-tracking-y-fix-design.md`
(rev-4-final, 6 cross-session review rounds). Report: `docs/research/bug_kloppy_tracking_y_inverted.md`.

## Decision

One spec, **sequenced PRs**. **PR-S94 (this ADR's shipped scope) = T1.** PR-S95 (T3, parse port) and
T2 (no-op) are recorded here for the decision trail.

1. **Shared coordinate-system extraction.** `_SoccerActionCoordinateSystem` moves to
   `silly_kicks/spadl/_kloppy_coordinates.py` with a `socceraction_coordinate_system(metadata)` helper;
   both gateways import it (DRY — events and frames cannot drift). The event path is **byte-identical**:
   the helper reads the same `metadata.coordinate_system.pitch_length/pitch_width` the inline
   construction did.

2. **CS-pin the tracking gateway — CS-ONLY signature.** `dataset.transform(to_orientation=HOME_AWAY,
   to_coordinate_system=socceraction_coordinate_system(...))`, **dropping `to_pitch_dimensions`**.
   **Gate 0 (SkillCorner, ~220k player-frames) proved this matters:** passing `to_pitch_dimensions`
   alongside the CS silently **overrides** the CS's vertical orientation → y stays inverted (candidate
   A is a *silent non-fix*). CS-only reproduces `(x, 68−y)` exactly with x byte-identical (the CS
   carries standardized 0–105/0–68 dimensions). This is the exact call form the event gateway uses.

3. **NOT a blanket `y = 68 − y` flip.** A blanket flip would double-invert an already-canonical
   provider. The CS-pin is a **no-op on canonical input** (kloppy applies each provider's
   native→canonical flip; identity if already canonical). Guarded by
   `tests/tracking/test_kloppy_cs_pin_noop_canonical.py`.

## Gate verdicts (all on real data, DGX)

- **Gate 0** — signature = CS-only (candidate A is a silent non-fix). `scripts/_tf48_cspin_equiv.py`.
- **Gate A** — Metrica was y-**inverted like SkillCorner**; the CS-pin flips it to canonical → the
  **Metrica calibration retrain trigger fires** (not a no-op). `scripts/_tf48_gate_a_metrica.py`.
- **Gate B** — full-match SkillCorner post-fix action↔shooter = **identity (0.16 m, n=497)**.
- **Gate D (T2)** — the **native** sportec adapter (`tracking/sportec.py`) is y-**correct**, confirmed
  per period (P1+P2, exact). Production IDSSE is **not broken**; T2 has no fix and no retrain. ET
  (P3/P4) unverified (no ET in the IDSSE set). `scripts/_tf48_gate_d.py`.
- **Gate C** — the lakehouse builds SkillCorner/Metrica frames via its **own** `convert.py` builders
  (`_bronze_metrica_to_frames`, `_bronze_skillcorner_to_frames`), **not** the silly-kicks kloppy
  gateway. So **PR-S94 fixes the calibration/pining path + external gateway consumers, NOT lakehouse
  production.** The lakehouse owns a separate y-check on its builders (asymmetry: Metrica flips y,
  SkillCorner does not; neither y-guarded) — the Gate-C handoff.

## Blast radius (calibration / external gateway consumers)

CORRUPTED (now fixed): `add_action_context` (`nearest_defender_distance`, …), `add_pressure_on_actor`
(incl. vy), `add_pre_shot_gk_*` distances/angles, `add_shot_goalmouth`. **Measured A/B refinement**
(`tests/tracking/test_y_blast_radius_ab.py`): under a frames-only y-flip, `nearest_defender_distance`,
`pre_shot_gk_distance_to_shot`, `pre_shot_gk_y` change; `actor_speed`, `pre_shot_gk_x`, and
**`pre_shot_gk_distance_to_goal` do NOT** — the last because the goal is at centre y=34, so the
distance is y-symmetric (correcting an over-listing). Frame-integrated / x-only aggregates
(`team_shape`, `defensive_line`) are isometry-immune. **NOT affected:** native sportec/IDSSE
(Gate D), Gradient Sports native, event-only providers (StatsBomb/Wyscout/Opta).

## PR-S95 (T3, planned) + the C4 release-coupling trade-off

PR-S95 single-sources the IDSSE/Sportec **parser** via a `silly_kicks/providers/sportec/parse.py`
parse+shape port (behind a `[parse-dfl]` extra), eliminating the **four-layer** dev/prod drift
(parse / smooth / velocity / convert); the lakehouse adopts the port and deletes its private parser.
**Trade-off (chosen):** once the lakehouse depends on `silly-kicks[parse-dfl]`, every future
DFL-parser change routes through a silly-kicks release (PyPI → wheel → terraform `==` pin → the
lakehouse's own dep-parity check) instead of a one-repo patch — the right hexagonal direction
(drift-elimination) at the cost of lakehouse change-latency. Data-quality (smoothing, velocity)
stays consumer-side, composed explicitly, with a single shared canonical callable.

## Alternatives rejected

- **Blanket `y = 68 − y`** — double-inverts already-canonical providers.
- **Candidate A** (keep `to_pitch_dimensions` + add the CS) — silent non-fix (Gate 0).
- **kloppy-shim for T3** (single-source only the converter) — leaves parser/smooth/velocity drift;
  PR-S95 uses the parse port instead.

## Consequences

- **Retrain (scoped):** VAEP + tracking **calibration** consumers for SkillCorner **and** Metrica
  (both inverted, both fixed). Lakehouse re-materializes only if its own builder check (Gate-C
  handoff) finds a bug — a separate lakehouse PR.
- New module `silly_kicks/spadl/_kloppy_coordinates.py`; the event path is byte-identical. **No new
  action-coupled aggregator → the tracking C4 aggregator count is unchanged (28).**
- References (silly-kicks): ADR-004 (gateway split / native adapters), ADR-006 (`output_convention`),
  ADR-019 (id-dtype seams), ADR-028 / ADR-029 (orientation — explicitly distinguished from this
  single-axis y mirror). The lakehouse's own ADR-029 (ET guard) and ADR-046 (terraform dep parity)
  are distinct documents in the lakehouse repo.
