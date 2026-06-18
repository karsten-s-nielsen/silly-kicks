# ADR-034: SkillCorner + Metrica bronze→frame builders single-source the coordinate/orientation/clock truth

| Field | Value |
|---|---|
| **Date** | 2026-06-18 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (cross-session lakehouse review, 3 rounds) |
| **Supersedes (in part)** | ADR-029 ("decided against a native metrica/skillcorner.convert_to_frames") |
| **Harmonizes with** | luxury-lakehouse ADR-053 (geometric frame-LTR net) |
| **Extends** | ADR-031 (cross-repo bronze contract + delete-and-depend) |

## Context

silly-kicks and the luxury-lakehouse independently maintained **three** copies of the
SkillCorner/Metrica bronze→frame coordinate transform: the silly-kicks kloppy gateway
(the oracle), and two lakehouse builders — `analytics/action_context/convert.py`
(the live `fct_action_context` mart) and the legacy `ingestion/tracking_context.py`.
The lakehouse copies exist because the lakehouse holds **bronze DataFrames**, not
kloppy `TrackingDataset` objects, and cannot feed the gateway. That triplication is the
structural source of the y-inversion / direction-double-flip / period-clock-mismatch
defects of ADR-019/028/029/031/053: coordinate/half/orientation truth re-derived in
parallel and re-broken independently. silly-kicks' own 4.29.0 kloppy y-fix (ADR-031)
never reached lakehouse prod for these providers, and ADR-031 Gate C stayed open.

ADR-029 ruled against a native converter — but on the reasoning that a **raw-file**
loader "still cannot consume bronze DataFrames." It shipped `orient_frames_to_ltr`
(flag-based) as the minimal orientation step, leaving rescale + clock + ball-z mapping
duplicated. ADR-053 (lakehouse) then added a flag-free geometric orienter
(`correct_frames_to_home_ltr`) because the flag path mis-oriented ~half the
metrica/skillcorner games (no bronze `home_team_start_left`) and could not fix the GS
extra-time per-feed flip; ADR-053 recorded that orientation "stays in silly-kicks" was
the appeal of its rejected flag-based option, i.e. it relayed the net upstream.

## Decision

Add two **pure, bronze-consuming** builders — `tracking.skillcorner.convert_to_frames`
and `tracking.metrica.convert_to_frames` — parallel to `tracking.sportec` /
`tracking.gradientsports`, owning the full bronze→canonical-oriented-frame transform
(rescale + period-relative clock + id-namespacing + `ball_z` recovery + GK derivation +
speed + LTR orientation). This is the **bronze-consuming** path ADR-029 said did not
exist; `orient_frames_to_ltr` is retained for authoritative-flag callers (not removed).

Promote ADR-053's geometric net into the library as
`tracking.orient_frames_to_ltr_by_geometry` (schema-adapted port; per-period home-GK
median-x anchor, point-reflect mis-oriented periods, idempotent). It is the canonical
orienter the bronze builders use; the lakehouse retires its copy and depends on it. In
the library, orientation is the builder's owned, normal operation, so normal per-period
flips are silent (a no-GK-anchor period warns; zero-home-match raises, ADR-019).

Provider-specific decisions, verified against real bronze:
- **Clock.** SkillCorner reuses the SK events converter's nominal `_PERIOD_START_SECONDS`
  (same-provider single-source; SK P2 raw clock starts at 2700). Metrica rebases per-
  `(period)` min timestamp, because the 3 sample games use **mixed** raw clocks
  (continuous vs period-relative) — a nominal offset corrupts all three.
- **GK.** SkillCorner passes its authoritative per-player roster flag through
  `derive_goalkeepers` (Tier-1 validated, PR-S86). Metrica seeds **no** native GK
  (`gk_jersey_numbers` is a flat, team-agnostic list; `derive_goalkeepers` ORs and
  never clears, so a mis-flag would reach the orientation anchor) and lets the validated
  positional algorithm derive it (Tier-2, ADR-007); the flat list is an observability
  count cross-check only.
- **`ball_z`.** SkillCorner `ball_z` (100% populated, 0–14.7 m) is recovered to the `z`
  column — unblocking the SkillCorner post-shot height features (TF-48 PSxG) that were
  silently null in production.

The bounded-context boundary moves: raw→bronze (Spark read, the
`skillcorner_tracking ⋈ skillcorner_matches` team/GK join) stays lakehouse ingestion;
bronze→canonical-oriented-frame (coordinate/geometry domain logic) is silly-kicks'.

## Validation (Gate C closure)

- **Primary, kloppy-independent:** an event-anchored action↔frame y-identity gate
  (link via silly-kicks' own `link_actions_to_frames`, reproject per ADR-028, assert the
  acting player's frame position ≈ the action start coordinate, per-`(team, period)`,
  off-centre y, metrica-y named non-skippable). This closes Gate C by construction.
- **Structural:** parity to the kloppy oracle (owner-gated e2e; z kept in parity —
  kloppy carries SkillCorner ball z — plus an independent z physical-range check).
- The promoted orienter's acceptance oracle is ADR-053's existing tests
  (`test_frame_ltr_correction.py` + `test_frame_orientation_golden.py`), mirrored so the
  port is provably equivalent to the lakehouse original (a cross-repo contract).

## Consequences

- Additive in silly-kicks (new modules + a new public orienter; existing converters /
  gateway untouched; in no default xfn list). No silly-kicks model retrain.
- **Lakehouse delete-and-depends both builder copies** and retires
  `correct_frames_to_home_ltr` (its retrain trigger: `z` populated, orientation correct
  where it was absolute). The interim net stays only as an idempotent GS-ET backstop
  until **TF-23b** (the geometric net on the GS native adapter) lets it be deleted
  entirely. The provider-conditional net must not ossify — TF-23b is the gate.
- The payoff is realized only at the lakehouse deletion (the precondition).
- **C4-free**: the new surface is two bronze→frame converters + a geometric frame-LTR
  orienter, not action-coupled `add_*` aggregators — the C4 count stays **28**.
- **Metrica input contract** (verified vs the kloppy oracle): bronze `y` must be SPADL
  bottom-to-top; a consumer landing bronze straight from a kloppy `TrackingDataset` must
  flip `1 − y` first (the lakehouse bronze already does). A separate relay: the lakehouse
  metrica `spadl_actions` event-y did not co-locate with the canonical tracking ball in a
  pilot (a lakehouse events data issue, not the TF-23 builder).

## Alternatives considered

| Option | Why rejected |
|---|---|
| Keep ADR-029's line (orientation-only upstream) | Leaves rescale/clock/ball-z triplicated — the bug-class. |
| Native raw-file loader (ADR-029 option a) | Cannot consume bronze; duplicates kloppy. |
| Build a third (bootstrap-flag) orienter | Worse than ADR-053's geometric net; re-derives consumer logic. |
| Flag GK per-(team, jersey) for Metrica | Unrecoverable from the flat team-agnostic list; over-engineering 3 frozen matches. |

## Related

- **Spec:** `docs/superpowers/specs/2026-06-18-tf23-skillcorner-metrica-bronze-frame-builders-design.md`
- **ADRs:** ADR-029 (orient_frames_to_ltr), ADR-031 (kloppy CS-pin + cross-repo bronze contract),
  ADR-028 (action-LTR reprojection), ADR-017 (period-relative clock), ADR-007 (GK identification),
  lakehouse ADR-053 (geometric frame-LTR net).
- **Attribution:** geometric frame-LTR orientation method promoted from luxury-lakehouse
  ADR-053 `correct_frames_to_home_ltr` — see NOTICE.
