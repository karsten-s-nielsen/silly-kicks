# ADR-125: Ball-trajectory dedup for the duplicate-frame velocity sawtooth (TF-79 follow-up) + z-provenance / honest-NaN `is_aerial`

| Field | Value |
|---|---|
| **Date** | 2026-10-15 |
| **Status** | Accepted |
| **Deciders** | Karsten |

> Supersedes the Gradient Sports "blunted ball" caveat in **ADR-124** (which was wrong — see Context).
>
> **Two decisions in one cycle (owner-ruled, spec §11):** (A) the provider-agnostic ball-trajectory dedup; (B) ball-`z` provenance (`z_source`) + honest-NaN `is_aerial`. Both ship together in 4.137.0 under this ADR. (A) is the primary GS fix; (B) folds in the ball-`z` spike finding so `is_aerial` stops trusting estimated z. The sections below cover (A); the **z-provenance** section at the end covers (B).

## Context

The TF-79 966-match validation (4.136.0) reported ball-contact detection **degraded on Gradient Sports** (event->contact agreement @0.5 s ~= 5 %, vs SkillCorner 79 %, IDSSE 45 %) and attributed it to a **"blunted / pre-smoothed ball."** A post-release spike disproved that cause:

- At **real touches** (event-anchored, raw x/y) the GS ball velocity-vector change is **|Δv| p50 ~= 3 m/s — as sharp as IDSSE (3.09)**. The ball is not blunted. The "|Δv| ~= 1" that motivated the blunted-ball claim was measured at *detected contacts* — a biased sample of the detector's own low-|Δv| over-fires, not real touches.
- The real cause: `_velocity_breaks` computes per-frame velocity by **raw consecutive-frame differencing** (`np.diff(xy) / dt`), and ball tracking carries **~50 % consecutive-duplicate positions** (IDSSE 42 %, GS 50 %). A duplicate frame gives `speed = 0`; the next real frame gives a jump -> a **0<->real velocity sawtooth** that masks the touch's real break. IDSSE has the same duplicates but slow per-frame motion, so its sawtooth is negligible; GS's faster motion makes it fatal. Frame **gaps are negligible** (0.1 %) — this is the duplicate-frame sibling of the ADR-120 gap problem.
- Deduping the ball trajectory before the break legs recovers GS agreement **~8–10x (0.05 -> ~0.4, = IDSSE's level)** and is **harmless to IDSSE** (control unchanged). So GS was a **detector-conditioning bug, not a degraded provider.**

## Decision

1. **Provider-agnostic dedup.** Every provider carries duplicate ball frames; the IDSSE control shows dedup is harmless to the good providers, so no provider branch.
2. **A pure helper `trajectory_dedup_mask`** (`silly_kicks/tracking/preprocess/_dedup.py`, the ADR-120 gap-handling sibling; numpy-only leaf, PRIVATE — no public surface) collapses runs of consecutive-duplicate positions (keeps index 0 + any point moving > `tol_m` = 1e-6 m from the immediately-preceding; **NaN/non-detection frames are kept, never collapsed**). `detect_ball_contacts` applies it per period immediately after the `sort_values("frame_id")`, before the RDP / velocity-break / |Δv| legs. Original `frame_id`s are preserved, so player attribution is unchanged.
3. **Option C ruled out** (consuming the ADR-120 derived velocity): RDP is **position-geometric** — it simplifies the position polyline and cannot use a velocity column — so the detector must read raw positions, and the three legs share one raw trajectory. Deduping that raw trajectory once is the coherent fix.
4. **Time-based windows.** The |Δv| and detection-quality windows become time-based (`_time_window`, searchsorted on `tt`, +/- 0.12 s / 0.2 s) instead of a fixed frame count, so they span the intended seconds on the sparser post-dedup sampling. `_velocity_breaks` is unchanged (already `dt = diff(t)`-based).

## Consequences

- **GS recovers** to IDSSE's level (spike + real-data e2e). **Contact values change on every provider** (GS massively, IDSSE/SkillCorner slightly) — **byte-identical is NOT claimed**; this is the intended correctness improvement. The contact-table **schema is unchanged by the dedup half** (the z-provenance section below adds `aerial_source`). **No retrain** (pure, model-free primitive). The 966-match validation memo is regenerated.
- **Accepted edge:** a pure **terminal** dead-stop — a ball that comes to rest and has no subsequent frame in the segment (rest to end-of-data) — is no longer detected via trailing static frames, because the dedup collapses them. Interior stops survive (the drop shows as fast-in / slow-out across the collapsed rest), and real traps have subsequent motion; the spike's net recovery is large, so this is an acceptable trade.
- **Per-provider yardstick (recorded, not a code change):** SkillCorner's events are a tracking-derived possession model, so validating tracking-detected contacts against them is partly **circular** -> its 79 % is optimistic; IDSSE's **independent** Sportec feed is the honest yardstick, and GS post-fix (~0.4) lands at IDSSE's level.

## Guards

`trajectory_dedup_mask` unit tests (run collapse, A->B->A preserved, float32-exact dups, NaN preserved, tol boundary); the detector two-sided recovery test (clean sampling detects the touch AND the duplicate-sawtooth fixture recovers it) + an exact resolved-contact-set identity control against a stripped twin (an over-collapsing dedup fails); the time-window red-green test (the old frame-count window measurably violates the seconds bound on a sparse fixture, ADR-056); `detect_ball_contacts` stays in `PURITY_ENTRIES`.

## z-provenance + honest-NaN `is_aerial` (folded in, spec §11)

### Context (ball-`z` spike)

The ball-`z` investigation found z is **present (non-NaN) on all tracking providers** but of **different provenance**: **MEASURED** on IDSSE/Sportec (dense optical, floored at 0, continuous) and Gradient Sports (dense optical; measured-but-noisy, z reaches −3.2 m at ground level); **ESTIMATED** on SkillCorner (broadcast — the loader artifact is literally `<mid>_tracking_extrapolated.jsonl`; non-physical signature: ball p50 ≈ 0.20–0.24 m up, off-ground 81–96 % of frames). So ADR-124's intent — `is_aerial` honest-NaN where z is absent **or estimated** — was unmet: the detector read SkillCorner's estimated z as if measured and emitted a boolean.

### Decision

5. **`z_source` frames column — provider DECLARES, never inferred.** A new tracking-frame column `z_source`, closed vocab `{measured, estimated, absent}`, `category` dtype (static per match, ADR-106). Added to **both** schema bases — `TRACKING_FRAMES_COLUMNS` (Int64-id base) and `KLOPPY_TRACKING_FRAMES_COLUMNS` (object-id base) — so every provider alias inherits it and the ADR-058 KLOPPY-mirrors-base gate stays green (SkillCorner **and** IDSSE/Sportec both build from the KLOPPY base). Set at each per-provider frame adapter from one central classification (`Z_PROVENANCE`): IDSSE/Sportec/Gradient Sports/Metrica `measured`, SkillCorner `estimated`, StatsBomb/SB360 `absent`, kloppy per underlying provider. Provenance is the adapter's ground truth (it knows if its z is extrapolated), never inferred from values.
6. **`is_aerial` honest-NaN.** In `detect_ball_contacts`, `is_aerial = <NA>` unless the contact frame's `z_source == "measured"` AND z is finite; a boolean (`z > aerial_z_m`) only then. **Fail-safe default:** when the input frames carry no `z_source` column (legacy/synthetic caller), default to `"absent"` — not `"measured"` — so a missing declaration yields honest-NaN, never a false boolean (fail-closed).
7. **`aerial_source` on the contacts table** (closed vocab = the `z_source` at the contact frame, `category`) + `first_contact_aerial_source` on `add_ball_contacts`, so a `<NA>` `is_aerial` is self-explaining (estimated vs absent vs measured). Registered in `feature_glossary`, the contact-columns contract and the SB360 freeze-frame verdict. Not geometric → no `reflection.py` entry.
8. **`validate_z_provenance` guard** (the `validate_gk_position_clamp` / `validate_velocity_regime` family): `validate_z_provenance` / `ZProvenanceDiagnosis` / `ZProvenanceWarning`, fails loud when a provider's z **signature** contradicts its **declared** `z_source`. The flagged signature is **PERSISTENT ELEVATION** (p50 off-ground AND off-ground fraction > ~0.7 — the SkillCorner extrapolation tell) OR all-flat/all-NaN; **ground-level negatives do NOT flag** (GS's −3.2 m is legitimately-measured ground noise). Thresholds pre-registered against the spike, never fit to the accuracy corpus. Signature-inference earns its place here as a VALIDATION, never the mechanism.

### Consequences

- **Public surface IS added** (the dedup half added none): `validate_z_provenance`, `ZProvenanceDiagnosis`, `ZProvenanceWarning`, `Z_SOURCE_VALUES` via `silly_kicks.tracking.__all__`. **Schema grows** by `z_source` (frames) + `aerial_source` (contacts) + `first_contact_aerial_source` (action enricher). `is_aerial` **changes on SkillCorner** (boolean → `<NA>`) — more values move, still **no retrain**.
- **Real-data e2e (DGX, one match per provider):** z_source emitted + correct per provider; `validate_z_provenance` consistent on every correctly-declared provider (SkillCorner declared `estimated`, off-ground 0.964, not flagged; GS negatives not flagged). SkillCorner contacts → `is_aerial` all `<NA>`, `aerial_source=estimated`; IDSSE/GS → real booleans, `aerial_source=measured`.

### Guards (z-provenance)

`test_tracking_schema` 20-col count + `z_source` in key set / category list / `TRACKING_CATEGORICAL_DOMAINS` + the `Z_PROVENANCE` map; per-provider z_source build tests (incl. SkillCorner-estimated); `_cast_to_declared_schema` cast; the category-dtype static-column gate; `feature_glossary` + contact-columns contract + SB360 verdict for `aerial_source`; `validate_z_provenance` unit tests both directions (estimated-signature-declared-measured flagged; measured clean; GS-negatives clean); the is_aerial honest-NaN two-sided test.
