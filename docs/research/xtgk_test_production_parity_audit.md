# xT-GK test-fixture ↔ production input-parity audit

**Date:** 2026-06-29 · **Scope:** handoff Item 2 (pre-Jeff verification). Do the `tests/tracking/test_xt_gk.py`
fixtures match the shape/conventions the lakehouse feeds `compute_xt_gk` in production? Green tests on
unrepresentative fixtures are exactly how the stale-grid and mocked-Spark guards gave false confidence.

**Method:** compared the test fixtures against (a) silly-kicks' *authoritative* `compute_xt_gk` input
contract (read from `_xt_gk.py` / `_gk_geometry.py` / the schema modules) and (b) the documented
production conventions (ADR-019 id-dtypes, `TRACKING_FRAMES_COLUMNS`, `spadlconfig` pitch dims, the
corrected directional grid). The lakehouse has **not yet wired the 4.36.0 xt_gk path into prod** (the
persist-coords migration is held), so the live `fct_action_values`/frame column dump is the lakehouse
side's to confirm once wired — flagged as **L** below. Everything else is verified here.

## Contract dimensions

| Dimension | `compute_xt_gk` requires | Test fixtures (`_gk_actions` / `_frames_for`) | Production | Verdict |
|---|---|---|---|---|
| **Action columns** | `game_id, action_id, team_id, player_id, period_id, time_seconds, type_id, start_x/y, end_x/y` | all present | SPADL `fct_action_values` superset | ✅ contract-complete |
| **Action id dtype** | dtype-agnostic via `_id_compat` (`canonical_id_series`/`ids_equal`) | **`int64` only** | int64 (SB/Opta/Wyscout), **nullable `Int64` (GS)**, **object str (kloppy: metrica/skillcorner)** | ⚠️ **gap** — fixtures don't exercise Int64/str ids in the FULL compute path; covered only by the ADR-019 invariance gate + `test_id_dtype_mismatch_string_frames` (mask only) |
| **Frame schema** | subset of `TRACKING_FRAMES_COLUMNS` + DAS cols (`vx,vy,team_in_possession`) + `source_provider`, `team_attacking_direction`, `is_goalkeeper`, `is_ball` | present (subset) | full 20-col `TRACKING_FRAMES_COLUMNS` from `convert_to_frames` | ✅ path-complete (the unused cols don't affect xt_gk) |
| **Provider / `source_provider`** | drives completion-variant auto-select (`variant_key_for_provider`) | **`"sportec"` only** → `gs` variant | `gradientsports`, `skillcorner`, `sportec`, `metrica` | ⚠️ **gap** — the full compute fixtures only exercise the `gs` variant; skillcorner-variant + base-rate-serve paths are covered by separate variant tests, not these |
| **Pitch dims** | `spadlconfig.field_length=105`, `field_width=68` | 105×68 | 105×68 | ✅ match |
| **Coordinate convention** | **LTR** (acting team attacks +x) + LTR grid | LTR (`team_attacking_direction="ltr"`) | LTR (ADR-022 gold) | ✅ match (orientation guard = Item 3, still open) |
| **xT grid amplitude** | injected `ExpectedThreat` (any fitted grid) | see below | corrected directional grid, **deep third raw xT ≈ 0.0085** | ❌ **PROVEN GAP** (headline) |

## Headline finding — grid amplitude (the proven divergence; also closes Item 5's doc-fix)

The DZV scale is dominated by the deep-third raw-xT amplitude. The fixtures diverged from production:

| Fixture | deep-third raw xT | deep `V_GK = xT·φ` | goalkick-origin DZV |
|---|---|---|---|
| `_fitted_xt` (flat ramp) | ~0.2 at x≈10 | ~0.4 | (off-scale — only for non-amplitude tests) |
| `_gk_realistic_xt` (cube ramp) | **col0 = 0**, col3 ≈ 0.008 | 0 at the goalkick origin | **~0** (understated) |
| **`_production_amplitude_xt` (NEW)** | **≈ 0.0085** | **≈ 0.02** | **≈ +0.018** |
| **Live WC2022 (lakehouse)** | ≈ 0.0085 | ≈ 0.02 | **+0.021** |

The cube-ramp fixture's **goalkick origin (col0) is raw xT = 0 → V_GK = 0 → DZV ≈ 0**, two orders below
the live +0.021. This is exactly why the ADR/CHANGELOG "DZV ~0.009 / O(0.01) on unit fixtures (deep
`V_GK` 0.005–0.01)" framing was misleading — it described the *understating* fixture, not production.
**Resolution:** added `_production_amplitude_xt` (deep third ≈ 0.0085) + `TestProductionAmplitude`, which
reproduces the live +0.02 DZV scale and guards against the fixtures silently understating it again. The
ADR-024 amendment / CLAUDE.md / CHANGELOG framing is corrected to match (Item 5).

Note (Item 5, analysis-side resolved): the 2× vs Jeff's La Liga ~0.009 anchor is **grid amplitude**, not a
form bug — our corrected global grid's deep third (raw xT ≈ 0.0085) is ~2× Jeff's implied deep value
(~0.004). Within his "sanity band, not a gate." The PEV/DZV *forms* are unchanged (Eyestone-confirmed).

## Recommendations

- **L (lakehouse, once xt_gk is wired):** dump the live `fct_action_values` + frame column list/dtypes
  feeding `compute_xt_gk` and diff against this contract — specifically confirm the id dtypes per provider
  and that the deep-third grid amplitude matches `_production_amplitude_xt`.
- **id-dtype coverage:** the ADR-019 invariance gate already fuzzes every `add_*` with int×str ids; the
  xt_gk full-compute fixtures staying `int64` is acceptable given that gate, but a follow-up could add an
  Int64/str-id variant to `TestComputeXtGk` if the lakehouse confirms GS/kloppy actually reach xt_gk.
- **provider coverage:** the completion-variant + base-rate-serve paths are gated by the variant tests;
  no change needed unless the lakehouse pools providers in one `compute_xt_gk` call (which raises by design).

## Verdict

Contract-complete on columns/pitch/coords; **one proven gap (grid amplitude) now fixed + guarded**; two
coverage notes (id-dtype, provider-variant) handled by separate gates. No `compute_xt_gk` correctness
issue surfaced — the fixtures were *unrepresentative on amplitude*, not wrong.
