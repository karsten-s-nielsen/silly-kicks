# SkillCorner corpus + visibility — measurement evidence

Raw probe outputs behind every registered figure in **ADR-038** and the design spec
`docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md` (rev 5).
All figures in that ADR/spec were *measured, not assumed*; these are the artifacts a reader
checks them against. Captured 2026-07-14 on the owner box against the pining API's expanded
SkillCorner listing (10 public + 98 owner-tier). Files are copied **verbatim** from the
session scratchpad — some carry cp1252 mojibake (`�`) in accented player names; that is the
raw capture, not corruption introduced here.

| File | What it measures | Supports (ADR-038 §) |
|---|---|---|
| `manifest_skillcorner_full.json` | The pining manifest for the SkillCorner matches: per-match `id`, `artifacts` map, `provenance`, `source.licence`, and the load-bearing **`visibility`** key (`public` for the redistributed open-data 10; `private` for the 98). This is the field the corpus taxonomy keys on. | §1, §2 |
| `competitions.txt` | The 98 owner-tier matches decomposed: competition × season (LaLiga 23/24 + 24/25, UCL 23/24 + 24/25), extra-time risk (13/98 `potential_overtime`), **pitch dimensions** (91 at 105×68, 7 down at 101–103 m — the source of the up-to-2.0 m goal-line error), and the **keeper confound** (Real Madrid supplies 99 of 198 GK slots; Courtois/Lunin/Kepa ≈ 49.5%). | §1, §4, §6, §7 |
| `cohort_uniformity.txt` | Per-match × 5-role artifact-size probe across all 98 (many rows `HTTP429` rate-limited; 23/98 fully sampled). Establishes the artifacts are the uniform SkillCorner V3 product and estimates the download cost (~0.46 GB compressed on the sampled subset; ~1.05 GB full). | §1 |
| `schema_new_events.txt` | `dynamic_events.parquet` dtypes — **294 columns**, byte-for-byte the same taxonomy as the canonical CSV product, so the SkillCorner SPADL converter runs unchanged. | §1 |
| `schema_new_freeze_frames.txt` | `freeze_frames.parquet` head + dtypes. Shows the per-player **`is_detected`** flag present in the feed, and that its only novel field vs full tracking is the (all-null here) `visible_area` camera polygon — i.e. the role is redundant. | §5, §1 |
| `schema_new_physical.txt` | `physical.parquet` — 29×36 match-level running aggregates (distance, HSR, sprints, psv99). No per-frame content, no silly-kicks consumer → not used. | §1 |
| `e2e_result.txt` | Full kloppy load + `convert_to_frames` + SPADL + `link_actions_to_frames` on one private match (1021404). Link rate **1.000**, median offset **0.000 s**, ball `z` on 100% of frames — and the **detection finding**: GK detected in **19.6%** of frames (17,915/91,410) vs outfield 66.6%; `is_detected` exists in the raw feed but the kloppy gateway discards it (`frames.visibility` all-null). | §5 |
| `orientation3_result.txt` | Same-player event↔tracking coordinate reconciliation, per (team, period), on match 1021404. `min(identity, 180°-reflect)` residual **0.0000 m** for 100% of events; a y-only flip is never the best fit → **no y-inversion** (ADR-031's CS pin holds; the away-team 180° is ADR-028 orientation, not a bug). This is the flip-signature evidence behind deviation (c) — the naive 26.98 m co-location median was a team-keyed orientation confound, orientation-resolved to ~1.2 m. | §3, §4, deviation (c) |

## What is NOT here (and why)

- The clamp-destruction figures (11.31% of ball rows snapped, up to 9.00 m; 3.2% of ball rows
  beyond the goal line) and the kloppy-vs-events pitch-length divergence (103.48 m assumed vs
  104.00 m declared, 0.263 m at the goal line) were measured on match 1886347 in-session; the
  numbers are recorded in ADR-038 §3–§4 and spec §1.6/§3.4. The manifest here contains that
  match's artifact map.
- The S1 rate-gate calibration (public-10 worst `player_frac(>3 m)` = 0.00086; a 4 m
  pitch-dim error = 0.00095, *inside* the clean band; catastrophic break = 0.34139) is
  recorded in ADR-038 §6 and pinned by `tests/tracking/test_skillcorner_s1_gate.py`.
