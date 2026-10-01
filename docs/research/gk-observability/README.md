# GK observability evidence (public tier)

Reproducible evidence behind the keeper-detection numbers cited by the TF-64 GK detection-gate spec
(`docs/superpowers/specs/2026-10-01-gk-detection-gate-design.md` §1). **Public tier only** — the 20
PUBLIC SkillCorner Open Data matches (A-League 2024/25, MIT licence), the only numbers citable in a
public document.

## The numbers the spec cites

| Claim (spec §1) | Value | Source |
|---|---|---|
| Keeper detected in live frames (baseline) | **17.6%** | `gk_observability_public.csv`, row `00 baseline (all live frames)`, column `gk_det` = `0.1757` |
| Keeper detected when the opponent attacks her box | **86.9%** | `gk_observability_public.csv`, row `12 OPP poss \| ball in own penalty area`, `gk_det` = `0.8694` |
| Detection collapses as play moves away | 0.85 near own goal → ~0 past midfield | `gk_observability_public_curve.csv` (`opp_gk_det` by `bin_center_m`) |

Corpus bound: **20 public A-League 2024/25 matches** (40 keeper-perspectives, `gk_matches` column), SkillCorner
Open Data, MIT licence. Frame rate 10 fps. These are the **only** citable numbers; the broader restricted
pilot stays outside this repo.

## Files

- `gk_observability_multi.py` — the probe (public tier). Per-frame keeper + "unit" (keeper + 4 deepest
  outfielders, all `is_detected`) detection by game context: opponent possession × ball zone, own block
  type, defending set pieces, moment-of-truth windows, baseline. Writes the three CSVs below.
- `outfield_detection_probe.py` — outfield per-player detection % (sizes TF-58's exposure). Writes
  `outfield_detection_public.csv`.
- `gk_observability_public.csv` — detection rate per context (+ IQR across matches). **Holds 17.6% / 86.9%.**
- `gk_observability_public_curve.csv` — detection vs ball-distance-from-own-goal, possession-split.
- `gk_observability_public_pergk.csv` — per (match, team, context) rates (spread).
- `outfield_detection_public.csv` — per-match outfield detection rate + coverage thresholds.

## Re-run

```bash
# public token, fail-closed public-only (multi.py reads the env; outfield probe defaults it)
export PINING_FOR_THE_DATA_TOKEN=test-token-pining-for-the-data
python docs/research/gk-observability/gk_observability_multi.py --tier public
python docs/research/gk-observability/outfield_detection_probe.py
```
Data is pulled via pining (the public SkillCorner Open Data endpoint), cached under the script dir; no
checked-in tracking data. Read-only; prints aggregates only; regenerates the committed CSVs.

## What is NOT here, and why

- **No restricted-tier artifacts.** The dual-tier source of truth (a `--tier private` path over 12
  restricted pilot matches, its match-id list, the owner-token censoring probe, and every `*_private*`
  / `*_by_context*` CSV derived from pilot data) stays in the external research workstream. Restricted
  data and its identifiers never enter this public repo (data-redistribution policy).
- **No paper PDFs.** Third-party copyright; attribution is in `NOTICE`.

The committed CSVs are context × detection-rate aggregates (and quartiles) — not reversible to player
positions, so they are shareable derived work.
