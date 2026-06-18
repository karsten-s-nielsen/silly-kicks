# BUG: kloppy-derived tracking frames have an INVERTED y-axis vs SPADL actions (SkillCorner + IDSSE)

> **STATUS: RESOLVED in 4.29.0 / 4.30.0 (ADR-031).** The kloppy TRACKING gateway
> (`tracking/kloppy.py::convert_to_frames`) now pins the canonical SPADL coordinate system via a
> CS-only `transform()` (T1, PR-S94, 4.29.0), and the y-inverting dev loader
> `scripts/_loader_pining.py::_kloppy_tracking_to_frames` was retired in favour of the IDSSE/Sportec
> DFL parse-port (T3, PR-S95, 4.30.0). SkillCorner + Metrica gateway frames are y-corrected; the
> native sportec/IDSSE path was already y-correct. This document is retained as the historical
> diagnosis. (NB: the lakehouse builds SC/Metrica via its OWN bronze builders, not the gateway — see
> ADR-031 and the TF-23 builders, ADR-034.)

**Severity (at discovery):** HIGH — silently corrupts every tracking-aware feature for all kloppy-based providers
(SkillCorner, IDSSE, and by the same path Metrica). Any feature that combines a SPADL action
coordinate with a sampled tracking-frame position is computed on a y-mirrored frame.
**Component:** the silly-kicks **kloppy tracking path** — `silly_kicks/tracking/kloppy.py`
(`convert_to_frames`, used for SkillCorner/Metrica) and `scripts/_loader_pining.py`
`_kloppy_tracking_to_frames` (used for IDSSE) — which produce a tracking y-axis opposite to the
SPADL action y-axis. **NOT** an orientation issue (ADR-028/029) — it is a single-axis y reflection,
outside the orientation family. **NOT** a clock/linkage issue (shooter time gaps are ~0.00–0.02 s).
**Found:** 2026-06-15, branch `pr-s93-shot-goalmouth`, via the TF-48 SkillCorner does-it-resolve smoke.
**Verified independent of TF-48:** the evidence below compares the SPADL action coordinate to the
ACTING PLAYER's own tracked position — no shot kernel involved.

## TL;DR
For SkillCorner and IDSSE, `action.start_y` (and `end_y`) is the mirror (`68 − y`) of the tracking
`y` for the same physical point. Native-adapter Gradient Sports is unaffected.
- **SkillCorner — fully confirmed:** flipping the tracking y (`y → 68 − y`) so it agrees with the
  actions makes `add_shot_goalmouth` resolution jump **0.123 → 0.605** on the 10 SC matches (matching
  the GS-at-10fps baseline). Single clean cause; the off-centre action↔shooter residual is ~0.2 m.
- **IDSSE — shares the y-flip, but it is NECESSARY-NOT-SUFFICIENT:** the y-flip residual is larger
  (~5.7 m, not ~0) and flipping the tracking y only partially / inconsistently helps (one match
  0.00→0.05, another 0.05→0.33, vs SC's 0.60). IDSSE has an ADDITIONAL, not-yet-isolated issue (see
  the IDSSE section) and needs its own investigation; do not assume the SC fix closes it.

## Evidence 1 — action `start` matches the SHOOTER's own tracked position only under a y-flip
`scripts/_tf48_clean_localize.py <provider>`. Restricted to **HOME (ltr) shots with OFF-CENTRE y**
(`|start_y−34|>8`) — for those the by-design action↔frame relationship is *identity*, removing the
near-symmetry-axis degeneracy (`identity≈y_flip` at y≈34, `identity≈x_flip` at x≈52.5) that an
unconstrained transform search overfits. Compares action `start` to the acting player's tracked
position at the shot time:

**SkillCorner** (matches 1886347, 1899585):
```
OFF-CENTRE HOME shots n=7:  median d_identity=45.3 m   median d_yflip=0.2 m   (time gap 0.00 s)
OFF-CENTRE HOME shots n=9:  median d_identity=37.1 m   median d_yflip=0.2 m   (time gap 0.00 s)
   start=(57.3,54.1) shooter=(57.2,13.9)  d_yflip=0.0   # 68−13.9 = 54.1, x identical
   start=(93.6,16.7) shooter=(93.7,51.3)  d_yflip=0.1   # 68−51.3 = 16.7
   start=(78.7,44.6) shooter=(78.8,23.4)  d_yflip=0.1   # 68−23.4 = 44.6
```
**IDSSE** (matches DFL-MAT-J03WMX, J03WN1):
```
OFF-CENTRE HOME shots n=1:  d_identity=32.1 m   d_yflip=1.0 m   (time gap 0.02 s)
OFF-CENTRE HOME shots n=5:  median d_identity=26.8 m   median d_yflip=5.7 m  (time gap 0.01 s)
   start=(84.9,47.1) shooter=(84.3,20.3)  d_yflip=0.8   # 68−20.3 = 47.7 ≈ 47.1
```
(IDSSE's y-flip residual is larger ~5 m than SkillCorner's ~0.2 m — a secondary, smaller offset on
top of the y-flip, possibly the kloppy event-location vs player-position gap; the y-flip dominates:
d_yflip ≪ d_identity.)

## Evidence 2 — flipping the tracking y FIXES resolution (root-cause + fix confirmation)
`scripts/_tf48_yflip_confirm.py skillcorner` — re-run `add_shot_goalmouth` with `frames["y"] = 68 − y`:
```
skillcorner 1886347:  resolved 4/23 (0.17) -> 15/23 (0.65)
skillcorner 1899585:  resolved 0/22 (0.00) -> 11/22 (0.50)
skillcorner 1925299:  resolved 2/21 (0.10) -> 14/21 (0.67)
skillcorner 1953632:  resolved 4/15 (0.27) ->  9/15 (0.60)
skillcorner TOTAL:    resolved 10/81 (0.123) -> 49/81 (0.605)
```
0.605 matches the GS-at-native-10fps baseline (~0.44–0.55 from the TF-48 sweep), i.e. after the
y-fix SkillCorner behaves like a normal 10 fps provider.

`scripts/_tf48_yflip_confirm.py idsse` — the y-flip is necessary but NOT sufficient for IDSSE:
```
idsse DFL-MAT-J03WMX:  resolved 0/21 (0.00) -> 1/21 (0.05)
idsse DFL-MAT-J03WN1:  resolved 1/21 (0.05) -> 7/21 (0.33)
```
i.e. inconsistent and well below SC's 0.60 — IDSSE has more than the y-flip.

## IDSSE — additional, not-yet-isolated issue (needs its own investigation)
Distinct from SkillCorner, on top of the shared y-flip:
1. The off-centre action↔shooter y-flip residual is ~5.7 m (SC's is ~0.2 m) — a second, smaller
   coordinate offset between the kloppy event coordinates and the kloppy tracking.
2. The BALL at the shot time often sits at the OPPOSITE end of the pitch (e.g. action implies the
   ball near x≈90 in possession-perspective, but the nearest-in-time ball sample is at x≈0). The
   shooter (player) tracking is fine at the same time (gap 0.01 s), so this is BALL-specific
   (`_kloppy_tracking_to_frames` maps `period_frame.ball_coordinates` — sparsity, a separate ball
   transform, or an x-orientation inconsistency are the candidates).
Net: IDSSE = the shared kloppy-tracking-y inversion PLUS an IDSSE-specific ball/coordinate problem.
Fix the y inversion first (it is real for IDSSE too), then re-diagnose the residual + ball placement.

## Localization (where the y-axes diverge)
Both kloppy tracking builders transform with the SAME call:
`to_pitch_dimensions(MetricPitchDimensions(x_dim=Dimension(0,105), y_dim=Dimension(0,68),
standardized=False, ...), to_orientation=Orientation.HOME_AWAY)`
— `silly_kicks/tracking/kloppy.py:105-112` (SkillCorner/Metrica) and
`scripts/_loader_pining.py:323-332` (IDSSE).

The events→SPADL paths DISAGREE with that tracking y, and AGREE with each other:
- SkillCorner actions come from the custom `silly_kicks/spadl/skillcorner.py` (`y_out =
  (y/half_width)*34 + 34`, a positive map, no flip);
- IDSSE actions come from the kloppy gateway `silly_kicks/spadl/kloppy.py convert_to_actions`.

Both independent event paths give `action_y = 68 − tracking_y`. Since the two event sources agree and
only the (shared) kloppy tracking transform disagrees, **the kloppy tracking transform's vertical
orientation is the outlier** — its y-axis is inverted relative to the SPADL convention.

**EXACT cause (confirmed by a second independent code review):** the two kloppy gateways normalize
differently. The EVENT gateway `spadl/kloppy.py:196-202` pins
`to_coordinate_system=_SoccerActionCoordinateSystem(...)` (`origin=Origin.BOTTOM_LEFT`,
`vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP`; class at `spadl/kloppy.py:293-326`) ⇒ events are
canonical. The TRACKING gateway (`tracking/kloppy.py:104-113`) and the dev loader
`_kloppy_tracking_to_frames` pass `to_pitch_dimensions` + `to_orientation=HOME_AWAY` but **never**
`to_coordinate_system` ⇒ they RETAIN each provider's kloppy-native vertical (`to_pitch_dimensions` only
rescales, it does not normalize vertical orientation). `grep` confirms `VerticalOrientation` /
`to_coordinate_system` / `BOTTOM_TO_TOP` / `Origin` appear in `spadl/kloppy.py` but NOWHERE in
`silly_kicks/tracking/`.

**FIX — CS pin, NOT a blanket flip.** Extract `_SoccerActionCoordinateSystem` to a shared location and pin
it on the tracking gateway too; kloppy then applies each provider's native→canonical flip CORRECTLY
(no-op if already canonical). The blanket `frames["y"]=68−y` in Evidence 2 was a CONFIRMATION ONLY — it
would DOUBLE-INVERT an already-correct provider, so **Metrica must not be blanket-flipped** (test it). The
CS pin also keeps events/frames from drifting (DRY) and corrects `derive_velocities` vy. Per-feature
downstream impact + the validation gates (Metrica, SC canonical re-check, the SHIPPED
`tracking/sportec.py` DFL native path, a real-data regression gate) are in the TODO.md entry; this is its
own brainstorm→spec→plan→PR, NOT bundled into TF-48.

## Blast radius
Every kloppy-provider tracking-aware feature that anchors on an action coordinate and samples frame
positions (`add_pre_shot_gk_*`, `add_pressure_on_actor`, `add_action_context`,
`defenders_in_triangle_to_goal`, `add_defensive_line`, `add_team_shape`, … and `add_shot_goalmouth`)
is computed on a y-mirrored frame for SkillCorner / IDSSE / Metrica. ADR-028's per-action reprojection
does NOT correct it (single-axis, not 180°). The synthetic `source_provider="synthetic"` invariance
fixtures (`test_action_ltr_mirror_invariance.py`) cannot catch it — they assume a clean 180° mirror.

## Reproduce
```bash
ssh karsten@192.168.68.73
source ~/.pining_env
export PINING_CACHE_DIR=~/Development/silly-kicks/xt_bandwidth_run/artifact_cache
cd ~/Development/silly-kicks
~/sk-s93-venv/bin/python scripts/_tf48_clean_localize.py skillcorner   # d_yflip≈0.2 vs d_identity≈40
~/sk-s93-venv/bin/python scripts/_tf48_clean_localize.py idsse         # d_yflip ≪ d_identity
~/sk-s93-venv/bin/python scripts/_tf48_yflip_confirm.py  skillcorner   # 0.123 -> 0.605 resolution
```
Scripts are read-only (load from cache, compare coords / run `add_shot_goalmouth`); `_tf48_*` are
throwaway, not committed. The TF-48 kernel itself is correct and Gradient-Sports-validated (all
ADR-030 floors pass on 64 GS matches); this bug is upstream of it and blocks SkillCorner/IDSSE/Metrica
coverage until the kloppy tracking y is aligned.
