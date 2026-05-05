# PR-S26 — Kloppy gateway `is_goalkeeper` hardening (silly-kicks-side derived GK identification)

**Date:** 2026-05-04
**Status:** Draft v4 (post 3rd lakehouse-review round; Q26 verification triggered material design change — always_run_algorithm promoted to in-scope)
**Target release:** silly-kicks 3.3.0
**Branch:** `pr-s26-kloppy-gk-hardening`
**ADR:** ADR-007 (new) — derived goalkeeper identification
**Closes:** TODO.md Tier 2 — TF-26
**Unblocks:** TF-13 (frame-based defending-GK fallback), TF-14 (defensive-line geometry), entire GKDV stack TF-15..TF-19

## 1. Context

silly-kicks 2.7.0 (PR-S19, ADR-004) established the `silly_kicks.tracking` namespace with a 19-column long-form schema including `is_goalkeeper: bool`. Native adapters (Sportec, PFF) populate this column reliably from provider metadata; the kloppy gateway populates it via `"Goalkeeper" in str(player.starting_position)` against the kloppy `Player.starting_position` field.

The 2026-05-04 PR-S26 brainstorm cycle uncovered a systematic data-quality issue in the kloppy gateway path. Empirical sweep of `soccer_analytics.dev_gold.fct_tracking_frames` (the lakehouse mart that consumes silly-kicks-shape tracking output):

- **Sportec/PFF native:** 100% of (frame, team) pairs have GK detected. Reliable.
- **Metrica (kloppy gateway):** 21.3% of (frame, team) pairs have GK detected. Asymmetric — typically one team's GK is missing.
- **SkillCorner (kloppy gateway):** 50% of (frame, team) pairs have GK detected. Same asymmetric pattern.

### Root cause diagnosis

Direct kloppy source inspection (`kloppy/infra/serializers/tracking/`):

- **`metrica_csv.py:81`** hardcodes `starting_position=PositionType.Unknown` for all players. The Metrica CSV format physically does not carry position role data — the bytes aren't there. Cannot be patched at the kloppy parser level.
- **`metrica_epts/metadata.py:127`** supports GK detection via XML metadata `ProviderPlayerParameters` if present — works conditionally on dataset metadata quality.
- **`skillcorner.py:402`** sets `starting_position=None` for extrapolated players (those reconstructed from neighbors rather than originally tracked). The original-tracked players (line 504) get correct mapping; extrapolated lose role identity.
- **`pff.py:54+302`** maps `"GK"` → `PositionType.Goalkeeper` from PFF roster. Reliable.
- **`tracab/parsers/metadata/common.py:13`** maps `"G"` → `PositionType.Goalkeeper` from Sportec/IDSSE metadata. Reliable.

### Empirical impact downstream

Cross-provider sweep across all 7 Sportec matches (14 (match, team) pairs).

**Validation surface — tiered by evidence strength:**

- **Tier 1 (external native ground truth):** Sportec `is_goalkeeper` from `tracab/parsers/metadata/common.py` (provider-roster-derived, ground-truth reliable). Reproducing this is meaningful agreement.
- **Tier 2 (no external ground truth):** Metrica `is_goalkeeper` from lakehouse jersey-#1 heuristic (provably wrong on the sweep — see below); SkillCorner `is_goalkeeper` for extrapolated-player matches (kloppy returns NaN). For these, the algorithm output is **not** independently corroborated; we report behavioral consistency only.

**Tier 1 algorithm comparison (Sportec, n=14 — agreement with native ground truth):**

| Algorithm | Sportec agreement | Description |
|-----------|-------------------|-------------|
| **B (`dist + x_var` rank-product)** | 0/14 = 0.0% | Catastrophic failure — `x_var` as primary signal fails in real match data over 100K+ frames per player |
| **C (6-feature rank-sum)** | 10/14 = 71.4% | Mediocre — noisy features (`y_var`, `gl_dwell`) drag the signal down |
| **B+ filtered (`dist + pa_dwell` with `n_frames ≥ 30%` filter)** | **14/14 = 100%** | Clean — strict criteria caught 12/14 standard GKs; sweeper-keeper fallback caught 2/14 high-line GKs |

Multi-GK detection via strict criteria: produces same single-GK output as native on this sample (no GK substitutions present), validating that the strict criteria don't cause false positives.

**Threshold justification — empirical pa_dwell distribution (Sportec, post-`n_frames ≥ 30%` filter):**

| Cohort | min | mean | max | n |
|--------|-----|------|-----|---|
| GK pa_dwell | 0.310 | 0.537 | 0.762 | 14 |
| Highest non-GK pa_dwell (post-filter) | 0.122 | 0.187 | 0.259 | 14 |
| GK − max non-GK separation (post-filter) | 0.059 | 0.351 | 0.595 | 14 |

The `pa_dwell ≥ 0.40` strict-criterion threshold:
- **Catches 12/14 GKs** (pa range 0.404–0.762 in this cohort)
- **Excludes 0/14 non-GKs** (max post-filter non-GK pa is 0.259) → **14-pp empirical safety margin** against false positives
- **Misses 2/14 GKs** (pa = 0.310, 0.320 — both DFL-OBJ-0028FW playing as sweeper-keeper across J03WPY and J03WR9) — these fall through to the sweeper-keeper fallback path which empirically picks them correctly

The 2-GK miss for strict criteria is **deliberate**: those are real sweeper-keepers, and the fallback path is designed to handle them. Lowering the threshold (e.g., to 0.30) to catch them in strict would shrink the non-GK safety margin from 14 pp to 5 pp — undesirable. The threshold is empirically tuned at 0.40 because:

1. It comfortably separates **GKs (mean 0.554)** from **non-GKs (mean 0.187)** with sub-population tails barely overlapping post-filter
2. The two-stage shape (strict + fallback) captures the bimodal GK distribution (line-keepers vs sweeper-keepers) without forcing a single threshold to fit both modes

Pre-filter (no n_frames filtering), one non-GK in J03WR9 (DFL-OBJ-J01CVM, ~5,853 frames = 4% of match) had pa_dwell=0.874 — a brief substitute outfielder placed near goal during a set-piece. The `n_frames ≥ 30%` filter excludes them; this is the failure case the filter is designed to catch.

**Tier 2 finding — Metrica lakehouse jersey-#1 heuristic is empirically wrong** (separate from algorithm validation): on the 6 Metrica (match, team) pairs the lakehouse flags, the flagged player_id systematically differs from the player whose positional behavior matches a GK pattern (low `dist_mean`, high `pa_dwell`, full match coverage). Behavioral GK had jersey 11 (home) / 25 / 28 (away); lakehouse picked jersey 1 (a player with ~30% match coverage and outfielder-like positional signature). **This is a finding about the lakehouse heuristic, not about B+ correctness on Metrica** — we lack external roster ground truth for Metrica. B+ identifies the player whose behavior matches a GK; whether that player IS the GK requires roster-side verification we don't have for this sample.

### Strategic framing

The kloppy `is_goalkeeper` issue is a force multiplier across the GKDV research arc (TF-15..TF-19). Without it fixed:
- TF-13 (frame-based defending-GK fallback) requires geometric heuristic — ~80–120 LOC plus contentious ADR
- TF-14 (defensive-line geometry) needs additional geometric GK exclusion fallback — ~50 extra LOC
- TF-15 GK influence primitives, TF-17 cross-attempt model, TF-18 ghost-GK regression all require branchy "if is_goalkeeper else geometric-fallback" code paths
- TF-19 GKDV composition multiplies this complexity by 4

With it fixed cross-provider:
- TF-13 collapses to trivial `is_goalkeeper=True` lookup (~30 LOC, no ADR)
- TF-14 uses simple `~is_goalkeeper` filtering — single canonical method
- All downstream features get clean provider-agnostic implementations

PR-S26 is therefore the foundational primitive for everything tracking-aware-GK-related on the roadmap.

## 2. Scope

### In scope (PR-S26)

- New module: `silly_kicks/tracking/_gk_identification.py` — pure-function B+ filtered algorithm
- Patch `silly_kicks/tracking/kloppy.py` — invoke fallback when kloppy's per-(match, team) GK count ≠ 1; emit `is_goalkeeper_source` column
- One-line addition to `silly_kicks/tracking/sportec.py` and `silly_kicks/tracking/pff.py` — emit `is_goalkeeper_source="native"`
- Schema growth in `silly_kicks/tracking/schema.py`:
  - `is_goalkeeper_source: object` added to `TRACKING_FRAMES_COLUMNS` (and per-provider variants inherit via dict merge)
  - `is_goalkeeper_source` added to `TRACKING_CATEGORICAL_DOMAINS` with frozenset `{"native", "derived"}`
  - `TrackingConversionReport.n_teams_gk_derived: int` field
  - `TrackingConversionReport.derived_gk_picks: dict[tuple[str, str], list[str]]` field (per Q11 — surfaces the algorithm's audit trail to consumers)
- ADR-007 — derived goalkeeper identification (includes Tier-1/Tier-2 validation distinction per Q1; threshold-justification distribution table per Q3)
- Test harness: 13 unit + 6 integration + 5 invariant + 1 perf benchmark = 25 tests
- Three new synthetic fixtures (~65 KB total)
- Extended lakehouse-derived fixtures (~5.4 MB total) — multi-match per provider
- New `scripts/build_synthetic_gk_fixtures.py` generation script
- Updates to existing `scripts/build_lakehouse_ci_fixtures.py` (extended frame counts + match diversity)
- New `scripts/regenerate_gk_baseline.py` + committed `tests/baselines/gk_identification_baseline.json`
- CHANGELOG `[3.3.0]` Added/Fixed/Internal entries
- TF-10 row body (b) absorbs the lakehouse-side cross-repo coordination

### Out of scope (deferred)

- **Multi-flavor `method=` API for GK identification** — premature; no academic alternatives exist. Single canonical algorithm.
- **Per-period or per-shift GK slicing** — over-engineered; the per-row schema already encodes per-frame identity via on-pitch presence.
- **Frame-side ball-carrier identification (TF-5)** — different feature, separate roadmap item.
- **Sport-other-than-football** (futsal, 7-a-side) — algorithm thresholds are 11-a-side-football-specific; surface as ADR-007 limitation.
- **Add jersey number column to output schema** — schema growth beyond scope.
- **Add per-player position role column** — role data unreliable across providers; defer until reliable role-data path exists.
- **Per-frame GK identity (`current_gk_player_id` column)** — redundant with `is_goalkeeper=True AND on this frame`.
- **Switching `fct_tracking_frames` to consume silly-kicks `is_goalkeeper`** — lakehouse-side, tracked in TF-10 row body (b).
- **Retiring lakehouse Metrica jersey-#1 heuristic** — lakehouse-side cascade from above.
- **Kloppy SkillCorner extrapolated-player upstream PR** — filed as PR-S26 follow-up after ship per Q7 = (C); avoids coupling to upstream review timeline.
- **Live in-CI Databricks query** — committed lh_derived fixtures + baseline JSON suffice; live coupling is brittle.
- **Per-match calibration of B+ thresholds** — hardcoded; promote to dataclass only if a non-standard pitch league surfaces a real need.

## 3. Architecture

### 3.1 Module layout

```
silly_kicks/tracking/
├── kloppy.py                 # patch site: always-run algorithm + emit source via agreement-check
├── sportec.py                # one-line: emit is_goalkeeper_source="native"
├── pff.py                    # one-line: emit is_goalkeeper_source="native"
├── schema.py                 # add column to all *_TRACKING_FRAMES_COLUMNS dicts;
│                             #   add categorical domain; add report field
└── _gk_identification.py     # NEW: B+ filtered algorithm (pure function, pandas in/out)
```

### 3.2 Data flow

```
Native paths (Sportec, PFF):
  raw provider data → silly_kicks.tracking.{sportec,pff}.convert_to_frames
                    → kloppy/native is_goalkeeper already correct
                    → emit is_goalkeeper_source="native" verbatim
                    → n_teams_gk_derived=0 always

Kloppy gateway path (v4 — always-run algorithm per Q26):
  kloppy TrackingDataset → silly_kicks.tracking.kloppy.convert_to_frames
                         → produce frames_df (post pitch-dim normalisation)
                         → snapshot kloppy's per-(match, team) GK player set
                         → invoke _gk_identification.derive_goalkeepers() unconditionally
                         → algorithm overwrites is_goalkeeper based on its own picks
                         → per-(match, team): source="native" iff algorithm.set == kloppy.set, else "derived"
                         → report.n_teams_gk_derived = count of (match, team) pairs where set disagreed
                         → report.derived_gk_picks = audit dict for those disagreement cases
```

### 3.3 Why a separate `_gk_identification.py` module

- Algorithm is pure-function, pandas-in / pandas-out (per ADR-004 invariant 1) — naturally separable
- Testable in isolation (unit tests don't need a kloppy `TrackingDataset`)
- Mirrors the established pattern: `_kernels.py`, `_direction.py` — leading-underscore private modules for algorithm primitives

**Module is private**, not a public API. External callers (including lakehouse-side TF-10 cutover) consume via `convert_to_frames` output. If a future need surfaces for direct algorithm access, promote to public `silly_kicks/tracking/gk_identification.py` at that point — non-breaking change since import path would simply gain a public alias.

### 3.4 Why algorithm always runs (no conditional trigger)

**Q26 verification result (2026-05-04 lakehouse-review round 3):** kloppy 3.18's `starting_position` semantics is **"kickoff role only,"** NOT "rostered role." Empirical evidence: across 7 idsse + 3 metrica + 10 skillcorner = 20 matches in `dev_gold.fct_tracking_frames`, **zero (match, team) pairs have 2+ players flagged `is_goalkeeper=True`** — yet J03WR9 demonstrably had real substitutions (DFL-OBJ-J01CVM with 5,853 frames is a clear substitute appearance). Conclusion: substitute GKs are *not* flagged by kloppy; only the kickoff GK is. The kloppy tracab parser at `tracab/parsers/metadata/common.py:46-47` maps `StartingPosition` per-player, but Sportec/IDSSE source XML only sets `StartingPosition="TW"` for the player who actually starts the match — substitute GKs have a different (or absent) field value.

**Implication:** the original "count != 1 → derive" trigger from spec v1–v3 was insufficient. For native-path matches with a real GK substitution: kloppy returns count=1 (only starter flagged), trigger doesn't fire, and silly-kicks output has `is_goalkeeper=False` for the on-pitch substitute GK from minute 70 onward. Per-frame query "the GK in this frame" returns nothing for post-substitution frames. **This is a correctness defect**, not a documentation concern.

**Resolution: `always_run_algorithm` promoted from deferred to in-scope.** The algorithm runs on every match; source semantics are agreement-based:

```
algorithm_picks = derive_goalkeepers(frames).picks_for(match, team)
kloppy_picks    = {p for p in players if kloppy_flagged(p) is True}

if algorithm_picks == kloppy_picks:
    source = "native"   # we ran the algorithm and confirmed kloppy was right
else:
    source = "derived"  # algorithm overrides; kloppy was wrong, asymmetric, or missed a sub
```

This is structurally cleaner than the count-based trigger:
- **Substitution case** (kloppy = {starter}, algorithm = {starter, sub}): set-disagree → source="derived", both flagged correctly in output. **Q26 defect resolved.**
- **Standard single-GK case** (kloppy = {starter}, algorithm = {starter}): set-agree → source="native", same output as before.
- **Metrica CSV case** (kloppy = {}, algorithm = {found_player}): set-disagree → source="derived" (same as before).
- **SkillCorner extrapolated case** (kloppy = {} for affected team, algorithm picks): set-disagree → source="derived".
- **Kloppy says count=1 but wrong player** (Q8 hypothetical): kloppy = {wrong_player}, algorithm = {right_player}: set-disagree → source="derived", algorithm corrects. **Q8 defect also resolved.**

**Performance characterisation (measured 2026-05-04 on J03WMX, 3.21M rows):** vectorised pandas; median wall-clock **712 ms on a full Sportec match** (3 runs: 665 / 712 / 725 ms). Dominant cost is the per-(match, team, player) `groupby.agg`; per-team dispatch is sub-millisecond.

Implications for production batch sizes (now that algorithm always fires):
- **Sportec season batch** (38 matches × 2 teams = 76 (match, team) pairs): algorithm runs on every match → ~27 s aggregate added compute (was: ~zero under count-based trigger; new cost is the price of correctness for substitution cases). Most teams will resolve to source="native" because algorithm and kloppy agree.
- **Metrica season batch** (38 matches): same as before (~27 s).
- **SkillCorner season batch**: same as before (~27 s, since algorithm fires regardless).
- **CI tests** (synthetic ~39K rows + lh_derived per-provider 33-67K rows × 7 fixtures): expected aggregate ~1-2 seconds.

The 27s/season cost is small relative to typical ETL batch budgets and is justified by closing the Q26 substitution defect.

## 4. Algorithm

### 4.1 `silly_kicks/tracking/_gk_identification.py` — public API

```python
def derive_goalkeepers(
    frames: pd.DataFrame,
    teams: pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[str]]]:
    """
    Identify goalkeeper(s) per (match_id, team_id) from positional behaviour.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS-shaped output. Required columns: match_id,
        team_id, player_id, x, y, is_ball, is_goalkeeper. Coordinates must
        be in 0-105 / 0-68 SPADL convention (post pitch-dim normalisation).
        Asserted at function entry; ValueError if x or y range exceeds
        SPADL bounds (-1.0 to 106.0 for x, -1.0 to 69.0 for y, allowing 1m
        slack for measurement noise).
    teams : pd.MultiIndex | None, default None
        (match_id, team_id) pairs to derive. **None means: derive for all
        teams in `frames`** (v4 always-run mode per Q26 — algorithm fires
        unconditionally; gateway resolves source label via post-hoc
        agreement check with kloppy's pre-existing flag set).

    Returns
    -------
    frames_out : pd.DataFrame
        Copy of input with is_goalkeeper overwritten on rows belonging to
        identified GK player(s) for affected teams; other rows unchanged.
        is_goalkeeper_source = "derived" for affected teams, "native" elsewhere.
    derived_picks : dict[(match_id, team_id), list[player_id]]
        Audit trail: which player_id(s) were flagged per (match, team).
        Single-element list in normal matches; 2+ in substitution scenarios.
        Surfaced via TrackingConversionReport.derived_gk_picks for caller
        introspection.

    Raises
    ------
    ValueError
        If required columns are missing, if NaN match_id/team_id is
        encountered on player rows (pipeline integrity issue), or if
        coordinate range falls outside SPADL bounds (caller forgot to
        normalize via `to_pitch_dimensions`).
    """
```

### 4.2 Hardcoded thresholds (module-level constants)

```python
_GK_N_FRAMES_FRAC = 0.30      # candidate filter: player must appear in
                              # >= 30% of team's max player-frame count
_GK_PA_DWELL_MIN  = 0.40      # strict GK criterion: in own/opponent
                              # penalty area for >= 40% of on-pitch frames
_GK_DIST_MAX_M    = 20.0      # strict GK criterion: mean distance to
                              # nearest goal-line < 20m
```

### 4.3 Per-player feature aggregation (vectorised)

For each `(match_id, team_id, player_id)` over rows where `is_ball == False`:

```
n_frames     = count(rows where this player appears with valid x, y)
dist_mean    = mean(min(x, 105 - x))                    # m, distance to nearest goal-line
pa_dwell     = mean( (x < 16.5 OR x > 88.5) AND
                     (13.84 < y < 54.16) )              # frac in own/opponent penalty area
```

(`x_var`, `y_var`, `gl_dwell`, `max_dist` were features used in the algorithm-comparison sweep but are NOT part of the locked B+ filtered algorithm. `gl_dwell = mean(min(x, 105-x) < 5.0)` — fraction of frames within 5m of a goal-line; included in the §1 comparison context for completeness.)

Penalty area defined as 16.5m × 40.32m (standard FIFA dimensions). Coordinate-symmetric: tests `x < 16.5 OR x > 88.5` so the algorithm works without knowing which goal each team defends — both goals' PAs count, since a GK is in *some* PA.

NaN handling:
- Rows with `is_ball=True` excluded from groupby
- Rows with NaN `x` or `y` excluded per-frame (pandas `var` / `mean` skip NaN by default)
- Rows with NaN `match_id` or `team_id` raise (pipeline integrity issue)

### 4.4 Two-stage ranking

```python
# Stage 1: candidate filter
team_frames_max = max(n_frames over all players in this team)
candidates = players with n_frames >= _GK_N_FRAMES_FRAC * team_frames_max

# Stage 2a: strict GK detection (multi-GK output natural)
strict_gks = candidates with (pa_dwell >= _GK_PA_DWELL_MIN
                              AND dist_mean < _GK_DIST_MAX_M)

if strict_gks is non-empty:
    flag all of them as is_goalkeeper=True
    return  # multi-GK output emerges naturally if N players pass

# Stage 2b: sweeper-keeper fallback (only when no candidate hits strict)
score = candidates['dist_mean'].rank(method='first', ascending=True) \
      + candidates['pa_dwell'].rank(method='first', ascending=False)
flag the lowest-score candidate (if multiple tie at min, take first by player_id ASC)
# Single GK output for sweeper-keeper case
```

**Pandas rank method = `'first'` is locked.** Deterministic given stable input order (we sort by player_id ASC before ranking to remove input-order dependence). Alternatives considered:
- `'average'` (pandas default): produces half-integer ranks on ties — unpredictable
- `'min'` / `'max'`: ties get the same rank — score arithmetic still ties
- `'dense'`: slightly more compact ranks — tiebreaker behavior less predictable
- `'first'`: each unique value gets a unique rank in input-order — most deterministic given a stable sort

**`x_var` was REMOVED from the score** post lakehouse-review (Q4). The empirical sweep showed `x_var` is a catastrophic primary signal (B = 0/14); even as a 0.01-weighted tiebreaker it (a) rarely fires given continuous-float `dist_mean` over thousands of frames, (b) when it does fire, would *disfavor* sweeper-keepers (who have higher x positional dispersion than line-keepers — exactly the population the fallback is for). Pure `dist + pa_dwell` rank-sum with deterministic input ordering is sufficient and principled.

### 4.5 Empirical justification (recap from §1)

Cross-provider sweep on `dev_gold.fct_tracking_frames` (2026-05-04):

- **idsse (Tier 1 — native ground truth, n=14):** B 0/14, C 10/14, **B+ 14/14 (100%)**
- **skillcorner (Tier 1 — native ground truth, well-formed cases, n=5):** **B+ 5/5 (100%)**
- **metrica (Tier 2 — no external ground truth):** B+ identifies the player whose positional behavior matches a GK pattern in all 6; lakehouse jersey-#1 heuristic flags a different (low-coverage outfielder-like) player in all 6. **This is a finding about the lakehouse heuristic; B+ Metrica correctness requires roster-side verification we don't have.**

The two-stage shape gives perfect separation in the Tier-1 sample. Strict criteria caught 12/14 standard-GK cases; sweeper-keeper fallback caught the 2/14 Sportec team-matches where the GK plays high (DFL-OBJ-0028FW, pa_dwell ≈ 0.31–0.32). Post-filter, GK pa_dwell separates from max non-GK pa_dwell with a 14-pp empirical safety margin — see §1 distribution table.

## 5. API surface

Per Q5 = (A): no new kwargs on any `convert_to_frames` signature. The patch is invisible to existing callers; behavior changes only by being correct.

### 5.1 Schema growth in `silly_kicks/tracking/schema.py`

```python
TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    # ... existing 19 columns ...
    "source_provider": "object",
    "is_goalkeeper_source": "object",  # NEW: {"native", "derived"}
}

# Per-provider variants inherit via dict merge:
KLOPPY_TRACKING_FRAMES_COLUMNS    # inherits the new column
SPORTEC_TRACKING_FRAMES_COLUMNS   # inherits
PFF_TRACKING_FRAMES_COLUMNS       # inherits

TRACKING_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    # ... existing ...
    "is_goalkeeper_source": frozenset({"native", "derived"}),  # NEW
}
```

### 5.2 `TrackingConversionReport` fields

```python
@dataclasses.dataclass(frozen=True)
class TrackingConversionReport:
    # ... existing fields ...
    n_teams_gk_derived: int  # NEW
    """Count of (match_id, team_id) pairs where the positional fallback
    fired (kloppy's native is_goalkeeper count was != 1). 0 means kloppy's
    native flagging was reliable across the whole input. ADR-007."""

    derived_gk_picks: dict[tuple[str, str], list[str]]  # NEW
    """For each (match_id, team_id) where the positional fallback fired,
    the list of player_ids the algorithm flagged as GK. Single-element
    list in normal matches; 2+ in substitution scenarios. Empty dict when
    no fallback fired. Useful for downstream auditing — consumers can
    spot-check 'for matches where source=derived, who did we pick?'.
    ADR-007."""
```

### 5.3 Native paths — one-line additions

`silly_kicks/tracking/sportec.py` and `silly_kicks/tracking/pff.py` get a single line at the end of `convert_to_frames`:

```python
out_df["is_goalkeeper_source"] = "native"
```

Native paths set `n_teams_gk_derived=0` always (they never invoke the fallback).

### 5.4 Kloppy gateway integration

Inside `silly_kicks/tracking/kloppy.py:convert_to_frames`, after pitch-dimension normalisation, the algorithm **always runs** (per Q26 — see §3.4):

```python
from ._gk_identification import derive_goalkeepers

# Snapshot kloppy's per-(match, team) GK player set BEFORE override
kloppy_gk_sets = (
    frames_df[~frames_df["is_ball"] & frames_df["is_goalkeeper"]]
    .groupby(["match_id", "team_id"])["player_id"]
    .agg(lambda s: frozenset(s.unique()))
    .to_dict()
)

# Always run the algorithm
frames_df, algorithm_picks = derive_goalkeepers(frames_df)
# algorithm_picks: dict[(match_id, team_id), list[player_id]]

# Set source based on agreement (per-team granularity)
def _resolve_source(match_id, team_id):
    kloppy_set = kloppy_gk_sets.get((match_id, team_id), frozenset())
    algo_set = frozenset(algorithm_picks.get((match_id, team_id), []))
    return "native" if kloppy_set == algo_set else "derived"

# Apply per-(match, team) source vectorised
team_keys = list(zip(frames_df["match_id"], frames_df["team_id"]))
frames_df["is_goalkeeper_source"] = pd.Series(
    [_resolve_source(m, t) for m, t in team_keys], index=frames_df.index
)

# is_goalkeeper itself: derive_goalkeepers has already overwritten the column
# for teams where algorithm and kloppy disagree; otherwise unchanged.

# Report fields:
n_teams_gk_derived = sum(
    1 for (m, t) in algorithm_picks
    if frozenset(algorithm_picks[(m, t)]) != kloppy_gk_sets.get((m, t), frozenset())
)
derived_gk_picks = {
    (m, t): algorithm_picks[(m, t)]
    for (m, t) in algorithm_picks
    if frozenset(algorithm_picks[(m, t)]) != kloppy_gk_sets.get((m, t), frozenset())
}
# derived_gk_picks flows into TrackingConversionReport.derived_gk_picks
```

`derive_goalkeepers` overwrites `is_goalkeeper` to match the algorithm's identification on every (match, team) pair; the gateway then resolves `is_goalkeeper_source` based on whether kloppy's original flag set matches. Source values are computed per-(match, team) but stored per-row (same value across all rows of the same team-match — dictionary-encoded Parquet collapses to ~zero overhead).

**Key behavioral change from v3:**
- v3: algorithm only ran when kloppy.count != 1; kloppy's flag preserved otherwise; substitution cases silently mis-tagged
- v4: algorithm always runs; kloppy's flag is replaced with algorithm output; source records agreement vs override; substitution cases correctly tagged

### 5.5 Backward compatibility

Additive only:
- Existing callers of `convert_to_frames` continue to work; output gains one column
- `TRACKING_CATEGORICAL_DOMAINS["is_goalkeeper_source"]` is new; existing categorical-validation code that iterates the dict naturally picks it up
- No removed surface, no renamed surface, no behavior changes for existing columns

## 6. Error handling and edge cases

### 6.1 Input validation in `derive_goalkeepers`

```python
required = {"match_id", "team_id", "player_id", "x", "y", "is_ball", "is_goalkeeper"}
missing = required - set(frames.columns)
if missing:
    raise ValueError(f"derive_goalkeepers: frames missing columns {sorted(missing)}")
```

### 6.2 NaN handling

| Source | Behavior |
|--------|----------|
| Ball rows (`is_ball=True`) | Excluded from feature aggregation (NaN team_id, NaN player_id per ADR-004 §2; groupby naturally skips). |
| Player rows with NaN `x` or `y` | Skipped per-frame in aggregation. `pa_dwell` and `dist_mean` are computed only over valid-coordinate frames. Handles Metrica's ~77% NaN-ball-coords case (memory `reference_lakehouse_tracking_traps`) — affects ball not players, but defensive design either way. |
| Player rows with NaN `match_id` or `team_id` | Loud raise (pipeline integrity failure upstream). Not silently skipped per `feedback_loud_raise_for_required_input_columns`. |

### 6.3 Edge case behavior

| Edge case | Behavior |
|-----------|----------|
| Empty `frames_df` | `gk_count_per_team` empty Series; `teams_needing_fallback` empty; algorithm doesn't fire; `n_teams_gk_derived=0`. Report well-defined; no exception. |
| Team has 0 players passing the n_frames filter | Should be impossible (max is computed over the team; at least the max-frames player passes). Defensive `AssertionError` if hit — logic bug. |
| All candidates fail strict criteria | Sweeper-keeper fallback fires: lowest B+ rank-sum picked. Single GK output. |
| All candidates pass strict criteria | All flagged `is_goalkeeper=True`. Invariant test catches if any (match, team) gets > 2 GKs flagged (assert violation in CI). |
| Single-player team (test-fixture degenerate) | `max(n_frames)` is that player's count; filter passes them; flagged. Tests cover this. |
| Match has `is_ball` rows but no player rows | `gk_count_per_team` empty. Algorithm doesn't fire. `n_teams_gk_derived=0`. Probably an upstream data error but doesn't crash. |
| Kloppy returns 2+ GKs for a team (rare; would require source XML to flag both starter and sub as starting role GK — empirically not observed in 20 matches per Q26) | Algorithm runs (always); output depends on candidate filter + strict criteria: (a) if 2+ candidates pass strict, all flagged (multi-GK preserved); (b) if only 1 passes strict, single GK output (kloppy's 2nd was a roster artifact); (c) if 0 pass strict, sweeper-keeper fallback picks single. Source = "native" if algorithm.set matches kloppy.set, "derived" otherwise. |
| **Real GK substitution** (Q26-confirmed defect resolved): kloppy returns count=1 (only kickoff GK flagged); on-pitch sub GK has is_goalkeeper=False from kloppy | Algorithm always runs; if both starter and sub pass strict criteria across their on-pitch windows, algorithm flags both → algorithm.set != kloppy.set → source="derived" → algorithm output overrides → both correctly flagged. Per-frame query `is_goalkeeper=True AND on this frame_id` returns the on-pitch GK throughout the match. |
| Brief substitute (<30% of match) coming on at end of game in goal-near area (e.g., emergency outfielder during corner-kick defense, mimicking the J03WR9/J01CVM scenario) | n_frames filter excludes them from candidates; algorithm output remains the original starter only → algorithm.set == kloppy.set → source="native". Information from kloppy preserved; the brief outfielder isn't promoted to GK. **No demotion concern** because we never demoted a real substitute — we never flagged them in the first place. |

### 6.4 Logging policy

- No `warnings.warn()` on fallback firing — expected path on Metrica/SkillCorner, not exceptional.
- `TrackingConversionReport.n_teams_gk_derived` is the audit surface; callers inspect it.
- Pure-function discipline (per ADR-005-style invariant): no print, no warn, all auditing through structured report.

### 6.5 Failure modes

| Condition | Exception | Message shape |
|-----------|-----------|---------------|
| Required columns missing | `ValueError` | `"derive_goalkeepers: frames missing columns [...]"` |
| Internal logic error (zero candidates after filter) | `AssertionError` | `"derive_goalkeepers: zero candidates after n_frames filter for ({match}, {team}); this is a bug"` |
| `match_id` / `team_id` NaN on player rows | `ValueError` | `"derive_goalkeepers: NaN match_id/team_id encountered (pipeline integrity issue)"` |
| Coordinate range outside SPADL bounds (Q2) | `ValueError` | `"derive_goalkeepers: coords must be SPADL 0-105/0-68; got x∈[{xmin},{xmax}] y∈[{ymin},{ymax}] (caller must run to_pitch_dimensions first)"` |

## 7. Testing strategy

Per Q6 = (B) full-spectrum harness.

### 7.1 Unit tests — `tests/tracking/test_kloppy_gk_hardening.py` (13 tests)

Tests the algorithm in isolation, no kloppy dependency:

| # | Test | What it validates |
|---|------|-------------------|
| 1 | `test_b_plus_score_function` | Hand-crafted feature vectors → B+ rank-sum picks GK candidate. |
| 2 | `test_candidate_filter_excludes_brief_substitute` | 11 players with n_frames=10000 + 1 brief sub with n_frames=500 placed near goal → filter excludes the substitute. |
| 3 | `test_strict_criteria_one_gk` | Standard match: only the actual GK has `pa_dwell≥0.4 AND dist<20`. Single GK output. |
| 4 | `test_strict_criteria_two_gks_substitution` | Substitution synth: starter + sub both pass strict. Both flagged. |
| 5 | `test_sweeper_keeper_fallback` | GK plays 25m off line, pa_dwell=0.2 (below threshold). Fallback fires; lowest B+ rank-sum picks the GK. |
| 6 | `test_no_strict_candidates_falls_back_to_lowest_score` | All candidates fail strict; falls back to single-pick. |
| 7 | `test_single_player_team_degenerate` | 1 player on team. Picked by default. |
| 8 | `test_empty_frames_no_exception` | Empty input → empty result; no raise. |
| 9 | `test_required_columns_missing_raises` | Missing column → `ValueError`. |
| 10 | `test_nan_match_id_raises` | NaN match_id on player row → `ValueError`. |
| 11 | `test_ball_rows_excluded_from_aggregation` | Ball rows present (NaN team_id, NaN player_id) → algorithm runs cleanly. |
| 12 | `test_pa_dwell_coordinate_symmetric` | Players in `x ∈ [88.5, 105]` AND `x ∈ [0, 16.5]` both compute correct pa_dwell. |
| 13 | `test_coord_range_outside_spadl_raises` (Q2) | Centred coords (`x ∈ [-52.5, 52.5]`) or 120/80 StatsBomb coords → `ValueError` "coords must be SPADL 0-105/0-68". Asserts loud failure when caller forgot pitch-dim normalisation. |

### 7.2 Integration tests — `tests/tracking/test_kloppy_gk_integration.py` (6 tests)

Round-trip via `convert_to_frames` on each fixture. **Tests are categorised by validation strength** (per Q6 lakehouse review):

- **External-truth tests** validate against an independent ground truth (kloppy native flag for Sportec/PFF/well-formed-SkillCorner — sourced from provider roster metadata).
- **Self-consistency tests** validate that the algorithm's output matches a committed snapshot (regression coverage); the *correctness* of that snapshot for cases where no external truth exists (Metrica, extrapolated-SkillCorner) is not independently verified.

| # | Test | Validation strength | What it validates |
|---|------|---------------------|-------------------|
| 1 | `test_sportec_lh_derived_native_path` | **External-truth** | All Sportec teams have `is_goalkeeper_source="native"`; `n_teams_gk_derived=0`; `is_goalkeeper` matches kloppy verbatim (kloppy native flag is provider-roster-derived ground truth). |
| 2 | `test_metrica_lh_derived_derived_path` | **Self-consistency** | All Metrica teams: `is_goalkeeper_source="derived"`; `n_teams_gk_derived` matches expected; flagged player matches committed baseline. **No external roster verification — algorithm output is the source of truth here.** |
| 3 | `test_skillcorner_lh_derived_path` | **Mixed** (external for well-formed, self-consistency for extrapolated) | SkillCorner: `is_goalkeeper_source` distribution per team-match matches the committed baseline. For the `"native"` cases, the test additionally asserts `is_goalkeeper` is preserved verbatim from kloppy (external-truth check). For `"derived"` cases (extrapolated-player matches), only baseline-match (self-consistency). Data-driven via baseline JSON. |
| 4 | `test_synthetic_substitution_fixture_two_gks_flagged` | **External-truth** | Loads `synthetic/gk_substitution.parquet` (constructed with known GK identities); both starter and sub flagged `is_goalkeeper=True`; `is_goalkeeper_source="derived"`. Synthetic ground truth is the construction itself. |
| 5 | `test_pff_native_path_emits_native_source` | **External-truth** | PFF native: source="native" on all rows; `n_teams_gk_derived=0`; `is_goalkeeper` matches kloppy verbatim (PFF native flag is provider-roster-derived ground truth). |
| 6 | `test_baseline_regeneration_deterministic` | **Self-consistency** | Run `scripts/regenerate_gk_baseline.py` on committed parquets; assert output JSON matches committed baseline byte-for-byte. (Per `feedback_codegen_for_data_to_code_integrity`.) Pure regression test — preserves whatever the algorithm produced when the baseline was generated. |

**Coverage gap acknowledged (per Q6 lakehouse review):** for Metrica + extrapolated-SkillCorner cases (test #2 + the `"derived"` portion of #3), no external-truth source exists in our fixtures. Bridging this requires either (a) a manual roster lookup for specific Metrica/SkillCorner matches (license + data-availability constraints), or (b) accepting algorithm-self-consistency as the validation surface for these provider paths. We chose (b) for PR-S26; if a future external-truth roster surfaces (e.g., licensed Metrica EPTS metadata or a paid SkillCorner roster API), an additional test can anchor against it.

### 7.3 Invariant tests — `tests/invariants/test_kloppy_gk_invariants.py` (5 tests)

Per `feedback_invariant_testing` memory:

| # | Test | Invariant |
|---|------|-----------|
| 1 | `test_gk_count_bounds_per_team_match` | For each (match, team) pair in any test fixture: `1 ≤ count(distinct player_id with is_goalkeeper=True) ≤ 3`. Hard fail outside this range. **Loosened from ≤2 to ≤3** post lakehouse review (Q12) to accommodate rare double-substitution scenarios (starter injured + 1st-sub injured + 2nd-sub plays); fail-loud if a real match needs ≥4 (would be an extreme outlier worth manual review). |
| 2 | `test_is_goalkeeper_source_enum_membership` | All `is_goalkeeper_source` values ∈ `{"native", "derived"}`. No NaN, no other strings. |
| 3 | `test_is_goalkeeper_source_consistent_within_team` | For each (match, team) pair, **filtering to player rows (~is_ball)**: all rows have the same `is_goalkeeper_source` value. (Ball rows are explicitly excluded — they have NaN `team_id` per ADR-004 §2 and would trip a naive groupby.) |
| 4 | `test_is_goalkeeper_source_native_implies_kloppy_agreement` | When source="native", the algorithm's identified player set equals kloppy's pre-existing flag set for that (match, team) — i.e., the agreement check that produced source="native" is reproducible from the saved is_goalkeeper column. (v4 update: source=native means algorithm-vs-kloppy agreement, not kloppy-count-equals-one — though in practice all current samples have count=1 due to Q26's kickoff-role-only kloppy semantics.) |
| 5 | `test_native_path_is_goalkeeper_unchanged` | For Sportec/PFF native input, `is_goalkeeper` value matches kloppy-side input verbatim (patch never changes native paths). |

### 7.4 Test fixtures

#### 7.4.1 Existing fixtures kept

- `tests/datasets/tracking/pff/realistic.parquet` — synthesized but valid for native-path testing
- `tests/datasets/tracking/pff/medium_halftime.parquet`, `tiny.parquet` — existing coverage retained

#### 7.4.2 New synthetic fixtures (~65 KB total)

| Path | Purpose |
|------|---------|
| `tests/datasets/tracking/synthetic/gk_substitution.parquet` (~30 KB) | Multi-GK: 2 teams × 11 outfielders + 1 starter GK + 1 sub GK each. Starter plays period 1 (~750 frames); sub plays period 2 (~750 frames). Both GKs have realistic positional behavior (pa_dwell≈0.7, dist≈10m). All rows `is_goalkeeper=False` (forces derive). Algorithm must flag 4 distinct players. |
| `tests/datasets/tracking/synthetic/sweeper_keeper.parquet` (~15 KB) | 1 team with sweeper-keeper GK (pa_dwell≈0.25, dist_mean≈18m, x_var moderate). Outfielders normal. All `is_goalkeeper=False`. Strict criteria fail → sweeper-keeper fallback fires → lowest B+ rank-sum picks the GK. |
| `tests/datasets/tracking/synthetic/brief_outfielder.parquet` (~20 KB) | 1 standard GK (full coverage, pa≈0.8) + 1 brief substitute outfielder appearing in only 10% of frames AND positioned near goal during their on-pitch time (mimics J03WR9 sub-near-goal failure case). All `is_goalkeeper=False`. n_frames filter excludes the substitute. |

#### 7.4.3 Extended lakehouse-derived fixtures (~5.4 MB total)

**Replaces** the existing 70-frame slim parquets at `tests/datasets/tracking/<provider>/lakehouse_derived.parquet` with multi-match, longer-window slices. Same path; same loader contract; just bigger and more varied. Modified `scripts/build_lakehouse_ci_fixtures.py` produces:

| Provider | Match count | Frames per match | Total per provider |
|----------|-------------|------------------|--------------------|
| **idsse (Sportec)** | 3 matches | 1500 (period 1, first 60s) | ~3 MB |
| **metrica** | 2 matches | 1500 | ~1.2 MB |
| **skillcorner** | 2 matches | 1500 | ~1.2 MB |

Sportec match selection for diversity (all expected to follow `"native"` path post-PR-S26 since native flagging works):
- 1× standard match (e.g., J03WMX) — verifies normal-case path
- 1× sweeper-keeper match (J03WPY or J03WOH — DFL-OBJ-0028FW plays high; pa_dwell ≈ 0.31–0.32 per §1 distribution) — kloppy native still gives count==1 so source="native"; the sweeper-keeper algorithm fallback path is exercised by `tests/datasets/tracking/synthetic/sweeper_keeper.parquet` separately
- 1× third standard match (e.g., J03WR9 sliced to a non-J01CVM window) — third Sportec data point for general regression coverage. **Per Q27 lakehouse review:** the n_frames filter case is exercised by `synthetic/brief_outfielder.parquet`, NOT by this fixture. The original v3 plan to time-slice J03WR9 around the J01CVM on-pitch window doesn't actually exercise the filter — within a 1500-frame slice, J01CVM would have ~100% relative coverage, not the ~4% production reality. Synthetic data with controlled frame ratios is the right tool for this assertion.

SkillCorner match selection (per Q20 lakehouse review — production-realistic asymmetric case must be in the fixture):
- 1× match with all original-tracked GKs (kloppy native gives count==1 for both teams → source="native" both teams)
- 1× **asymmetric** match where home team's GK is original-tracked (count==1 → source="native") and away team's GK is extrapolated (count==0 → source="derived"). This is the dominant production case in §1's "typically one team's GK missing" finding. Within-match, the two teams have *different* `is_goalkeeper_source` values — exercises the per-(match, team) trigger granularity of §3.4.
- (No purely-uniform-derived SkillCorner fixture; the asymmetric one above already exercises the derived path on the away team. Adding a second purely-derived match would be redundant.)
- `build_lakehouse_ci_fixtures.py` selection logic: query `fct_tracking_frames` for SkillCorner matches grouped by `(match_id, team_id)` GK count; pick one with both counts==1 (uniform native) and one with mixed counts (asymmetric).
- `test_skillcorner_lh_derived_path` correspondingly asserts: invariant test #3 ("source consistent within team") still passes per-team in the asymmetric case (each team has uniform source); the asymmetric-WITHIN-match scenario is the production realistic case our fixture now covers.

Metrica match selection:
- Both matches expected to trigger `"derived"` path (Metrica CSV format always lacks role data)

#### 7.4.4 Generation scripts

| Path | Purpose |
|------|---------|
| `scripts/build_lakehouse_ci_fixtures.py` (modified) | Pull extended slices from lakehouse; deterministic given env credentials. |
| `scripts/build_synthetic_gk_fixtures.py` (new) | Generate the 3 synthetic fixtures deterministically; no external dependencies. |
| `scripts/regenerate_gk_baseline.py` (new) | Generate `tests/baselines/gk_identification_baseline.json` from committed parquets. Per `feedback_codegen_for_data_to_code_integrity`. |

#### 7.4.5 Baseline JSON

`tests/baselines/gk_identification_baseline.json` — per-(provider, match_id, team_id) entry with:
- `expected_gk_player_ids`: list[str]
- `expected_source`: `"native" | "derived"`
- `expected_n_frames_per_gk`: list[int] (audit only)

### 7.5 Performance budget — `tests/tracking/test_kloppy_gk_perf.py` (parametrized — 2 benchmark cases)

Per Q9 + Q22 lakehouse review:

| # | Test | Fixture | Rows | Budget | Assertion mechanism |
|---|------|---------|------|--------|----------------------|
| 1 | `test_derive_goalkeepers_runtime_budget[synthetic]` | `synthetic/gk_substitution.parquet` | ~39K (1500 frames × 26 players) | `benchmark.stats.median < 0.100s` AND `benchmark.stats.max < 0.200s` | pytest-benchmark `min_rounds=5`, `max_time=2.0s` |
| 2 | `test_derive_goalkeepers_runtime_budget[lh_sportec]` | Sportec lh_derived (1 of the 3 matches) | ~33K (1500 frames × 22 players) | `benchmark.stats.median < 0.150s` AND `benchmark.stats.max < 0.300s` | pytest-benchmark `min_rounds=5`, `max_time=2.0s` |

**Hard-ceiling assertion mechanism (per Q30):** locked to `benchmark.stats.max < <hard_ceiling>` (worst-case observed time across rounds). Alternatives considered: `mean + 2*stddev` (less interpretable, depends on noisy stddev estimates with small n_rounds); `median` (already covered by primary budget). `max` is the simplest worst-case guard.

**Production-scale gap (per Q28):** both budgets are at fixture-scale (~10⁴ rows × less than the production 3×10⁶ rows). They guard against catastrophic same-scale regression, not against non-linear regressions that only manifest at production scale. Production-scale runtime is **documented but not asserted in CI** — re-measure when investigating performance concerns. The 712ms baseline is the reference number for owner-side monitoring.

Documented expectations (not asserted in CI; recorded for plan-side reference, **measured 2026-05-04 on J03WMX**):
- Full Sportec match (~3M rows × 22-26 players, ~145K frames per period × 2 periods): **712 ms median wall-clock** (3 runs: 665/712/725 ms)
- Full Metrica season batch (38 matches → all run algorithm): expected ~27 s aggregate
- Full Sportec season batch (38 matches → all run algorithm post-Q26 always-run): expected ~27 s aggregate
- CI fixture sweep (~7 derive calls across all integration tests on extended lh_derived + synthetic): expected ~1-2 s

### 7.6 Test count summary

- **Unit:** 13 tests
- **Integration:** 6 tests
- **Invariants:** 5 tests
- **Performance:** 1 benchmark test
- **Total new tests:** 25, all in regular suite (not e2e)
- **Estimated runtime addition:** ~3-5 seconds
- **No skipped tests** — every code branch has at least one assertion (per `feedback_no_silent_skips_on_required_testing`).

## 8. Documentation

### 8.1 ADR-007 — derived goalkeeper identification

**Path:** `docs/superpowers/adrs/ADR-007-derived-goalkeeper-identification.md`

**ADR is the durable post-merge record; spec dies after PR ships. ADR-007 must ABSORB §1's content directly, not just cite the spec** (per Q25 lakehouse review).

Sections to author:
- **Header** (date, status, deciders) per existing ADR template
- **Context** — kloppy gateway data quality issue; per-provider source-line refs (verbatim from §1's "Root cause diagnosis")
- **Decision** — silly-kicks-side B+ filtered + provenance column (§3.1, §4 algorithm; §5 schema)
- **Empirical evidence (Tier 1 — external native ground truth)** — cross-provider sweep results (Sportec 14/14, SkillCorner well-formed 5/5); **port the §1 pa_dwell distribution table verbatim** so the threshold justification is in the ADR (per Q3 + Q25)
- **Empirical finding (Tier 2 — no external ground truth)** — Metrica jersey-#1 finding (lakehouse heuristic wrong on 6/6); explicit caveat that this is a finding about lakehouse, not B+ correctness on Metrica (per Q1 honest re-framing)
- **Alternatives considered** — kloppy upstream PR / multi-flavor `method=` / per-period slicing
- **Consequences**:
  - Schema and data: new column + new TrackingConversionReport field
  - **Algorithm-uniform `is_goalkeeper` semantics** (Q18 + Q26 resolution): `is_goalkeeper=True` consistently means "this player's positional behavior over the match matched a GK pattern" regardless of `is_goalkeeper_source`. The source label is metadata about how the identification was reached, not about what `is_goalkeeper=True` means. Q26's empirical finding (kloppy 3.18 `starting_position` is kickoff-role-only; substitute GKs not flagged) drove the v4 design change to `always_run_algorithm`: algorithm fires on every match; output reflects algorithm identification; source = "native" iff algorithm.set == kloppy.set, else "derived". Real GK substitutions are now correctly tagged (both starter + sub flagged via multi-GK strict criteria). **Downstream consumer guidance unchanged**: per-frame query `is_goalkeeper=True AND on this frame_id` works correctly because off-pitch players have no rows at that frame.
  - **Operator telemetry for case (b) misfires** (per Q24): `TrackingConversionReport.derived_gk_picks` is the audit channel. Signal: for any (match, team) where `n_teams_gk_derived > 0` and `derived_gk_picks` lists a single player_id but the underlying kloppy input had count=2, the discarded second player was demoted by the n_frames filter. Lakehouse-side ingestion may log to `bronze.tracking_audit` if it proves a real concern.
  - Lakehouse cutover path (TF-10 row body (b))
  - Future kloppy upgrade compatibility — clean shift `derived → native` for SkillCorner extrapolated cases when fixed upstream
  - **Tier 2 limitation as durable known limitation** (per Q6 follow-up): external roster verification of Metrica/SkillCorner-extrapolated GK identity is not in scope; tracked as TODO.md research-future-work entry; revisit when roster-side ground-truth source becomes available
- **Related ADRs** — ADR-004 (tracking namespace charter), ADR-005 (tracking-aware features), ADR-006 (direction-of-play)

### 8.2 NOTICE file

Algorithm is original (no academic citation). The empirical-sweep-driven recommendation does not introduce new academic dependencies. Per `feedback_academic_attribution_discipline`:

- No new entry in NOTICE's "Mathematical / Methodological References" section
- ADR-007 carries the design rationale internally
- `_gk_identification.py` module docstring is succinct: "Original empirical heuristic for cross-provider GK identification; thresholds and stage shape are tuned against the 2026-05-04 cross-provider sweep documented in ADR-007. No academic prior art directly maps to this algorithm — closest-to-goal and dwell-time-in-region are general spatial-positional reasoning patterns, not specifically attributable to one paper."

(Per Q16 lakehouse review — earlier draft "inspired by Bauer & Anzer 2021 §3" was over-attribution; B&A §3 is about ball-carrier identification, not GK identification, and the shape similarities are too generic to constitute meaningful citation.)

### 8.3 CHANGELOG `[3.3.0]` entries

```markdown
## [3.3.0] — 2026-MM-DD

### Added
- `is_goalkeeper_source ∈ {"native", "derived"}` column on tracking output
  for all providers (TRACKING_FRAMES_COLUMNS + per-provider variants).
  Surfaces the path used to identify the goalkeeper in each (match, team).
- `TrackingConversionReport.n_teams_gk_derived` field — number of
  (match, team) pairs that used the silly-kicks-side positional fallback.
- ADR-007: derived-goalkeeper-identification.

### Fixed
- `silly_kicks.tracking.kloppy.convert_to_frames`: kloppy 3.18's
  `starting_position` flagging is unreliable on Metrica (CSV format lacks
  role data — kloppy hardcodes `Unknown`) and SkillCorner (extrapolated
  players get `starting_position=None`). Added silly-kicks-side
  positional-fallback GK identification when kloppy's per-(match, team)
  GK count is not 1. Empirical post-fix coverage: 100% (frame, team)
  pairs have GK detected on all 4 providers (was 21–50% on Metrica and
  SkillCorner).
- `silly_kicks/tracking/sportec.py`, `silly_kicks/tracking/pff.py`:
  emit `is_goalkeeper_source="native"` (single-line addition each).

### Internal
- New `silly_kicks/tracking/_gk_identification.py` — B+ filtered algorithm
  (private; promote to public if external callers need direct access).
- New `TrackingConversionReport.derived_gk_picks` field — algorithm audit
  trail per (match_id, team_id) for downstream consumer introspection.
- Test harness: 13 unit + 6 integration + 5 invariant + 1 perf benchmark
  = 25 tests; 3 new synthetic fixtures (~65 KB); extended
  lakehouse-derived fixtures (~5.4 MB) with 3 Sportec, 2 Metrica,
  2 SkillCorner matches × 1500 frames each (SkillCorner selection includes
  one asymmetric-team match where home is native + away is derived).
  Generation scripts:
  `scripts/build_lakehouse_ci_fixtures.py` (extended) and new
  `scripts/build_synthetic_gk_fixtures.py`.
```

### 8.4 PR description callout

The PR-S26 PR description includes a "For lakehouse consumers" callout linking to:
- The new column documentation
- The 6/6 Metrica disagreement empirical finding (§1 here)
- TF-10 row in TODO.md (cross-repo umbrella that absorbs the lakehouse-side cutover work)

## 9. Consumer-handshake (lakehouse boundary)

Per Q8 = (B) standard handshake, with TF-10 absorbing cross-repo coordination details.

### 9.1 Schema column-set growth

Net additions to silly-kicks tracking output per provider:

| Variant | New columns |
|---------|-------------|
| `TRACKING_FRAMES_COLUMNS` | `is_goalkeeper_source: object` |
| `KLOPPY_TRACKING_FRAMES_COLUMNS` | (inherits) |
| `SPORTEC_TRACKING_FRAMES_COLUMNS` | (inherits) |
| `PFF_TRACKING_FRAMES_COLUMNS` | (inherits) |

`TRACKING_CATEGORICAL_DOMAINS` gains `is_goalkeeper_source: frozenset({"native", "derived"})`.

Lakehouse boundary CI test (in the lakehouse repo, not silly-kicks) gains the new column to its expected column-set assertion.

### 9.2 Lakehouse-side cutover (tracked in TF-10 row body (b))

Post-PR-S26, `fct_tracking_frames` can switch to consuming silly-kicks-derived `is_goalkeeper` directly, retiring per-provider heuristics:

- **Metrica jersey-#1 heuristic:** empirically wrong on 6/6 sampled (match, team) pairs in 2026-05-04 sweep — should be retired
- **SkillCorner extrapolated-player handling:** silly-kicks fallback handles the bug naturally
- **Sportec/PFF native:** unchanged behavior; `is_goalkeeper_source="native"` confirms kloppy-side correctness

The cutover decision sits with the lakehouse team. PR-S26 ships the silly-kicks primitive with empirical evidence; the lakehouse team picks up timing.

### 9.3 Future kloppy upgrade compatibility

If kloppy upstream eventually fixes the SkillCorner extrapolated-player bug (filed as PR-S26 follow-up per Q7 = (C)), `is_goalkeeper_source` shifts from `"derived"` to `"native"` for affected SkillCorner cases. **No schema breakage**; this is an efficiency upgrade (skips the derive path) and a correctness signal that "kloppy got it right unaided."

## 10. Sequencing

1. **PR-S26** (this spec) — kloppy gateway hardening
2. **PR-S27** — TF-13 + TF-14 (frame-based defending-GK fallback collapses to trivial lookup; defensive-line geometry uses simple `~is_goalkeeper` filtering)
3. **PR-S28+** — GKDV stack (TF-15 → TF-19) builds on the now-clean `is_goalkeeper` primitive

PR-S26 is the structural prerequisite for all downstream GK-aware work.

## 11. Out of scope (recap from §2 with detail)

| Non-goal | Rationale |
|----------|-----------|
| Multi-flavor `method=` | No academic alternatives; premature config sprawl |
| Per-period GK slicing | Per-frame on-pitch presence already encodes per-frame identity |
| Frame-side ball-carrier (TF-5) | Different feature, separate roadmap item |
| Sport-other-than-football | Algorithm thresholds are 11-a-side-football-specific (16.5m PA, 105m goal-distance); enforced at function entry via SPADL coord-range assertion (Q2) |
| Jersey number column | Schema growth beyond scope |
| Per-player position role column | Role data unreliable across providers |
| Per-frame `current_gk_player_id` column | Redundant with `is_goalkeeper=True AND on this frame` |
| Lakehouse `fct_tracking_frames` cutover | Lakehouse-side; TF-10 row body (b) |
| Lakehouse Metrica jersey-#1 heuristic retirement | Lakehouse-side cascade |
| Kloppy SkillCorner extrapolated-player upstream PR | Filed as PR-S26 follow-up after ship per Q7 = (C) |
| Live in-CI Databricks query | Brittle; lh_derived fixtures + baseline JSON suffice |
| Per-match calibration of B+ thresholds | Hardcoded; promote to dataclass only on real need (per Q15 — module constants are mockable via monkeypatch; dataclass adds value only when exposing config to callers, which we're not) |
| Optuna sweep for B+ thresholds | No optimisation headroom — already 100% on Tier-1 Sportec ground truth |
| ~~`always_run_algorithm` opt-in mode~~ | **Promoted to in-scope in v4** per Q26 verification — algorithm now runs unconditionally and source is agreement-based. Resolves both Q8 (kloppy-says-1-but-wrong) and Q26 (sub GK silently mis-tagged on substitution match) defects. |
| External roster verification of Metrica/SkillCorner-extrapolated GK identity | Q6: would convert Tier-2 self-consistency tests to Tier-1 external-truth tests; requires either licensed Metrica EPTS metadata or paid SkillCorner roster API. Not available for PR-S26. |

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Algorithm produces wrong GK on a future Metrica/SkillCorner match (no external truth available) | Baseline JSON regression test catches deviation from current snapshot; conservative strict criteria minimise false positives; `derived_gk_picks` in TrackingConversionReport is queryable for spot-checks. **Acknowledged: this is self-consistency, not absolute correctness — see Q6 in §7.2 categorisation.** |
| Multi-GK output not validated against real substitution data | Synthetic substitution fixture (`synthetic/gk_substitution.parquet`) covers the multi-GK code path with constructed-known-truth; `n_teams_gk_derived` + `derived_gk_picks` in conversion report let us spot-check when 2-GK output appears in production. |
| Kloppy version bumps (3.18 → 3.19+) change `starting_position` semantics | Native-path test #5 asserts the algorithm-output `is_goalkeeper` for Sportec/PFF; integration tests assert end-to-end behavior. The agreement-check semantics are stable across kloppy versions because the algorithm is the source of truth in v4. |
| Lakehouse boundary CI test breaks on column-set assertion | Coordinated update via TF-10 row body (b); the consumer-contract handshake is the migration channel. |
| Performance regression on full season batches | Algorithm runs unconditionally (v4 always-run); pytest-benchmark performance budgets on synthetic + lh_derived fixtures (§7.5); documented production-scale baseline (712 ms median per Sportec match) for owner-side telemetry comparison. |
| **Kloppy says count=1 but the wrong player** (Q8) | Documented assumption — the `count != 1` trigger does not fire in this case, and silly-kicks ships kloppy's wrong answer. Would manifest on Metrica EPTS data with bad `ProviderPlayerParameters` metadata (rare; we don't have a fixture). Mitigation: defer to a future `always_run_algorithm` opt-in mode if this appears in production. Telemetry: monitor `is_goalkeeper_source="native"` cases for downstream feature anomalies; if any traces back to a kloppy-wrong-but-singular GK, escalate. |
| **Brief substitutes silently demoted to outfield** (Q7) | Documented behavior: when kloppy returns count=2 but the brief sub has `n_frames < 30%` of match, only the starter is flagged. Information loss accepted in exchange for noise reduction (cleaner output for matches with kloppy roster artifacts). Surfaceable via `derived_gk_picks` for review. |
| Coordinate-system precondition violated by caller (frames not yet pitch-dim-normalised) | `derive_goalkeepers` enforces SPADL coord-range at function entry (per Q2); `ValueError` raised loudly rather than producing silent garbage. Defends against future callers who skip the `to_pitch_dimensions` step. |
| **Native vs derived `is_goalkeeper` semantic asymmetry** (Q18) — RESOLVED in v4 via always-run algorithm | Q26 empirical verification confirmed kloppy `starting_position` is "kickoff role only," not "rostered role." v4 promotes `always_run_algorithm` to in-scope: algorithm fires on every match; `is_goalkeeper` reflects algorithm output uniformly; source="native" iff algorithm and kloppy agree on player set. Substitution cases now correctly tag both starter + sub via algorithm's multi-GK strict criteria. **No semantic asymmetry remains** — `is_goalkeeper=True` consistently means "this player's behavior matched a GK pattern," regardless of source label. The source label distinguishes "kloppy already had it right (we confirmed)" from "we corrected kloppy." |

## 13. Definition of done

- [ ] All 25 new tests passing in regular suite (13 unit + 6 integration + 5 invariant + 1 perf benchmark)
- [ ] All existing tests passing (no regression)
- [ ] `ruff check` + `ruff format --check` clean
- [ ] `pyright` clean
- [ ] `/final-review` (mad-scientist-skills:final-review) run + addressed
- [ ] CHANGELOG `[3.3.0]` entry complete with all symbols/columns enumerated
- [ ] **CHANGELOG date `2026-MM-DD` resolved to actual merge date** (Q17)
- [ ] ADR-007 written and committed; **§1 distribution table + Tier-1/Tier-2 framing absorbed verbatim into ADR-007** (not just cross-referenced — per Q25, ADR is the durable post-merge record); §1 will not survive after the PR ships, so the ADR carries the empirical evidence forward
- [ ] **ADR-007 Consequences section covers all 6 sub-bullets enumerated in §8.1** (Q31): (1) Schema and data; (2) Algorithm-uniform `is_goalkeeper` semantics (Q18+Q26 resolution narrative); (3) Operator telemetry for case-(b) misfires (Q24); (4) Lakehouse cutover path; (5) Future kloppy upgrade compatibility; (6) Tier 2 limitation as durable known limitation (Q6 follow-up; cross-link to TF-27 in TODO.md)
- [ ] `tests/baselines/gk_identification_baseline.json` committed and matches script output
- [ ] All 3 synthetic fixtures committed (`tests/datasets/tracking/synthetic/{gk_substitution,sweeper_keeper,brief_outfielder}.parquet`)
- [ ] **3 extended lh_derived parquet files** committed (one per provider: `tests/datasets/tracking/{idsse,metrica,skillcorner}/lakehouse_derived.parquet`); each parquet contains multiple matches (idsse: 3 matches; metrica: 2 matches; skillcorner: 2 matches including 1 asymmetric-team) concatenated into one file per provider — replaces the existing 70-frame slim layout (Q29 cleanup)
- [ ] TODO.md TF-26 row deleted (per `feedback_todo_grooming_delete_dont_annotate`)
- [ ] PR description includes "For lakehouse consumers" callout with cross-link to TF-10 row body (b)
- [ ] Lakehouse second-opinion review run (per `feedback_lakehouse_second_opinion_pattern`) — **2026-05-04 first round complete; spec v2 incorporates 17 items (15 accept, 1 partial, 1 reject-with-reason)**; further rounds optional
- [ ] Single commit per the user's standing rule (no WIP commits + squash; explicit approval before commit)
- [ ] No worktree usage (per `feedback_no_worktrees`)

---

End of spec.
