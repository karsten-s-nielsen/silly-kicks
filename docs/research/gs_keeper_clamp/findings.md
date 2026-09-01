# Gradient Sports goalkeeper-position clamp (27.5 m)

**Finding.** Gradient Sports' broadcast tracking **clamps the goalkeeper's tracked position to a hard
maximum of 27.5 m from its own goal line** — universal, exact, and across matches. A keeper who
actually sweeps higher is pinned at exactly 27.5 m. The clamp lives in the **raw provider data**;
silly-kicks flags the keeper by native roster identity and passes the coordinates through faithfully,
so it is **not** a silly-kicks or lakehouse defect.

Surfaced by the TF-60 PR3 (ghost-GK sweeper) re-fit: the sweeper's `>30 m` keeper coverage was
**Gradient Sports 0.0 %** (vs SkillCorner 0.24 %, Sportec/IDSSE 11.5 %), which prompted this
investigation. **Consequence:** any Gradient Sports goalkeeper-depth analysis — GK influence, ghost-GK,
`xt_gk`, the TF-19 / TF-60 counterfactual GK arms, or the factual keeper position itself — is invalid
whenever the real keeper is beyond 27.5 m. PR3 itself is unaffected: the sweeper learned the
high-sweeper regime from IDSSE and its parity/MAE are unchanged.

Reported-not-gated. Measured on the `sk_stageB_448` 179-match public corpus (WC2022 Gradient Sports +
IDSSE + SkillCorner), silly-kicks at the TF-60 PR3 commit.

## Evidence

### 1. The clamp is a hard ceiling, not a natural tail (extracted training labels)

Per-provider keeper goal-relative x (`gk_x`) over the 179-match corpus (786 k GS / 185 k SC / 77 k
Sportec keeper labels):

| provider | p50 | p90 | p99 | p99.9 | max | frac > 30 m |
|---|---|---|---|---|---|---|
| **gradientsports** | 13.9 | 25.4 | **27.5** | **27.5** | **27.5** | **0.0000** |
| skillcorner | 3.6 | 10.8 | 22.1 | 34.4 | 49.5 | 0.0024 |
| sportec | 14.6 | 31.2 | 46.7 | 52.4 | 52.5 | 0.1148 |

`p99 = p99.9 = max = 27.5` for GS is a **numeric clamp** (everything above ~27.5 is pinned to exactly
27.5), where SkillCorner and Sportec taper naturally to 49.5 / 52.5 m.

### 2. The clamp is keeper-specific

On the raw GS frames, keepers are confined to within 27.5 m of a goal line — **0.0000** of keeper
frames fall in the mid-band (27.5–77.5 m from a goal) — while **64.6 %** of GS *outfield* frames are in
the middle third. So it is the goalkeeper role that is capped, not a pitch-coordinate clamp.

### 3. silly-kicks is faithful (not the cause)

- GS keeper identity is **100 % native** (`is_goalkeeper_source == "native"`): the real roster keeper,
  tracked by player identity (`tracking/gradientsports.py` flags the keeper from the roster
  `positionGroupType == "GK"`, following the player, not the deepest position).
- silly-kicks' GS adapter applies **no coordinate clamp** (`derive_goalkeepers`' only distance
  constant is `_GK_DIST_MAX_M = 20.0`, and native GK identity means derivation is not even used here).

### 4. The clamp is in the raw provider data — universal and exact

Parsing the raw GS tracking (`homePlayers` / `awayPlayers`, centred coordinate system, keeper tracked
by roster shirt number) for the started GK of each team:

| match | keeper | raw `min|x|` | max dist-from-goal |
|---|---|---|---|
| 10502 | #23 / #1 | 25.00 | **27.50 m** |
| 10503 | #23 / #1 | 25.00 | **27.50 m** |
| 10512 | #1 / #22 | 25.00 | **27.50 m** |
| 3816 | #23 / #21 | 25.00 | **27.50 m** |
| 3831 | #1 / #1 | 25.00 | **27.50 m** |
| 3846 | #12 / #1 | 25.00 | **27.50 m** |

Every keeper, every match: `min|x| = 25.00`, max distance = **27.50 m exactly**, zero frames beyond
(176 k frames on 10502; ~7 k sampled on the rest). The value does not vary by match or keeper — a fixed
constant in the GS tracking pipeline, almost certainly a "goalkeeper zone" constraint in their model.

## The detector

`silly_kicks.tracking.validate_gk_position_clamp(frames) -> GkClampDiagnosis` detects this signature
provider-agnostically: per `(game_id, team_id)` it flags a **hard ceiling with a pileup** on the
keeper's distance from the nearest goal line (`min(x, 105 - x)`, so no orientation / goal-map is
needed). A natural keeper spends ~0 frames at its single highest sweep; a clamped one is pinned at the
ceiling in many. On real frames the separation is clean (~20×):

| provider | ceiling | pileup @0.1 m | verdict |
|---|---|---|---|
| gradientsports | 27.5 m | 1.4 % – 14.9 % | **clamped** |
| sportec / idsse | 37–52 m (natural) | 0.01 – 0.07 % | natural |
| skillcorner | 35–40 m (natural) | 0.00 – 0.04 % | natural |

It emits a filterable `GoalkeeperClampWarning`, and `tracking.gradientsports.convert_to_frames` calls
it automatically so every GS frame consumer is told (a silent pass-through was the trap). Defaults
`ceiling_tol_m = 0.1`, `pileup_threshold = 0.01`, `min_keeper_frames = 200`; the threshold is
conservative (a false positive on a natural provider is worse than missing a rarely-sweeping GS
keeper). Decision: ADR-083.
