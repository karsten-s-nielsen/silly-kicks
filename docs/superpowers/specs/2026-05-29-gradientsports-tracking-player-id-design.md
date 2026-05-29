# Gradient Sports tracking player-id resolution (TF-24 PR-A) — design

**Date:** 2026-05-29
**Status:** Design — pending user review
**Type:** Correctness fix + new public helper (additive). Minor release.
**Sequence:** PR-A precursor to the TF-24 Optuna calibration harness (PR-B). Independently
valuable: also fixes a silently-broken GS ball-carrier/DAS path in the luxury-lakehouse
consumer (lakehouse cross-confirmed; will adopt this helper as an AC-1 follow-up).

## 1. Problem (fact-checked against local GS WC2022 data, match 10502)

Gradient Sports **tracking** frames identify players **only by `jerseyNum`** (string,
e.g. `"8"`), split into `homePlayers`/`awayPlayers` arrays — there is **no player id** in
the tracking stream. Gradient Sports **events** identify players by `gameEvents.playerId`
(int, e.g. `8342`). The roster (`Rosters/<match>.json`) is the join table:
`player.id` (string `"8342"`), `shirtNumber` (string), `team.id` (string),
`positionGroupType` (e.g. `"GK"`). `(team.id, shirtNumber) → player.id` is **unique**
(verified 51/51, 0 dups on match 10502); after a `str`-cast, **32/32** event `playerId`s
match roster ids (`8342` also appears in `players.csv` `id`).

`silly_kicks.tracking.gradientsports.convert_to_frames` **passes `player_id` through** (it
is a required `EXPECTED_INPUT_COLUMNS` field) — so whoever builds the input frames must
perform the jersey→roster→int join, and there is **no helper to do it**. Consequently
`infer_ball_carrier` run on GS tracking emits a carrier whose id cannot be joined to the
events SPADL `player_id`, so any tracking↔events linkage for GS (carrier accuracy, DAS,
team-in-possession, sameteam features) is silently broken.

**Evidence this bit production:** the TF-24/TC-3 lakehouse calibration measured GS carrier
accuracy = **0.0 across every Stage-1 trial** (dragging global accuracy 0.57→0.38). The
lakehouse independently confirmed its own GS jersey→pid resolver returns a **string** id
while its events SPADL `player_id` is `LongType`/int — same bug, uncaught because GS has no
tracking oracle in its differential.

**Events SPADL id-space (fact-checked, current source):** `SPADL_COLUMNS` declares
`player_id`/`team_id` as `int64`; the GS events converter emits both as `int64`
(`silly_kicks/spadl/gradientsports.py:420-421`, via `Int64.fillna(0).astype("int64")`,
where `0` is the null-actor sentinel). The tracking schema
(`GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`) stores `player_id`/`team_id` as nullable
`Int64`. Pandas compares `Int64(5) == int64(5)` by value, so a tracking `Int64` carrier id
joins a `int64` event id correctly **when the underlying integer is the roster id**.

## 2. Goal

Add a public, pure helper that resolves GS tracking jersey numbers to the **events SPADL
`player_id` space**, so GS tracking frames are joinable to GS events SPADL — turning GS
carrier accuracy from 0.0 to a real value. Reusable by every GS tracking consumer (the
calibration harness and the lakehouse), not calibration-specific.

## 3. API

New public function in `silly_kicks/tracking/gradientsports.py`, re-exported from
`silly_kicks.tracking`:

```python
def add_gradientsports_player_ids(
    jersey_frames: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[pd.DataFrame, GradientsportsRosterReport]:
    ...
```

**`jersey_frames` (input):** long-form GS tracking rows (one row per player/ball per frame),
carrying at least:
- `team_side` : `"home"` / `"away"` (player rows) — from which array the row came; ball rows
  may be `None`/NaN.
- `jersey_number` : object/string (e.g. `"8"`); NaN for ball rows.
- `is_ball` : bool.
- (plus the other tracking fields the caller will feed to `convert_to_frames`:
  `game_id`, `period_id`, `frame_id`, `time_seconds`, `frame_rate`, `x_centered`,
  `y_centered`, `z`, `speed_native`, `ball_state` — passed through untouched.)

**`roster` (input):** normalized from `Rosters/<match>.json` by the caller, required columns:
- `team_id` : int (cast from roster `team.id`).
- `shirt_number` : object/string (roster `shirtNumber`).
- `player_id` : int (cast from roster `player.id`).
- `position_group_type` : object/string (roster `positionGroupType`); optional — if absent,
  `is_goalkeeper` is all-False with a warning.

**Contract — id-space (H2):** `home_team_id` / `away_team_id` MUST be the **events SPADL
`int64` team ids** (the same space the GS events converter emits, `gameEvents.teamId` →
`int64`). The helper writes `team_id` verbatim from these kwargs — if a caller passes
wrong-space ids (e.g. the lakehouse's latent GS `team_id`-as-string), `sameteam`/`fs.team`
features break silently while carrier accuracy still looks fine. This is the same bug class
as the player-id space and is guarded by the team-level test assertion in §4.

**Behaviour:**
1. Map `team_side` → `team_id` (`home_team_id` / `away_team_id`); ball rows → `team_id = pd.NA`.
2. **Normalize the join key on both sides (M1):** `str(...).strip()` both `jersey_number`
   (frames) and `shirt_number` (roster), so `"8"` vs `"08"` vs `" 8"` vs an int-typed `8`
   cannot silently miss (a miss → NA → carrier 0, the exact failure mode).
3. **Enforce roster join-key uniqueness BEFORE the join (N1 — guards silent row explosion):**
   a left-join on a non-unique `(team_id, normalized shirt)` would multiply every tracking
   row for that player (one frame row → N) and pick an arbitrary `player_id` — a
   worse-than-original corruption. §1 verified uniqueness on match 10502 (51/51, 0 dups), but
   future matches (mid-season number reassignment, a sub sharing a number, a data error) may
   violate it. So: detect duplicate `(team_id, normalized shirt)` keys, **`drop_duplicates(keep="first")`**,
   emit a `UserWarning` (`stacklevel=2`), and record `n_duplicate_roster_keys` in the report.
   **Never silently multiply rows.**
4. Left-join `(team_id, normalized jersey)` → the deduped roster `player_id` (int). Output
   `player_id` as nullable **`Int64`**. **Unmatched jersey → `pd.NA`, never `0`** (0 is the
   events null-actor sentinel; emitting 0 would spuriously collide — see
   `feedback_fillna_sentinels_trip_parity`). Ball rows → `player_id = pd.NA`.
5. `is_goalkeeper` (bool) = roster `position_group_type == "GK"` for the joined player;
   False for ball/unmatched. **GK vocabulary is pinned to the literal `"GK"`** (fact-checked
   on match 10502; M3) — a regression test pins the literal. **Zero-GK warning (N2):** when
   `position_group_type` is present, there are player rows, and **zero** rows match `"GK"`,
   emit a `UserWarning` listing the observed `position_group_type` values (`"no GK found;
   positionGroupType values were {…}"`) — so a vocabulary drift announces itself instead of
   silently producing a GK-less match (the same silent class as the original bug), degrading
   `defending_gk` / `gk_influence` for GS.
6. Return a **copy** of `jersey_frames` with `player_id`, `team_id`, `is_goalkeeper` set
   (additive; never mutates input — `feedback_additive_columns_over_inplace_mutation`), and
   a `GradientsportsRosterReport` (frozen dataclass) with: `n_player_rows`,
   `n_matched`, `n_unmatched`, `unmatched_jerseys` (set of `(team_id, jersey_number)`),
   `roster_size`, `n_duplicate_roster_keys`. Output row count **equals** input row count
   (the dedupe in step 3 guarantees no explosion).
7. **Loud warn on a degenerate match rate (M2):** when player-row unmatched fraction is
   ≥ 0.5 (or `n_matched == 0`), emit a `UserWarning` (`stacklevel=2`) — this is the precise
   signature of a wrong team-id-space / roster mismatch (the silent bug being fixed). The
   report still carries the full counts; the warning makes the regression announce itself
   instead of relying on the caller inspecting the report. **Never raises** on unmatched
   (NaN-tolerant per ADR-003).

The output frames satisfy `convert_to_frames`'s `EXPECTED_INPUT_COLUMNS` for
`player_id`/`team_id`/`is_goalkeeper`; the caller then runs `convert_to_frames` as today.
Raw JSONL flattening (bz2, home/away arrays → long form) stays in a caller walkthrough, not
the library (mirrors the GS **events** dual-API "caller normalizes, converter converts"
boundary).

**Why a standalone helper (not a `roster=` kwarg on `convert_to_frames`, not a raw loader):**
single responsibility (conversion vs identity resolution are distinct), consistency with the
events dual-API pattern, isolated unit-testability, and an explicit graceful report — per the
PR-A design discussion. Out of scope: overloading `convert_to_frames`; bz2/JSONL loading in
the library.

## 4. Testing

**Unit (synthetic, CI):** `tests/tracking/test_gradientsports_player_ids.py`
- Join correctness: `(team_side, jersey)` → expected roster `player_id`; home vs away
  disambiguation (same jersey number on both teams resolves distinctly).
- **Join-key format drift (M1):** frames `"8"` vs roster `"08"` / `8` (int) / `" 8"` still
  match after `str().strip()` normalization.
- Output dtype: `player_id` is `Int64`; values equal the int roster ids; matches the events
  SPADL `int64` space by value.
- `is_goalkeeper`: True only for the GK jersey; False elsewhere. **GK vocabulary pin (M3) +
  zero-GK warning (N2):** a roster using `"Goalkeeper"`/`"G"` (not `"GK"`) yields all-False
  `is_goalkeeper` **and** emits the zero-GK `UserWarning` listing the observed values — pins
  the literal *and* proves a vocab drift announces itself (not a silent GK-less match).
- **Roster key uniqueness / no row explosion (N1):** a roster with a duplicate
  `(team_id, shirt_number)` → output row count **==** input row count (no left-join
  explosion), a `UserWarning` is emitted, `report.n_duplicate_roster_keys > 0`, and the
  resolved `player_id` is the deterministic `keep="first"` pick.
- Unmatched jersey → `player_id = pd.NA` (**not 0**); ball rows → `player_id`/`team_id` NA,
  `is_goalkeeper` False; report counts correct; no exception.
- **Degenerate match-rate warning (M2):** ≥ 50% unmatched (or `n_matched == 0`) emits a
  `UserWarning` (`stacklevel=2`); a healthy match rate does not.
- `position_group_type` absent → `is_goalkeeper` all-False + a warning (`stacklevel=2`).
- Input not mutated (additive copy).
- Public-API Examples present (CI `test_public_api_examples` gate) + re-export from
  `silly_kicks.tracking`.

**Synthetic end-to-end join (CI — H1, the in-CI regression guard):**
`tests/tracking/test_gradientsports_player_ids.py::test_synthetic_join_yields_nonzero_carrier`
- Tiny GS tracking frames (jersey-keyed, both teams, a ball) + a synthetic GS events SPADL
  frame with `int64` `player_id`/`team_id` drawn from the SAME synthetic roster ids →
  `add_gradientsports_player_ids` → `convert_to_frames` → `infer_ball_carrier` →
  `link_actions_to_frames` → **assert carrier accuracy > 0**. This runs in CI (no real data)
  and guards the exact bug signature: that the resolved `Int64` player_id joins **by value**
  to a real-shaped `int64` events SPADL id. Distinct from the unit join test, which stops at
  dtype/value and never exercises the linkage.
  - **Limit of H1 (N4 — keep the e2e):** this test builds both the carrier ids and the
    events ids from the *same* synthetic roster, so it proves the join **mechanics**
    (`Int64`↔`int64` by value) but **cannot** catch a real id-derivation divergence (the
    events converter's `Int64.fillna(0).astype(int64)` vs roster `str→int` yielding different
    integers for the same player). That alignment is established only by §1's 32/32
    fact-check and the env-gated **real-data e2e** — which therefore must NOT be dropped on
    the assumption that H1 covers it.
- **Team-level non-degeneracy (H2b):** the fixture is **deliberately constructed with mixed
  possession (N3)** — some actions where the carrier is a teammate, some where it is an
  opponent — so that a `sameteam` comparison (`carrier.team_id == action.team_id`) is
  **neither all-True nor all-False**. (Without mixed possession the assertion is vacuous.)
  This catches a `team_id`-space mismatch that a player-id-only assertion would miss.

**e2e (real data, `@pytest.mark.e2e`):** `tests/tracking/test_gradientsports_player_ids_e2e.py`
- Source GS WC2022 (**owner-tier / private**) from the **pining-for-the-data mock provider API**
  (Bearer → 302 → presigned S3), authenticated with the **owner-tier token from the
  `PINING_FOR_THE_DATA_TOKEN` env var** (API base URL defaults to the deployed mock API,
  overridable via `PINING_API_URL`); skip only if the token is unset. **No local paths / no
  hardcoded token** in committed code. The token IS set on the dev machine, so this runs for
  real as a **hard ship gate** (must not skip). Fetch the `metadata`/`roster`/`tracking`/`events` artifacts, build
  jersey-keyed frames + roster, run `add_gradientsports_player_ids` → `convert_to_frames` →
  `infer_ball_carrier`, then time-link the raw on-ball event actor ids
  (`gameEvents.playerId` — the events int `player_id` space) via `link_actions_to_frames` and
  compute carrier accuracy. **Assert accuracy > 0** (the ultimate empirical proof the 0.0 bug
  is fixed on real data; the synthetic H1/H2b test is the in-CI proxy).

## 5. Housekeeping
- **Version:** minor (additive public helper) → confirm at bump time against current latest.
- **Docs:** CHANGELOG `### Added` (helper + the GS carrier-id fix narrative); a short tracking
  walkthrough note showing the `jersey_frames + roster → helper → convert_to_frames` flow;
  `silly_kicks.tracking` re-export.
- **No new ADR** (within ADR-004 tracking-namespace charter; it's a converter-adjacent helper).
- **GS licence gate:** no GS raw rows committed and **no local data paths** in committed code;
  the e2e fetches gated data from the pining-for-the-data API at runtime (owner token via env),
  commits nothing GS-derived.
- **Downstream:** the lakehouse adopts this helper (AC-1 follow-up, tracked lakehouse-side),
  replacing its hand-rolled string-pid resolver.

## 6. Out of scope (→ PR-B / follow-ups)
- The calibration harness itself (PR-B).
- Changing `infer_ball_carrier` / `LinkParams` / off-ball defaults (the apply step).
- A raw-JSONL GS tracking loader in the library.
- **A general "tracking jersey → events id-space" resolver (M4).** This is the **GS-specific
  instance** of a broader pattern — Metrica/SkillCorner also resolve tracking player ids and
  may share the latent join-space risk. We deliberately ship the GS helper now (concrete,
  fact-checked) rather than over-generalize; a shared abstraction is a future consideration
  only if a second provider is shown to need the same treatment.
