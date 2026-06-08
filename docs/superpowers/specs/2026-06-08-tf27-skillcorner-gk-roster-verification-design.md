# TF-27 — SkillCorner external-roster GK verification (PR-S86)

**Status:** design approved 2026-06-08, pending spec review
**Origin:** 2026-05-04 PR-S26 lakehouse review Q6; ADR-007
**Branch:** `pr-s86-tf27-skillcorner-gk-roster`

## 1. Motivation

`silly_kicks.tracking._gk_identification.derive_goalkeepers` (PR-S26, ADR-007)
identifies the goalkeeper per `(game_id, team_id)` from positional behaviour,
because kloppy's native `is_goalkeeper` flag covers only ~21–50% of
SkillCorner/Metrica teams. The algorithm carries **Tier-1** validation against
Sportec native ground truth (14/14) but only **Tier-2** algorithm-self-consistency
for SkillCorner.

SkillCorner roster data is available via pining-for-the-data (public token), so we
can anchor `derived_gk_picks` against an **independent external ground truth** and
upgrade SkillCorner from Tier-2 to Tier-1. This closes the discoverable follow-up
ADR-007 logged as a durable known limitation.

The Metrica arm of the original TF-27 was **retired** (2026-06-08): public Metrica
samples are anonymized (integer player ids, no roster), so no external truth exists
to anchor against — a permanent data block, not a deferral.

## 2. Empirical basis (verified against the live pining backend, 2026-06-08)

Probed SkillCorner match `1886347` (one of the 10 public A-League matches):

- **Ground truth is explicit.** `match.json` `players[].player_role.acronym == "GK"`
  flags exactly one player per team — the **starting** GK (substitutes carry role
  `"SUB"`, not their position). Distribution for the match: `GK: 2` (one per team),
  e.g. `id=51009` (team 1805, "R. Scott"), `id=285188` (team 4177, "A. Paulsen").
- **Join key is unambiguous.** Tracking-frame `player_id` (dtype `object`/string)
  joins to `match.json players[].id` (cast to string): intersection 22/22 players.
  The other two id fields are NOT frame ids — `team_player_id` and `trackable_object`
  both have **zero** overlap with frame `player_id`. This is the easy-to-get-wrong
  detail; it is pinned.
- **The algorithm is already correct here.** `derive_goalkeepers` returns a 2-tuple
  `(frames_out, derived_picks)` (`_gk_identification.py:30`); its `derived_picks` (2nd
  element) was `{(1886347, 4177): ['285188'], (1886347, 1805): ['51009']}` — exact
  agreement with the roster GK for both teams (2/2). Note the keys are
  `(game_id, team_id)` with `team_id` as **int**.
- **Apples-to-apples with the gated path.** Re-confirmed on the **preprocessed** seam
  (`skillcorner.load → convert_to_frames → _preprocess`, i.e. smoothing + velocities) —
  the exact frames the e2e will gate on, not just raw `convert_to_frames`: identical
  exact-equality picks, and every rostered GK id present in the frame `player_id` set.

So TF-27 is a **validation** exercise expected to pass, with a hard gate that fails
loudly if `derive_goalkeepers` ever regresses.

## 3. Architecture

**No `silly_kicks/` surface change.** SkillCorner `match.json` parsing is
consumer/loader territory (ADR-001 hexagonal contract; raw-file parsing is exactly
the TF-23 native-loader concern, which is demand-gated). All new code is test +
scripts support.

### 3.1 Components

1. **`tests/_skillcorner_sample.py`** (new shared test support)
   - `SAMPLE_DIR` (env `SKILLCORNER_SAMPLE_DIR`, default temp path) + `MATCH_IDS`
     (the 10 A-League ids) + `find_artifact(match_dir, suffix)` — **factored out of
     the existing `tests/spadl/test_skillcorner_e2e.py`** (small, related DRY refactor;
     that test is updated to import them).
   - `build_skillcorner_gk_truth(meta: dict) -> dict[str, list[str]]`: pure function
     mapping `{str(team_id): [str(gk_player_id), ...]}` from `players[]` where
     `player_role.acronym == "GK"` (normally one per team — the starter). Teams with
     **zero** GK-acronym players are **omitted** from the dict (no anchor possible);
     the e2e reports and skips them. A per-team list (not a scalar) keeps the rare
     2-rostered-GK case robust. **Never raises** on cardinality.
   - `compare_gk_picks(truth, derived_picks, *, match_id, subset_allowlist=frozenset())
     -> AgreementResult`: pure comparator — the hexagonal seam that makes the §3.1.4 CI
     claim real (both the e2e and the CI unit test call THIS, so the real comparator is
     CI-covered and there is no second drifting code path). **Operates on ONE match**:
     `truth: dict[str, list[str]]` (that match's roster, keyed `str(team_id)`),
     `derived_picks: dict[tuple, list[str]]` (the 2nd element of `derive_goalkeepers`
     for that match's frames, keyed `(game_id, team_id)`). It **string-casts the team
     key on both sides** (truth keys are `str(team_id)`; derived keys carry `team_id` as
     `int` — e.g. `(1886347, 4177)` — so an un-cast lookup misses every time). The
     **allowlist is threaded here, not post-processed** (§4): for a team in
     `subset_allowlist` (a set of `(str(match_id), str(team_id))`), the rule is
     `truth[team] ⊆ set(picks[team])`; for every other team it is exact equality
     `set(picks[team]) == set(truth[team])`. Returns a frozen `AgreementResult` with
     per-team `matched` / `mismatched` / `no_roster_gk` breakdown + `short_name`
     diagnostics, and `is_perfect` (all teams-with-truth pass *their applicable* rule —
     so an allowlisted team does NOT flip `is_perfect`). `AgreementResult`s aggregate
     (`+` / a `merge` helper) so the e2e sums per-match results into one verdict.

2. **`scripts/download_skillcorner_sample.py`** (new)
   - Idempotent fetch of the 10 matches' `events` (`_dynamic_events.csv`),
     `metadata` (`_match.json`), and `tracking` (`_tracking_extrapolated.jsonl`)
     into `SAMPLE_DIR`, reusing `_loader_pining` fetch helpers + the public token.
   - Skips artifacts already on disk. Side benefit: **also unblocks the existing
     SkillCorner SPADL e2e**, which currently skips because the sample dir is empty.

3. **`tests/tracking/test_gk_skillcorner_roster_e2e.py`** (new, `pytest.mark.e2e`)
   - **Strictly per match** (do NOT merge truth dicts across matches): the 10 public
     matches are one league, so the same `team_id` recurs (probe teams 1805/4177 will
     reappear) and `truth` is keyed by bare `str(team_id)`. A merged truth dict would
     last-match-win and validate match A's roster GK against match B's frames. So: for
     each match, build `truth = build_skillcorner_gk_truth(match.json)`, build `frames`
     via the **reused** `_loader_pining` seam (§3.3 — NOT a fresh inline load+convert,
     which could omit pitch-dim normalization and trip `derive_goalkeepers`' SPADL-bounds
     guard, `_gk_identification.py:88-93`), `_, picks = derive_goalkeepers(frames)`,
     `result = compare_gk_picks(truth, picks, match_id=mid, subset_allowlist=...)`;
     then aggregate the per-match `AgreementResult`s.
   - **Join-key guard (loud, not skip):** before comparing, assert **every rostered GK
     id appears in the frame `player_id` set** for the match (a GK with no frames can't
     be derived → guaranteed false mismatch), plus a substantial overall overlap (most
     players; probe was 22/22). The id mapping was verified on 1/10 matches; any
     id-scheme drift on the other 9 must **fail loudly** as the structural problem it is,
     never masquerade as a GK mismatch or "no data".
   - Assert the aggregate `is_perfect` (every team-with-truth passes its applicable rule).
     On failure the message lists each team's `mismatched` with `short_name`. Teams with
     no roster GK are reported (`no_roster_gk`) and skipped (must be zero per the probe;
     defensive).
   - Per-match skip when artifacts (incl. tracking) absent; class-level skip when no
     data at all — mirrors `test_skillcorner_e2e.py`.

4. **`tests/tracking/test_gk_skillcorner_roster.py`** (new, regular suite — NOT e2e)
   - Synthetic, deterministic, no network. Exercises `build_skillcorner_gk_truth` on a
     hand-built `match.json`-shaped dict and `compare_gk_picks` directly (the SAME
     comparator the e2e uses). Covers: the normal one-GK mapping; a 2-rostered-GK team
     (both returned, no raise); a zero-GK team (omitted → `no_roster_gk`); the
     int-vs-str team-key cast; `compare_gk_picks` returning `is_perfect` for an exact
     match, `mismatched` for an over-identified pick (starter + an outfielder — the
     precision case the equality rule must catch), and a miss for a wrong pick; the
     **`subset_allowlist`** path (an over-identified allowlisted team passes via `⊆` and
     does NOT flip `is_perfect`, while a non-allowlisted over-identified team still
     fails). **Cross-match collision guard:** two synthetic matches sharing a `team_id`
     with *different* GKs, compared per-match then aggregated, must not cross-contaminate
     (the analogue of the composite-key guard) — a merged-truth implementation fails this
     test. Also a tiny synthetic frame set where a planted player dwells in the penalty
     area near goal so end-to-end `derive_goalkeepers` → `compare_gk_picks` is exercised
     once without network.
   - Rationale: the e2e does not run in CI (`-m "not e2e"`), so without this the
     comparator would be unexercised in CI — the "silently-skipping tests hide
     breakage" trap. Sharing `compare_gk_picks` makes that coverage real.

### 3.2 Data flow (e2e)

```
pining (download once) ── SAMPLE_DIR/{id}/{*_match.json, *_tracking_extrapolated.jsonl, *_dynamic_events.csv}

PER MATCH (never merge truth across matches — team_ids recur in the A-League sample):
   match.json ──► build_skillcorner_gk_truth ──► {str(team_id): [gk_id,...]}        ┐
   tracking  ──► _loader_pining.build_skillcorner_frames(paths, N) ──► frames ──►   │
                          _, picks = derive_goalkeepers(frames)  # picks: {(game_id, team_id): [...]}
                          compare_gk_picks(truth, picks, match_id, allowlist) ──► AgreementResult(match)
AGGREGATE: sum per-match AgreementResults ──► assert is_perfect
```

### 3.3 Frame-building reuse (DRY + coord-correctness)

`_loader_pining._build_skillcorner(paths, match_id, tracking_limit)` already does the
proven `kloppy.skillcorner.load → tracking.kloppy.convert_to_frames → _preprocess`
pipeline that yields SPADL-bounds (0-105 / 0-68) frames. Extract a focused seam
`build_skillcorner_frames(paths, tracking_limit) -> frames` and have `_build_skillcorner`
delegate to it (its `(actions, frames, home)` return contract is **preserved
byte-for-byte** — calibration is unaffected). The e2e calls `build_skillcorner_frames`
so there is ONE frame-construction path, not a second divergent one. (The probe confirmed
the preprocessed seam yields in-bounds coords + correct picks, so this is DRY + insurance,
not a bug fix.)

`_build_skillcorner` already builds `frames` (from the tracking dataset) and `actions`
(from the events CSV) from **independent sources**, so the extraction cannot perturb
`actions` — the equality guard is really protecting the `frames` sub-expression. The
guard asserts `frames.equals(frames_via_seam)` (and `home`/`actions` identity) on a
capped sample, so a future drift in the seam is caught.

## 4. Assertion design & edge cases

- **Default rule: exact set equality per team** — `set(derived_picks[team]) ==
  set(truth[team])`. This is what the probe empirically supports (one derived pick ==
  one rostered GK, 2/2) and — unlike recall-only `∈` — it catches **over-identification**
  (the starter plus spurious outfielders), which is exactly what a Tier-1 "picks the
  right GK" gate must catch. `derive_goalkeepers` Stage 2a can return multiple players
  (`_gk_identification.py:142-146`), so this precision bound is load-bearing.
- **Genuine sweeper-keeper exceptions are allowlisted inside the comparator, not
  post-processed.** If a real unprobed match yields a legitimate multi-pick (e.g. a
  sweeper detected alongside the starter), add that specific `(match_id, team_id)` to the
  `subset_allowlist` passed to `compare_gk_picks` **with a written justification**; the
  comparator then applies `truth[team] ⊆ set(picks[team])` for that team and exact
  equality everywhere else, and `is_perfect` honours it (so the gate logic lives in the
  one CI-tested place — the e2e never re-derives a second rule). Precision can't silently
  rot: every non-allowlisted team keeps strict equality.
- **Frame cap `N = 8000`** (probe-confirmed: correct picks at 8000 frames; bounds
  runtime over 10 tracking files). The starter plays from frame 0, so an early-frame cap
  also *anchors* the starter truth and side-steps later GK substitutions.
- **GK substitution / red card** (rare): the starter still dwells in-goal for the capped
  window, so equality holds; documented as a known edge handled by the allowlist if a
  real match violates it (a finding to handle, not a silent pass).
- **ID dtypes (ADR-019-aware):** join `player_id` on `str(...)` both sides (frame
  `player_id` is `object`, roster `id` is `int`); AND string-cast the **team key** both
  sides in `compare_gk_picks` (truth `str(team_id)` vs derived `(game_id, int team_id)`).

## 5. Testing / TDD order

1. **Red→green (unit):** write `tests/tracking/test_gk_skillcorner_roster.py` first
   → implement `build_skillcorner_gk_truth` + `compare_gk_picks` (+ `AgreementResult`)
   in `tests/_skillcorner_sample.py` → green.
2. **Reuse seam:** extract `build_skillcorner_frames` in `_loader_pining`; assert the
   delegated `_build_skillcorner` path is unchanged.
3. **e2e:** write `test_gk_skillcorner_roster_e2e.py`; run
   `download_skillcorner_sample.py`; run the e2e against real data → green, OR it
   surfaces a genuine `derive_goalkeepers` gap → systematic-debugging → fix **within
   this work** (a red e2e is in scope, not deferred); a genuine sweeper multi-pick →
   allowlist with justification (§4).
4. Full suite `-m "not e2e"` + `ruff format --check` + `ruff check` + `pyright
   silly_kicks/` stay green (no library change expected, but verified).

## 6. Documentation closure (part of this work)

On a green e2e, this PR also:
- **ADR-007** — record SkillCorner as Tier-1 external-roster-validated (was Tier-2
  self-consistency); Metrica stays the documented permanent limitation.
- **CLAUDE.md** — update the PR-S26 tracking note ("only Tier-2 … extrapolated-
  SkillCorner" → SkillCorner Tier-1).
- **TODO.md** — remove the TF-27 SkillCorner row (validated/closed), per the
  "don't leave closed items in TODO sections" rule.
- **CHANGELOG.md** + version bump (patch — test/validation + docs; no `silly_kicks/`
  behaviour change) across `pyproject.toml` / `__init__.py` / `TODO.md` /
  `CHANGELOG.md` (version-bump hard gate). The CHANGELOG explicitly notes the
  `_loader_pining._build_skillcorner` → `build_skillcorner_frames` **seam refactor**
  (guarded byte-for-byte) so a future calibration regression has a breadcrumb.

## 7. Genuinely out of scope (on merit, not deferral)

- **A `silly_kicks/` SkillCorner `match.json` parser** — architecturally wrong
  (ADR-001; the TF-23 native-loader concern). Belongs in the loader/consumer layer.
- **Metrica external verification** — impossible on public anonymized data
  (no roster). Permanent (ADR-007).
- **Speculative `derive_goalkeepers` rewrite** — the algorithm is validated-correct;
  changing it without a failing case is undisciplined. (A failing case *is* in scope.)

## 8. References

- ADR-007 (derived GK identification, validation tiers).
- ADR-001 (converter/identifier conventions — parsing stays out of the library).
- ADR-019 (id-dtype safety at tracking seams — the string-cast join).
- `tests/spadl/test_skillcorner_e2e.py` (the e2e convention being mirrored).
- pining-for-the-data: public token serves SkillCorner; roster lives inside
  `match.json` `players[].player_role`.
