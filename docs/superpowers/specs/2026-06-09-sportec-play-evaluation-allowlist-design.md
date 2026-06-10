# Design: DFL `play_evaluation` success-allowlist (sportec completion robustness)

**Date:** 2026-06-09
**Status:** Draft v3 — incorporates external review rounds 1+2 (part-deux session, 2026-06-09) →
pending user review → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** TODO.md "DFL `play_evaluation` full vocabulary (4.20.1 follow-up)". Refines the 4.20.1
BUG-2 fix (sportec pass/set-piece completion from native DFL `play_evaluation`).

## Context

The 4.20.1 BUG-2 fix made sportec pass/set-piece completion read the native DFL `Evaluation`
attribute (`play_evaluation`): `result = fail` iff the token is **exactly** `"unsuccessful"`, else
`success`. The TODO asked for a "larger pull" to confirm no reason-coded failure tokens (e.g.
`unsuccessfulBecauseOfFoul`) exist league-wide — any such are currently left `success`.

### Two facts that reframe the task

1. **The "larger pull" is unobtainable.** The accessible DFL corpus is the public IDSSE set =
   **7 Bundesliga matches** (pining's IDSSE is those same 7; `play_evaluation` is a raw bronze field
   absent from the gold mart). There is no larger DFL corpus to probe (confirmed with user
   2026-06-09). The TODO's *discovery* premise cannot be satisfied — mirrors the xS-refit
   premise-falsified pattern. **But the 7 matches we already have are still usable** as a regression
   tripwire (see the owner-gated e2e below) — that is a different purpose than a discovery pull.

2. **A success-allowlist obviates the discovery pull.** kloppy — the reference DFL/Sportec parser,
   maintained against the full DFL format spec — resolves pass completion with a **success-allowlist**
   (`kloppy/infra/serializers/event/sportec/deserializer.py:390`):

   ```python
   if event_chain["Play"]["Evaluation"] in ("successfullyCompleted", "successful"):
       result = PassResult.COMPLETE
   else:
       result = PassResult.INCOMPLETE   # everything else → fail
   ```

   Instead of enumerating every possible *failure* token, treat the small, stable *success* set as
   the allowlist and fail everything else. Any unseen reason-coded failure is then handled **by
   construction**. The authoritative success vocabulary is `{successfullyCompleted, successful}`
   (kloppy-confirmed; the 4.20.1 lakehouse run independently observed exactly
   `{successfullyCompleted, successful (×23), unsuccessful, NULL}` over 10,497 events —
   `tests/spadl/test_sportec_completion.py:7,44`).

This also **aligns the native sportec converter with the kloppy gateway** — today the two sportec
paths could disagree on identical DFL data; they would now agree.

## Decision

Adopt a **guarded success-allowlist** (Approach A) with a warn-on-unknown net, single-sourcing both
**extraction and classification**, plus an **owner-gated e2e** over the 7 IDSSE matches and a
committed-fixture CI regression test. We rely on the kloppy source as the vocabulary authority.

### The predicate

silly-kicks cannot adopt kloppy's predicate verbatim: kloppy assumes `Evaluation` is always present
(`else → fail`), but silly-kicks supports `play_evaluation` being **optional**. A naive allowlist
(`fail iff not in success-set`) would map **every** pass to `fail` when the column is absent
(non-DFL sportec-like data) — catastrophic. So fail only on a **non-empty, non-success** token:

> **`result = fail` iff `(pass / set-piece) AND value is non-empty AND value ∉ {successfullyCompleted, successful}`.**

| `play_evaluation` | today | new |
|---|---|---|
| `successfullyCompleted` / `successful` | success | success |
| `unsuccessful` | fail | fail |
| `unsuccessfulBecauseOfFoul` (unseen reason-code) | **success** ← the gap | **fail** |
| empty / null / column absent | success | success (guarded) |
| benign non-empty token (hypothetical) | success | fail (+ warns) |

On the IDSSE/DFL data we have, the only non-success token is `unsuccessful`, so output is
**byte-identical** — this is robustness hardening, not a re-mapping. Effectively **not a retrain
trigger** on observed data; the Hyrum surface (a DFL stream carrying failure tokens beyond
`unsuccessful` would shift its fail distribution) is noted in the CHANGELOG and asserted absent by
the e2e.

## Components

### 1. Three module-level helpers — `silly_kicks/spadl/sportec.py`

Single source of truth for the whole `play_evaluation` seam (mirrors the ADR-018 `_is_goal` /
`_is_owngoal` pattern). Both call sites route through **all three** — extraction *and*
classification *and* the warn are single-sourced, so no normalization or vocabulary can drift:

```python
_SUCCESS_EVAL = ("successfullyCompleted", "successful")   # authoritative DFL success set (kloppy-confirmed)
_KNOWN_EVAL = frozenset(_SUCCESS_EVAL) | {"unsuccessful", ""}  # everything we expect; anything else → warn

def _extract_play_eval(df: pd.DataFrame) -> np.ndarray:
    """Normalize the optional DFL Evaluation column to a clean str array (absent/null → "")."""
    if "play_evaluation" in df.columns:
        return df["play_evaluation"].fillna("").astype(str).to_numpy()
    return np.full(len(df), "", dtype=object)   # len(df), not a caller-passed n (mismatch-proof)

def _play_evaluation_is_fail(play_eval: np.ndarray) -> np.ndarray:
    """Non-empty, non-success DFL Evaluation → fail (success-allowlist; kloppy-aligned).

    Empty/absent/null → not-fail (the success default), so a missing column never mass-fails
    non-DFL sportec-like data. Exact camelCase match (DFL is consistent camelCase; a case-variant
    would fail+warn by design — deliberately NOT `.str.lower()` like sibling qualifiers).
    """
    return (play_eval != "") & ~np.isin(play_eval, _SUCCESS_EVAL)

def _warn_unexpected_play_eval(play_eval: np.ndarray) -> None:
    """Surface any token that is neither a success token nor the known `unsuccessful` failure."""
    unexpected = set(np.unique(play_eval)) - _KNOWN_EVAL
    if unexpected:
        warnings.warn(
            f"sportec: unexpected play_evaluation token(s) {sorted(unexpected)} treated as fail "
            f"(not in success allowlist {_SUCCESS_EVAL}); verify against the DFL spec.",
            stacklevel=2,
        )
```

- **Site 1** (`_build_raw_actions`, `~855-858`, open-play pass + FreeKick / Corner / ThrowIn / GoalKick):
  `play_eval = _extract_play_eval(df)` (replaces the `_opt("play_evaluation", "")` closure use);
  `is_eval_fail = _play_evaluation_is_fail(play_eval)`; `result_ids[is_pass_or_setpiece & is_eval_fail] = fail`;
  `_warn_unexpected_play_eval(play_eval[is_pass_or_setpiece])`.
- **Site 2** (`_synthesize_gk_distribution_actions`, `~1069-1077`, synth punt-goalkick / throwOut-pass
  inheriting the parent Play's eval): `synth_eval = _extract_play_eval(src)`;
  `result_ids_synth = np.where(_play_evaluation_is_fail(synth_eval), fail, success)`;
  `_warn_unexpected_play_eval(synth_eval)`.

**Warn coverage (M1 fix).** A naive single warn over `is_pass_or_setpiece` would miss synth parents:
a punt Play has `play_goal_keeper_action="punt"` → excluded from `is_pass` (`sportec.py:804`,
`is_pass = is_play & (play_gk == "") & ...`), so a novel token appearing only on a punt parent would
be failed by the synth path but never surfaced. Warning at **both** sites over each site's relevant
rows (main pass/set-piece tokens at Site 1; synth-parent tokens at Site 2) covers the disjoint sets.

### 2. Tests

**Unit (regular suite, CI-everywhere) — `tests/spadl/test_sportec_completion.py`.** TDD, red-first,
**only the genuinely-new cases** (`successful` and `""` are already covered at lines 44-45):
- `unsuccessfulBecauseOfFoul` (unseen reason-code) → **fail** — the headline behavior.
- `play_evaluation` column **absent** → no mass-fail (all pass/set-piece stay success).
- a benign non-empty token → fail **and** `pytest.warns(UserWarning)`.
- synth punt-goalkick with an unseen failure token → fail (parity with the main path).
- a behavioral single-source guard: a fixture exercising both the main and synth paths yields the
  identical eval→result mapping (forbids the two sites drifting).

**Committed-fixture regression (CI-everywhere, M3) — `tests/spadl/test_sportec_completion.py`.**
The native converter consumes a pre-parsed **DataFrame**, not XML (there is no native DFL-XML parser
— that is the deferred TF-23), so the committed regression uses the existing `_ev(...)` native-shape
builder (`test_sportec_completion.py:22`), **not** the kloppy-gateway `sportec_events.xml`. Build the
full observed distribution `{successfullyCompleted, successful, unsuccessful, ""}` + one reason-coded
token, and assert the per-row `result_id`s are exactly the allowlist mapping (byte-identical to the
exact-match converter on the clean subset, `fail` on the reason-coded token) and the warn is silent
on the clean subset — locking "robustness hardening, not re-mapping" into every CI leg. (A true
"native == kloppy-gateway agreement" test would need the same match in both shapes, not cleanly
available from committed fixtures — the bronze e2e covers the native side; kloppy's own allowlist
covers the gateway side.)

**Owner-gated e2e (H1) — `tests/spadl/test_sportec_playeval_e2e.py` (new).** `@pytest.mark.e2e`,
gated on `DATABRICKS_HOST/HTTP_PATH/TOKEN` + `databricks-sql-connector` importable (the discipline of
the shipped `tests/test_xthreat_nll_lakehouse_e2e.py`; skips in public CI). **Source = Databricks
`bronze.idsse_events`, not pining** — pining's IDSSE loader (`_loader_pining._build_idsse`) parses via
the **kloppy gateway** (`sportec.load_event` → `spadl_kloppy.convert_to_actions`), which consumes the
`Evaluation` attribute internally and never surfaces `play_evaluation` to the **native** converter
this PR changes. The native converter is only fed real DFL events on the Databricks bronze path
(`_loader_databricks._convert(provider="idsse")` → `sportec_spadl.convert_to_actions`), and
`bronze.idsse_events` carries `play_evaluation` (verified live, 1 of 247 cols).

Wiring (events-only — `result_id` from `play_evaluation` is orientation-independent, so the heavy
tracking pull + orientation derivation in `load_matches` are unneeded): a thin
`SELECT … FROM <_table("idsse","events")> WHERE match_id IN (…)` (a small `fetch_idsse_events` read
helper mirroring `fetch_action_values`, pure I/O), then per match
`filter_extratime_frames(ev, …)` (defensive; Bundesliga has no ET) →
`sportec_spadl.convert_to_actions(ev, home_team_id=<events team_id mode>, home_team_start_left=True)`
(`home_team_id` is orientation-only — any consistent value leaves `result_id` unchanged). Assert
across all 7 matches:
- **No unexpected-token warning fires** (`warnings.catch_warnings(record=True)`) — positive proof the
  allowlist `∪ {unsuccessful}` covers the real vocabulary (the byte-identical condition).
- The observed non-success `play_evaluation` token set **== {"unsuccessful"}** exactly.
- Liveness band: goalkick fail-rate ∈ **[0.05, 0.60]** (catches both an all-success regression — the
  original BUG-2 — and an all-fail regression; loose to stay non-flaky on 7 matches; CLAUDE.md notes
  DFL goalkicks ~71% complete). Fills a standing gap — there is **no** committed/owner-gated
  sportec/IDSSE e2e today (the 4.20.1 10,497-event check was a one-off).

Before writing the assertions: a sanity probe confirming `play_evaluation` is populated on match 1
through the bronze→native path (the "0 unmapped"-style check).

## Out of scope

- Shots (own outcome handling via `shot_outcome_type`; `play_evaluation` does not apply) — unchanged.
- The kloppy gateway path (already a success-allowlist; this change makes the **native** converter agree).
- A discovery re-pull of DFL data (unobtainable; kloppy is the authority, the e2e + warn are the nets).
- Other providers — sportec/DFL-specific.

## Housekeeping (one feature branch, one commit)

- Single commit on `pr-s90-sportec-play-evaluation-allowlist`; explicit per-commit approval + the
  git-commit sentinel.
- Run `/final-review` before committing.
- Remove the "DFL `play_evaluation` full vocabulary" item from `TODO.md` (Technical Debt → Blocked or
  Deferred).
- **Version: 4.21.3** (4.21.2 was taken by the parallel session's xT-NLL e2e; reconcile at commit
  time per the PR-S88 precedent). Bump across the version sites + `uv lock` + a dated CHANGELOG entry
  noting: success-allowlist, **kloppy-aligned**, **byte-identical on observed DFL data** (robustness
  hardening; Hyrum surface = a DFL stream with failure tokens beyond `unsuccessful`).
- No ADR / NOTICE (refines the 4.20.1 BUG-2 fix; no new methodology). C4-free.

## Verification

1. Default suite green: `python -m pytest tests/ -m "not e2e and not slow" -q` (new completion + fixture
   regression tests + the existing `test_sportec_completion.py` + the broader sportec suite).
2. Full CI lint locally: `ruff check` + `ruff format --check` (whole tree) + `pyright silly_kicks/`.
3. e2e collects-but-skips without `DATABRICKS_*` / the connector (not errored).
4. Owner run (`DATABRICKS_HOST/HTTP_PATH/TOKEN`, in an **isolated env with the connector** — never the
   main `.venv`, whose `pandas<2.3.0` connector conflict ABI-breaks it): the 7-match e2e is
   warn-silent, non-success tokens == {"unsuccessful"}, goalkick fail-rate ∈ [0.05, 0.60].
