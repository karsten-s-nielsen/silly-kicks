# `secured_reception` robust chronological sort — Design + Plan

- **Date:** 2026-09-19
- **Status:** Draft — pending independent (lakehouse) review
- **Author:** silly-kicks maintainer session (Claude Opus 4.8)
- **Target release:** **PATCH `4.119.1`** (owner-decided 2026-09-19). Re-confirm `v4.119.1` is free at commit-prep (a concurrent sk session may ship first).
- **Decision:** ADR-065 amendment (retrofit a missed §3d mart-reading consumer). A new ADR-099 is optional (owner's call) — this applies an existing rule, it is not a new decision.
- **Reported by:** luxury-lakehouse 4.119.0-adoption cycle (`add_packing` in AC enrich raises on a full IDSSE/DFL match). Investigation: `D:\Development\_reviews\2026-09-19-sk-add-packing-chronological-bug-investigation.md`.

## 1. Motivation

`add_packing` → `tracking/_packing.py:326` (`secured_reception`) raises
`ValueError: time_seconds must be non-decreasing within each (game_id, period_id) group`
on a persisted action mart whose `action_id` is non-chronological. This blocks the lakehouse AC re-materialize (`add_packing` runs per IDSSE match).

**`secured_reception` is a mart-reading, window-scan `add_*` consumer that ADR-065 §3d's robust-sort retrofit MISSED.** ADR-065 §3d established that such a consumer must order by the robust key `(game_id, period_id, time_seconds, action_id)` via `_sort_actions_chronological_or_action_id`, NOT `action_id` alone, **precisely because a persisted mart bypasses the `_finalize_output` guard and may carry non-chronological `action_id`** (verbatim in that helper's docstring, `spadl/utils.py:1614-1622`). The §3d retrofit landed on 6 consumers (`spadl/utils.py:204/426/637/1151` + atomic mirrors) — but not on `secured_reception`.

**Fresh conversions are NOT the problem** (correcting the lakehouse's diagnosis): sk 4.119.0 `spadl.sportec.convert_to_actions` sorts chronologically (`sportec.py:656`), assigns `action_id` chronologically (`:666`), and routes through `_finalize_output` (the CALL at `sportec.py:678`), which runs `_assert_chronological_action_id` (`utils.py:1634`; the raise is `utils.py:1660`). It emits chronological `action_id` or raises at convert. The failing data is a persisted/stale mart (§3d condition), not fresh-convert output.

## 2. Root cause — a scan-vs-anchor ORDER MISMATCH (not just a strict guard)

`secured_reception` (`_packing.py:233`):
- resolves its reception ANCHOR via `receiver_pos = _resolve_next_touch_positions(actions)` (`:288`), which sorts by the **robust §3d key** (`utils.py:1297-1305`) and resolves next-touch positions by `.shift(-1)` in that CHRONOLOGICAL order (returned input-position-aligned);
- but then SCANS the secured window in **`action_id`-alone order** (`:322` `grp.sort_values("action_id")`) and builds `rank`/`idx` from it, asserting non-decreasing time (`:325-326`).

On a chronological mart these two orders coincide. On a **non-chronological** mart they DIVERGE: anchors are resolved in time order, the window is scanned in `action_id` order — so even without the raise the labels would be **wrong**, not merely crash-prone. The `:326` raise is a fail-fast masking the mismatch. The correct fix is to scan in the SAME robust order the anchor helper already uses; the raise then becomes unnecessary.

Second, independent strictness: `_assert_chronological_action_id` excludes non-finite `time_seconds` (`utils.py:1653-1655`); `secured_reception:325` runs `np.diff` over the whole group including NaN → `NaN >= -1e-9` is False → raises. So it also rejects NaN-time actions the converter tolerates.

## 3. Design

**One change, one function.** In `secured_reception`, order the per-group scan by the robust key `_resolve_next_touch_positions` uses, and drop the raise:

- `_packing.py:322`: replace `grp.sort_values("action_id", kind="stable")` with `_sort_actions_chronological_or_action_id(grp)` (import from `spadl.utils`) — key `(game_id, period_id, time_seconds, action_id)`, stable, NaN-time last. This makes `idx`/`rank` the SAME chronological order the anchors were resolved in.
- `_packing.py:325-326`: **remove the raise.** The robust sort makes `time_seconds` non-decreasing over the finite rows by construction; NaN-time rows sort last and resolve to `<NA>` (a coverage token, not a violation) — matching `_assert_chronological_action_id`'s finite-only semantics.
- Update the now-stale comments (`:264-266`, `:319-321`): the scan follows the **robust chronological order the positions helper resolves in**, not `action_id` alone.

**Correctness invariant (the load-bearing point):** `secured_reception`'s scan order == `_resolve_next_touch_positions`'s resolution order (both = the robust key). This is what the fix restores; a test pins it.

**Scope — verified minimal:**
- Only `secured_reception` is order-sensitive in `_packing`. `compute_packing_metrics` (`:119`) is per-row geometry (order-insensitive); the other `_packing` sort (`:302`, on `_pos_tf49`) is unrelated.
- The atomic packing path reuses the shared `_packing.secured_reception` (no separate atomic `secured_reception`; `atomic/spadl/utils.py` already uses the robust helper) — so this one fix covers atomic too. **Reviewer/impl to confirm** the atomic `add_packing` mirror calls the shared function and needs no second edit.

## 4. Contract / impact

- **No retrain, no re-materialize, C4-free.** On already-chronological input (every FRESH sk conversion, and any mart the §3d guard would pass) the robust sort is a no-op and the raise never fired → **byte-identical secured labels**. New behaviour occurs ONLY where `secured_reception` previously CRASHED (non-chronological / NaN-time mart) — a crash→correct-value fix, not a value change on working inputs.
- **Public surface unchanged** — `add_packing` / `secured_reception` signatures unchanged; `packing_xfns` unchanged (and remains in no default xfn list, ADR-039).
- Unblocks the lakehouse AC re-materialize on non-chronological persisted marts.

## 5. Tests (TDD, red-first)

- **Regression (red-first):** a `secured_reception` fixture with (a) finite-descending `action_id` (e.g. action_id 1 @ t=2814.500, 2 @ 2814.106) and (b) a NaN-`time_seconds` row → on the OLD code raises at `:326`; on the fix returns correct tri-state labels (NaN-time → `<NA>`). Watch it fail first.
- **Consistency (the invariant):** on the non-chronological fixture, assert `secured_reception`'s scan order equals `_resolve_next_touch_positions`'s resolution order for the same group (both keyed on `(game_id, period_id, time_seconds, action_id)`) — the property whose absence caused wrong labels. A deliberate reintroduction of `sort_values("action_id")` must flip a label (non-vacuity).
- **Parity (no-op on chronological):** on an already-chronological fixture, secured labels are **byte-identical** before/after the change (guards the no-retrain claim). Prefer a committed real chronological slice (e.g. from `tests/datasets/sportec/idsse_slice/` post-`convert_to_actions`, which is chronological by construction).
- Full `pytest -m "not e2e"`, ruff, pyright green.

## 6. Non-goals

- **Not** changing `convert_to_actions` / `action_id` numbering — sk already assigns `action_id` chronologically (ADR-065); the lakehouse's "renumber in convert" alternative is redundant and mis-targets the layer.
- **Not** re-materializing the lakehouse corpus — that is the lakehouse's own LH-SPEC-02 live-confirm (this crash CONFIRMS their corpus carries non-chronological `action_id`); the sk fix makes `add_packing` robust regardless.
- **Not** touching the converter-boundary `_assert_chronological_action_id` raise — fail-fast at conversion is correct; only the mart CONSUMER must robust-sort (§3d).

## 7. Files

- `silly_kicks/tracking/_packing.py` — `secured_reception` (`:322` sort → robust helper; remove `:325-326` raise; comment updates; add the `_sort_actions_chronological_or_action_id` import).
- `tests/tracking/test_packing*.py` (or a new `test_packing_secured_reception_order.py`) — the 3 tests above.
- `CHANGELOG.md` (Fixed entry), `CLAUDE.md` (amend the ADR-065 §3d bullet to add `secured_reception`/`_packing` to the retrofitted mart-reading-consumer list — it was omitted), `silly_kicks/_version.py` (next-free, at commit-prep). Optional `docs/superpowers/adrs/ADR-099-*` if the owner wants a standalone record; otherwise an ADR-065 amendment note.

## 8. Open items — post-review status (lakehouse APPROVE 2026-09-19)

- **OI-1 — atomic mirror: RESOLVED.** No separate atomic `secured_reception`. Std `tracking/features.py:88` imports `secured_reception` from `._packing`; std `add_packing` calls it (`:1672`). Atomic `atomic/tracking/features.py:488` `add_packing` delegates to the std `add_packing` (`:509`, `_std`) and drops `packing_secured` (numeric columns only) — so it reaches the SAME shared `_packing.secured_reception` via `_std`, or not at all. One fix in `_packing.secured_reception` covers std + atomic; no second edit.
- **OI-2 — NaN-time window arithmetic:** impl-time verification — confirm the `t_last`/`secured_window_seconds` arithmetic (`_packing.py:327+`) resolves trailing NaN-time rows to `<NA>` (never NaN into a finite row's label). Pinned by the regression fixture's NaN-time row.
- **OI-3 — ADR form: ADR-065 amendment** (add `secured_reception`/`_packing` to the §3d retrofit list) — no new ADR needed (this applies an existing rule). Owner may still elect a standalone ADR-099.
- **OI-4 — version: RESOLVED = PATCH `4.119.1`** (owner-decided 2026-09-19). Re-confirm `v4.119.1` free at commit-prep.
- **PKSR-SPEC-01 (reviewer):** reconciled — §1 now states `sportec.py:678` is the `_finalize_output` CALL and the guard raise is `utils.py:1660` (via `_assert_chronological_action_id`, `utils.py:1634`).
- **PKSR-SPEC-03 (reviewer, optional): DECLINED** — a finite-only post-sort runtime assert is redundant-by-construction after the robust sort (it cannot fire → a "guard that cannot fail", discouraged per repo discipline); the §5 consistency test is the guard. Owner may override.

## 9. Commit / approval gate

Single feature branch off `main` (e.g. `fix/packing-secured-reception-robust-sort`). One coherent, fully-tested commit. **No `git commit` / `git push` / tag / publish without explicit owner approval** for that action; approval of this doc is not commit authority; never tag before CI green; owner publishes. On release, reply to the lakehouse with the version to pin.
