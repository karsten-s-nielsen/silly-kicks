# Consistent extra-time direction handling across per-period-absolute converters — design

**Date:** 2026-05-30
**Status:** Design — **v3**, lakehouse review **rounds 1 + 2 complete** (2026-05-30). SemVer decided: **4.0.0, fail-loud** (user, 2026-05-30). §9 resolved. Companion: **ADR-010**. Remaining round-2 polish (D–J) lands during implementation.
**Type:** Correctness behaviour change in an active production code path. **SemVer = 4.0.0 (major)** — see §5.
**Sequence:** Surfaced by the TF-24 calibration dry-run. The pining-loader `_apply_et_direction` interim fix is **calibration-only** (see §7) and independent of the library decision here. Decision record: **ADR-010** (silly-kicks) + a paired lakehouse ADR.

## 1. Problem (fact-checked against current source)

Per-period-absolute providers flip coordinates **per period** by the home team's start direction (ADR-006). Extra time (periods 3/4) needs a separate `home_team_start_left_extratime` flag. The native per-period-absolute converters handle a **missing** ET flag **inconsistently** — and crucially, the **message wording** diverges even among the converters that already raise:

> **Source-audit correction (implementation, 2026-05-30):** an earlier draft of this table claimed Sportec **events** + Metrica **events** "silently default". That is **not** true against the live source: both gained an ET-period guard in **PR-S23 / silly-kicks 3.0.1** ("Sportec + Metrica per-period direction-of-play correctness") and already **raise** — just with per-converter ad-hoc message wording. The **only** genuinely silent converter is **Sportec tracking**. The table below reflects the verified state.

| Converter | File:line | ET-without-flag behaviour (verified) |
|---|---|---|
| GS **tracking** | `tracking/gradientsports.py:114` | **raises `ValueError`** (ad-hoc wording) |
| GS **events** | `spadl/gradientsports.py:326` | **raises `ValueError`** (ad-hoc wording) |
| Sportec **tracking** | `tracking/sportec.py:132` (via `direction.home_attacks_right_per_period`) | **silent default** (p3→`bool(None)=False`, p4→`True`) — the one genuine gap |
| Sportec **events** | `spadl/sportec.py:510` (guard since 3.0.1) | **raises `ValueError`** (ad-hoc wording) |
| Metrica **events** | `spadl/metrica.py:159` (guard since 3.0.1) | **raises `ValueError`** (ad-hoc wording) |
| kloppy gateway (SkillCorner/Metrica tracking) | — | unaffected (kloppy normalises orientation upstream) |

So **Sportec tracking fails silent** (wrong ET coordinates, no signal — the dangerous path), while the other four raise but with **four divergent messages**. This change closes the silent gap **and** unifies all five behind one shared guard / one message.

**Empirically confirmed (TF-24 dry-run, 2026-05-30):** a GS match with ET crashed first the GS **tracking** converter, then (after the loader fix) the GS **events** converter — i.e. both GS guards fire. The implementation source-audit (with `git log -L`) then confirmed Sportec/Metrica **events** also raise (since 3.0.1) and that **Sportec tracking** is the lone silent sibling.

## 2. Decision

Add a **public, shared** guard and apply it symmetrically so **all** per-period-absolute converters (Sportec + Metrica, tracking **and** events) **raise** on ET-without-flag, consistent with GS. Correctness-first: never silently ship wrong geometry.

```python
# silly_kicks/tracking/direction.py  (PROMOTED to public — the lakehouse imports it for pre-flight self-validation)
def require_et_direction(period_ids, home_team_start_left_extratime, *, source: str) -> None:
    """Raise ValueError if ET periods (3/4) are present but the ET start direction is unset."""
    if home_team_start_left_extratime is None and pd.Series(period_ids).isin([3, 4]).any():
        raise ValueError(
            f"{source}: data contains ET periods (period_id in {{3, 4}}) but "
            "home_team_start_left_extratime was not provided. Set it from the match metadata "
            "(e.g. homeTeamStartLeftExtraTime), or filter ET out before converting."
        )
```

Applied at: `tracking/sportec.py`, `tracking/gradientsports.py` (refactor its inline raise to the shared guard), `spadl/sportec.py`, `spadl/gradientsports.py` (refactor), `spadl/metrica.py`. **Cross-provider parity:** identical exception type + message format everywhere (asserted by a parametrised test — §6).

**Public surface (lakehouse review minor a):** `require_et_direction` is exported from `silly_kicks.tracking` (and re-exported from `silly_kicks.spadl` for the events side) so the lakehouse can pre-flight-validate a batch before calling the converters, and the cross-repo **sentinel** (§4) can call it.

## 3. Affected surface

### silly-kicks (this change)
- **Full module rename `tracking/_direction.py → tracking/direction.py`** (review B — single public home, no private/public mirror). All importers (`tracking/sportec.py`, `tracking/gradientsports.py`, `spadl/{sportec,gradientsports,metrica}.py`, any others) update their imports. `require_et_direction` lives here, re-exported from `silly_kicks.tracking` (+ `silly_kicks.spadl` for events) + tests.
- `tracking/sportec.py`, `spadl/sportec.py`, `spadl/metrica.py` — add the guard (**breaking for ET matches** processed without the flag).
- `tracking/gradientsports.py`, `spadl/gradientsports.py` — refactor inline raise → shared guard (no behaviour change).
- CHANGELOG + the §6 regression gates.

### luxury-lakehouse (Phase A — separate lakehouse PR; see §4)
`src/analytics/action_context/pipeline.py` passes **no** ET flag:
- **line 130 (GS)** — **already crashes today** on any GS ET match (pre-existing latent prod bug; fix ASAP, independent of this spec).
- **line 89 (Sportec/IDSSE)** — currently **silently mis-orients** ET; after this change **raises**.

## 4. Cross-repo coordination — hard sequence + sentinel (lakehouse review point 2)

A breaking change across two repos needs an ordered, CI-enforced sequence, **not** "lockstep":

- **Phase A.0 — bronze schema prerequisite (review C), conditional:** verify whether the **bronze metadata already carries the ET start direction per provider** (this is the same per-provider data inventory as the §8 audit). DFL/Sportec XML in particular is unverified. If absent today, Phase A.0 is a separate lakehouse PR that extracts the ET start direction from provider metadata into bronze (per-provider, per-source-file) **before** Phase A.1. If the field is already present, Phase A.0 is a no-op.
- **Phase A.1 — lakehouse PR (silly-kicks pin UNCHANGED):** `MatchMeta` gains an ET start-direction field (events + tracking); `pipeline.py` passes `home_team_start_left_extratime=` to **both** `convert_to_frames` and `convert_to_actions`. Tests for **both** ET-present and ET-absent paths. After Phase A.1, prod is already correct under the *current* silly-kicks (the flag is simply accepted today).
- **Phase B — silly-kicks ships the guard, then lakehouse PR-2:** bump the silly-kicks pin floor to the guard version. Because Phase A already passes the flag, the bump is mechanical and cannot break prod.
- **Sentinel test (lakehouse PR-2):** assert that, for any in-scope per-period-absolute provider (IDSSE/GS/Metrica), if `MatchMeta` lacks the ET field **and** any match in scope has ET periods, the pipeline raises loudly with a helpful pin-mismatch message. **Placement (review D): per-batch** — call `require_et_direction` at the existing per-batch `convert_to_frames` call site (cheapest plumbing, no new pipeline wiring; the per-batch validation cost is negligible). Converts "hope we ordered right" into "CI fails if we didn't."

## 5. SemVer — OPEN DECISION (lakehouse review point 5)

ET matches are production data, so adding a raise in an active path is a behavioural breaking change. Three options:

1. **Fail-loud now, minor (3.31.0).** Pragmatic (silly-kicks precedent); the Phase-A-first sequence + sentinel make it prod-safe. But strict SemVer says a behavioural break ≠ minor.
2. **Fail-loud now, major (4.0.0).** Honest SemVer for a breaking behaviour change; aligned with the no-silent-degradation rule (no warn window). **Recommended** — clean, and the coordination already makes it safe.
3. **One-release deprecation cycle.** 3.31 emits `DeprecationWarning` on ET-without-flag (visible in logs, non-crashing); 4.0 raises. Lets the lakehouse adopt over two PRs without coordinated breakage — **but ships fail-soft for one release window, violating the user's no-silent-degradation rule** ([[feedback prefers fail-loud]]).

**DECIDED (user, 2026-05-30): Option 2 — fail-loud now, 4.0.0.** The deprecation cycle's only advantage (decoupled adoption) is already provided by the Phase-A-first sequence + sentinel, *without* the one-release fail-soft window; and a major bump is the honest signal to **all** downstream consumers (the lakehouse is one of potentially many).

**Operational cost note (review E):** a silly-kicks 4.0.0 bump cascades wider in the lakehouse than a minor would — `pyproject.toml [spadl]` pin, 6 trainer `_REQUIRED_SK_MIN` constants, the enforcing test, 25+ PEP 723 wheel URLs (`bump_wheel.py`), the TF env spec, and the retrain orchestrator runtime assert. This is **scripted** (`bump_wheel.py`) and **not** a deciding factor — recorded for transparency.

## 6. Test plan (lakehouse review point 6)

- `require_et_direction`: raises on ET-present + None; no-op on no-ET; no-op when flag provided. Parametrised over `source`.
- **Cross-provider parity (review G):** parametrised over {sportec-tracking, gs-tracking, sportec-events, gs-events, metrica-events}: (a) all raise the **same exception type + message format** (message-template equality; catches drift); **and (b) happy-path equivalence** — **with** the flag provided, each produces coordinate-correct output and the flag actually takes effect equally (catches "guard fires identically but post-guard logic diverged silently").
- **Per-converter ET correctness:** correct per-period flip **with** the flag for both ET-start orientations; raises **without** it.
- **RT-only golden no-regress (review F):** commit **frozen golden parquet fixtures captured against silly-kicks 3.30** (= current production behaviour) under `tests/regressions/extratime/`; the gate asserts the post-change run output **== the frozen golden** for regular-time-only fixtures across all touched converters (and feeds future-release regression detection). This is how "before vs after" is tested in a single PR.
- **Events-side round-trip (explicit):** dedicated mirror tests for `spadl/sportec.py` + `spadl/metrica.py` (the silent-default converters gaining the guard).
- **Real-data ET fixtures (review H):** commit **at least one captured ET window per converter family** — one Sportec/IDSSE ET window, one GS ET window (frames + actions) — and run the per-converter ET tests against them alongside the synthetic 22-player fixtures. Real + synthetic together catch both logic bugs and real-data-shape bugs (the lakehouse AC-1 dead-ball committed-parquet pattern). Source from lakehouse bronze if silly-kicks lacks ET data.
- GS existing ET raise tests stay green after the shared-guard refactor.

## 7. Calibration vs production — explicit (lakehouse review point 4)

The `_apply_et_direction` ET-**filter** in `scripts/_loader_pining.py` (and to-be-added `scripts/_loader_databricks.py`) is acceptable **only** because calibration is sample-based — dropping ET frames loses a little signal, never correctness. **AC-1 production MUST NOT filter ET** — `compute_action_context` processes ET matches end-to-end and must **source the real ET flag** from bronze metadata (Phase A). To prevent the filter pattern leaking into production:
- **DECIDED (review J): `silly_kicks.tracking.utils.filter_extratime_frames` — public, calibration-labelled.** Docstring: *"Drop ET periods (3/4) from a frames DataFrame. Calibration / sampling only — production must source `home_team_start_left_extratime` from provider metadata via MatchMeta, NOT drop ET. Use `require_et_direction` (the public guard) to validate the production path."* Public-with-warning DRYs future loaders (TF-24, TF-25, …) that would otherwise reimplement it, while the docstring preserves the production-vs-calibration boundary.
- `require_et_direction` (the guard) is the production-path public surface; `filter_extratime_frames` is the calibration-path public surface. The TF-24 loader's inline `_apply_et_direction` collapses to `filter_extratime_frames`.

## 8. Historical-data audit + remediation (lakehouse review point 3 — NEW, lakehouse-executed)

Production has run the **silent-default Sportec/Metrica** path for months; some IDSSE/Metrica ET matches likely have wrong-geometry frames/actions already in `bronze.spadl_action_context` and every downstream consumer (marts, embeddings, ScoutGPT training data). Per the user's no-silent-degradation rule, discovering this retrospectively owes an audit + remediation:

1. **Audit query (lakehouse):** count matches with `period_id IN (3,4)` in bronze tracking + events, per provider. Report the count.
2. **Correctness assessment:** for each affected match, was the silent default (p3→False, p4→True) the *true* ET orientation (recover the real ET start direction from provider metadata, if available)? Quantify how many were actually mis-oriented vs accidentally-correct.
3. **Remediation:** re-process the mis-oriented matches post-fix; flag/invalidate any golden tests + downstream model-training artifacts derived from them.

This section is **lakehouse-owned** (their data + their compute); the spec records the obligation so it isn't silently skipped. Output of the audit (the count + mis-orientation rate) should feed back before the silly-kicks guard ships, so the blast radius is known.

## 9. Decisions (resolved) + hard ship gate

1. **SemVer:** ✅ **4.0.0, fail-loud** (user, 2026-05-30).
2. **Public-helper shape:** ✅ `require_et_direction` public (guard); ✅ `filter_extratime_frames` public, calibration-labelled (§7).
3. **`direction.py` shape:** ✅ full rename `_direction.py → direction.py` (review B, §3).
4. **Events-side scope:** ✅ guard applies to {sportec, metrica, gs} **events** + {sportec, gs} **tracking** (metrica tracking goes via kloppy here; its events path is in AC-1).
5. **ADRs (review A):** ✅ silly-kicks **ADR-010** authored (this PR); lakehouse authors a paired ADR (MatchMeta ET field + pre-flight sentinel + version-coupled coordination), cross-referencing ADR-010.
6. **HARD SHIP-BLOCKER (review I): ✅ SATISFIED (2026-05-30).** The lakehouse §8 historical audit completed and returned a **clean zero** — IDSSE 0 / Metrica 0 ET matches ever processed (the silent-default path never ran on ET data); GS 5 ET matches but GS already raised + carries `homeTeamStartLeftExtraTime` end-to-end. **No mis-oriented production data, no remediation owed.** 4.0.0 may publish normally. (Audit memo: lakehouse `memory/project_et_direction_section_8_audit.md`. Real GS ET fixture delivered to `tests/regressions/extratime/gs_et/`; IDSSE/Metrica ET fixtures synthesized in silly-kicks — none exist in bronze.)

**Implementation may begin** on the library change (the round-2 polish D–J is folded into the test/impl plan above). Phase-A lakehouse work proceeds in parallel; the silly-kicks **ship** waits on decision 6.

## 10. Alternatives considered

- (a) Both warn + default — rejected: ships wrong coords with an unseen warning.
- (b) Leave the asymmetry — rejected: latent correctness bug + a trap.
- (c) Auto-derive ET direction from positions — rejected: re-introduces silent-wrong.
- (d) Call-site-only patches (no library change) — rejected as the long-term fix; the loader `_apply_et_direction` is the *interim* unblock only.

## 11. References

- ADR-006 + 3.0.1 erratum (per-period-absolute Sportec/Metrica/GS).
- `_direction.home_attacks_right_per_period` (silent default); `tracking/gradientsports.py:114` + `spadl/gradientsports.py:326` (existing raises); `tracking/sportec.py:132-139`, `spadl/sportec.py`, `spadl/metrica.py` (silent call sites).
- TF-24 loader interim fix `scripts/_loader_pining.py:_apply_et_direction`.
- luxury-lakehouse `src/analytics/action_context/pipeline.py:89,130`.
- User rule: no-silent-degradation (fail-loud over deploy-and-observe).
