# silly-kicks Agent-Support — Phase 3 (MCP `coords` aspect + howto link-check) Design Spec

| Field | Value |
|---|---|
| **Date** | 2026-10-09 |
| **Status** | APPROVED — 2 independent `/review-spec` APPROVE (A + B); 1 shared SHOULD-FIX applied (ADR-032→056); owner-ratified §7 decisions 2026-10-09. Next: writing-plans. |
| **Deciders** | Karsten S. Nielsen |
| **Scope** | Two Phase-2 **Deferred** items (`docs/.../2026-09-30-...-phase2-mcp-design.md` §11). Tooling + one new read-only lib seam. No version bump unless the surface gates demand one; no retrain; C4 container set unchanged. |

Builds on Phase 1 (PR #263, `docs/howto/` runbooks + `.claude/` shims) and Phase 2 (PR #264, `silly_kicks/mcp/` read-only tripwire server). Carries the same workflow: independent spec/plan/impl reviews in SEPARATE sessions (author never reviews own), one approval-gated commit, feature branch (no worktree), reviews to `D:\Development\_reviews\`.

---

## 1. Context

Phase 2 shipped `diagnose_provider` with `aspect ∈ {keeper, convention, id_dtype}`, each binding an existing read-only lib seam. The `coords` aspect was deferred: **no coords-diagnosis seam existed at HEAD** (verified 2026-10-09: the only coord-adjacent validator is `validate_velocity_regime`, unrelated). Separately, the Phase-1 `docs/howto/` runbooks cite ADRs, `docs/context/` docs and code paths by hand, with no gate that those references resolve — doc-rot is silent.

This phase: (1) build the missing read-only **coordinate-integrity** lib seam and bind it as `diagnose_provider`'s `coords` aspect; (2) add a CI-enforced **reference link-check** for the class-2 doc store.

**Charter (inherited, binding):** the MCP server is a pure adapter — each tool composes a fail-loud read-only load/memo seam with a bound compute seam, owns no analysis logic, never writes the tree, never runs corpus-scale. All analysis lives in the library; the server only arg-normalizes + JSON-coerces.

## 2. Non-goals / what this does NOT do

- **Orientation / direction of play** — owned by `check_orientation` (binds `measure_rc4_orientation.measure` + `orient_frames_to_ltr_by_geometry`). `diagnose_coordinates` does NOT re-measure or assert direction; it states so in its docstring and carries it as an explicit note in the diagnosis.
- **Flagging legitimate off-pitch tracking positions.** Per the provider contracts, tracking frames are legitimately off-pitch (SkillCorner `skillcorner.py:46` "tracking is full of legitimately off-pitch positions"; SB `_sb_coordinates.py:63` "y inverted, NOT clipped"), while actions ARE clipped to SPADL bounds (`gradientsports.py:885`). The diagnosis treats a frames out-of-pitch fraction as INFO, never an error; only actions-off-pitch and GROSS out-of-range (a scale bug) raise flags.
- **Asserting a y-convention.** The kloppy y-inversion is a known, contract-handled trap (`_kloppy_coordinates.py:6`); y is reported as INFO, never hard-asserted (no ground truth per match).
- No corpus-scale wrappers, no memo regeneration, no tree writes (charter).

## 3. Item 1 — the `coords` aspect

### 3.1 New lib seam (where the analysis lives)

**`silly_kicks.spadl.diagnose_coordinates(actions, frames, *, params=CoordinateDiagnosisParams()) -> CoordinateDiagnosis`** — a pure, read-only function (pandas in, dataclass out; zero I/O, zero mutation; `copy`-free reads only). Home = `silly_kicks.spadl`: coordinates are the SPADL `[0,105]×[0,68]` frame, and `spadl` already hosts the sibling `detect_input_convention` seam the `convention` aspect binds. Either `actions` or `frames` may be `None` (a provider load without one), but not both (raise).

New PUBLIC module symbols → MUST register in `tests/test_public_api_examples.py` `_PUBLIC_MODULE_FILES` with a runnable doctest on each public symbol (or a recorded `_EXAMPLES_DEBT` deferral with a note). Run the surface gates (`test_public_api_examples`, `c4`, `metric_contracts`) locally before push (the Phase-2 lesson: the surface gate caught an unregistered module AFTER impl review passed).

### 3.2 Dataclasses (frozen)

```
@dataclass(frozen=True)
class CoordinateDiagnosisParams:
    field_length: float = spadlconfig.field_length      # 105.0 m
    field_width: float = spadlconfig.field_width         # 68.0 m
    off_pitch_tol_m: float = 1.0        # recording-noise band outside the pitch rectangle
    gross_range_factor: float = 2.0     # |coord| beyond factor×dimension ⇒ a scale/convention bug, not out-of-play
    # DECISION (OQ2): ship a classmethod for_provider(provider) returning the NEUTRAL default for EVERY
    # provider in v1 (the repo CoverShadowParams.for_provider seam) — promotable later with no API break.

@dataclass(frozen=True)
class CoordinateAxisStats:
    min: float; p01: float; p50: float; p99: float; max: float   # robust percentiles (NaN-skipping)

@dataclass(frozen=True)
class CoordinateTableDiagnosis:
    table: str                       # "actions" | "frames"
    n_rows: int
    x: CoordinateAxisStats
    y: CoordinateAxisStats
    inferred_scale: str              # spadl_meters | normalized_0_1 | scale_0_100 | suspect | undetermined (min-n)
    out_of_pitch_fraction: float     # fraction of rows outside [0,L]×[0,W] beyond off_pitch_tol_m
    gross_out_of_range_fraction: float
    coord_nan_fraction: float        # fraction of rows with ANY NaN in the diagnosed coord columns (INFO)
    all_coords_nan: bool             # every diagnosed coord cell is NaN (the coords_all_nan predicate)

@dataclass(frozen=True)
class CoordinateDiagnosis:
    actions: CoordinateTableDiagnosis | None
    frames: CoordinateTableDiagnosis | None
    flags: list[str]                 # the authoritative flag set (see 3.4)
    notes: list[str]                 # INFO/non-claims (orientation-not-measured; frames-off-pitch-legit; y-convention INFO)
```

Diagnosed coord columns: **actions** → `start_x, start_y, end_x, end_y`; **frames** → `x, y` (NOT `z`/`speed`; `z` is NA-on-ball-row by construction, ADR-106). Frame coords are float32 STORAGE (ADR-106) → upcast the slice to float64 at the read boundary (the repo-wide kernel rule) before stats.

### 3.3 The checks (each grounded)

1. **Scale classification** (per table) — DECISION (OQ1): classify on two pitch-geometry signals, never eyeballed magnitudes. **(a) Aspect ratio** of the robust x/y spans — SPADL metres has x-span ≈105 vs y-span ≈68 (**ratio ≈1.54**), any normalized scale (0–1 or 0–100) is **square (ratio ≈1.0)** — so the ratio alone separates `spadl_meters` from normalized. **(b) Magnitude decade** (p99) then splits normalized into `normalized_0_1` vs `scale_0_100`. Robust p01/p99 (NaN-skipping), never min/max (off-pitch tails); below a min-n return `undetermined` (never guess); neither → `suspect`. The EXACT bands + a synthetic fixtures table are **pre-registered in the plan before coding** (construct discipline, not tuned to a provider). A units heuristic/tripwire, not a proof of units.
2. **Out-of-pitch fraction** — rows outside `[0,field_length]×[0,field_width]` beyond `off_pitch_tol_m`. For **actions** a nonzero value is a defect (clipped by contract) → flag. For **frames** it is INFO (legitimate off-pitch).
3. **Gross out-of-range fraction** — `|x| > gross_range_factor×field_length` or `|y| > gross_range_factor×field_width`. On EITHER table this signals a scale/convention bug (an out-of-play player is metres off-pitch, not 2×105 m) → flag.
4. **NaN / coverage** — `coord_nan_fraction` = rows with ANY NaN coord (INFO). `coords_all_nan` fires on **true all-coords-NaN** (`all_coords_nan` — every diagnosed coord cell NaN, NOT any-per-row ≥1.0, which would over-claim on start-only-NaN actions). `actions_start_nan` fires on **start coords only** (`start_x`/`start_y` NaN fraction > 0) — never on end-only NaN.

### 3.4 Flags (authoritative, on the diagnosis)

`coords_scale_suspect` (any table `inferred_scale` ∉ {`spadl_meters`, `undetermined`} — i.e. ANY non-SPADL scale incl `normalized_0_1`/`scale_0_100`/`suspect`; the §3.3 primary bug-class tripwire), `actions_out_of_pitch` (actions `out_of_pitch_fraction > 0` beyond tol), `coords_gross_out_of_range` (any table `gross_out_of_range_fraction > 0`), `coords_all_nan` (any table `all_coords_nan`), `actions_start_nan` (actions start-coords NaN fraction > 0). Empty list ⇒ clean. Order is deterministic (defined list, filtered).

### 3.5 MCP binding (adapter only)

`_diagnose(loaded, aspect)` gains `if aspect == "coords": return diagnose_coordinates(getattr(loaded,"actions",None), getattr(loaded,"frames",None))`. `_flags(aspect, diag)` for `coords` returns `list(diag.flags)` (the lib is authoritative; the server adds nothing). `diagnose_provider`'s docstring aspect list updates to `{keeper, convention, id_dtype, coords}`. `_json_safe` already handles frozen dataclasses + nested lists → no serialization change. The `coords` entry drops out of the Phase-2 `_diagnose` `ValueError` known-list.

## 4. Item 2 — the howto/context link-check

**`tests/test_howto_links_wired.py`** — a structural guard in the `test_*_wired` idiom (runs under `pytest tests/`; NO `ci.yml` edit needed). Read-only; parses markdown, resolves references against the repo tree.

**Breadth:** `docs/howto/*.md` AND `docs/context/*.md` (both class-2; gold-standard, scope-not-a-concern). The always-loaded `AGENTS.md` is out (its pointers are budget-gated by `test_agents_md_budget.py` already; revisit only if review asks).

**Reference kinds resolved:**
1. **Relative markdown links** `](<path>)` (and `](<path>#anchor)`) → the target file exists (relative to the md file's dir, then repo root) AND, when an `#anchor` is present, the slugified heading exists in the target md (both HARD; DECISION OQ3).
2. **ADR mentions** `ADR-NNN` → exactly one `docs/superpowers/adrs/ADR-NNN-*.md` exists (HARD).
3. **Code-path refs** matching `(silly_kicks|tests|scripts)/[\w/]+\.py` → the file exists (HARD). A `:NNN` line-suffix is a SOFT WARNING — reported, non-failing: line pins shift on every edit, so hard-failing them would train authors to drop citations (DECISION OQ3).
4. **Doc-path prose mentions** `docs/[\w./-]+\.md` → the target file exists (HARD). This is the DOMINANT reference kind in these docs (bare paths written in prose, not `](...)` links — measured 39 doc-paths vs 0 md-links @50fab4c).

**Semantics:** collect every reference across the in-scope files; fail loud listing each unresolved `(file, ref, kind)`. **Non-vacuity (ADR-056 idiom):** assert floors MEASURED from the live tree (ADR-refs 465, code-paths 120, doc-paths 39, md-links 0 @50fab4c), pinned ~50% below — `adr≥200, code≥60, docp≥20`; no md-link floor (none present; the resolver is kept for future) — so a parser that silently matches nothing fails rather than passing green. External `http(s)://` links are out of scope (no network in CI; the Phase-1 "optional CI link-check" note scoped internal refs). **Ceiling:** a wrong-but-existing reference (a mis-numbered `ADR-NNN` whose file happens to exist — the ADR-032-vs-056 case this cycle fixes) is NOT catchable by a link-check; semantic-cite correctness is out of scope.

## 5. Testing (TDD — tests first, each from both sides)

**`diagnose_coordinates`** (`tests/spadl/test_diagnose_coordinates.py`; DECISION OQ4), synthetic frames/actions:
- SPADL-valid in-bounds → `inferred_scale == "spadl_meters"`, empty flags.
- Normalized 0–1 coords → `normalized_0_1`, `coords_scale_suspect`.
- 0–100 coords → `scale_0_100`, `coords_scale_suspect`.
- Actions with a row outside the pitch beyond tol → `actions_out_of_pitch`; the SAME out-of-pitch rows in FRAMES → NO flag (legit off-pitch), `out_of_pitch_fraction` reported.
- Gross (10×) coords → `coords_gross_out_of_range` on both tables.
- All-NaN coords → `coords_all_nan`; actions start-NaN → `actions_start_nan`.
- `actions=None`+`frames=None` → raises; one-None → the present table diagnosed, the other `None`.
- Purity: input frames/actions unmodified (compare pre/post); float32 frames tolerated (upcast at boundary).
- Determinism: same input → identical dataclass.

**MCP `coords` aspect** (`tests/mcp/`): `diagnose_provider(..., aspect="coords")` returns the JSON-safe dict with `findings` + `flags`; unknown aspect still raises; `_json_safe` round-trips the nested `CoordinateDiagnosis`.

**Link-check** (`tests/test_howto_links_wired.py` has its OWN precondition test — the ADR-056 non-vacuity discipline, a guard is only as good as its fixture): a fixture md string with a KNOWN-broken ADR ref / md link / code path → the resolver reports exactly those; a fixture with all-valid refs → clean; the non-vacuity floor asserts the real doc tree yields ≥ the floor counts.

**Surface/registry:** `test_public_api_examples` (new `spadl` public symbols registered + doctested-or-debted), `c4` (container set unchanged — `diagnose_coordinates` is a lib fn, not a new container; confirm the C4 test stays green), `metric_contracts` (N/A — not a metric family; confirm no accidental registration needed).

## 6. Workflow & gates

Feature branch (one; no worktree). Order: write seam tests → `diagnose_coordinates` → MCP binding + test → link-check guard → run the full `not e2e` suite + the surface gates locally. Nothing deferred/cut without explicit owner approval. Independent `/review-spec` (this doc) and `/review-plan` BEFORE coding; independent `/review-impl` BEFORE the commit; all reports to `D:\Development\_reviews\` (`2026-10-09-agent-support-phase3-{spec,plan,impl}.md`). One fully-tested, approval-gated commit; PR + main CI green before any merge (merge is a separate owner go). No `silly_kicks` runtime behavior change beyond the additive new function; docs: `docs/howto/mcp.md` (the new aspect), the `AGENTS.md` MCP/howto pointers if a rule changes, CHANGELOG.

## 7. DECISIONS (ratified by owner 2026-10-09, gold-standard; both reviewers endorsed)

1. **Scale classification** — the two-signal method (aspect ratio ≈1.54 vs ≈1.0, then magnitude decade; robust p01/p99; `undetermined` below min-n) per §3.3. The EXACT bands + synthetic fixtures table are pre-registered in the PLAN before any code.
2. **`for_provider`** — ship the `for_provider(provider)` classmethod seam (repo convention); v1 returns the NEUTRAL default for every provider; promotable later with no API break.
3. **Link resolution** — md file links + `#anchor` (slugified heading) + `ADR-NNN` file + code-path file are HARD; `:NNN` line-suffix is a SOFT WARNING (reported, non-failing). Semantic-cite correctness is out of scope (the ceiling in §4).
4. **Test placement** — `tests/spadl/test_diagnose_coordinates.py`.
5. **Flag tokens (FROZEN; Hyrum)** — `coords_scale_suspect`, `actions_out_of_pitch`, `coords_gross_out_of_range`, `coords_all_nan`, `actions_start_nan` (§3.4).
6. **In-cycle doc-rot fix** — `AGENTS.md:123` + `docs/context/conventions-core.md:58` mis-label the precondition-test idiom "ADR-032" (ADR-032 = pitch-control-at-target); corrected to **ADR-056** (byte-neutral → `test_agents_md_budget` safe). Lands in this cycle's single final commit. A link-check does NOT catch this class (file exists); surfaced by the reviewers, owner-approved to fix here.

## 8. Alternatives considered

| Option | Why not |
|---|---|
| Put coords logic in `server.py` | Violates the charter (server owns no analysis). The compute must be a lib seam. |
| A new `silly_kicks.diagnostics` module co-locating keeper/convention/id_dtype/coords | No precedent; the existing aspect-seams are deliberately scattered in their domain packages (`spadl`/`tracking`). Lower surface to add one function to `spadl`. Revisit only if a 2nd coords-like seam appears. |
| `coords` = re-use `check_orientation` / `detect_input_convention` | Those measure direction/attack-convention, not scale/range/NaN integrity — the orthogonal gap this fills. |
| Link-check as a standalone script (not a pytest guard) | A `test_*_wired` guard is CI-enforced automatically (`pytest tests/`) and matches the repo's structural-guard idiom; a script needs a `ci.yml` wire + a human to run it. |
| Validate external `http(s)` links too | No network in CI; flaky; out of the internal-refs scope. |

## 9. Attribution

Internal (agent-support follow-up). Builds on the Phase-2 MCP spec and the `test_*_wired` structural-guard idiom; the coordinate contracts cited are `spadl/config.py` (ADR-050 geometry constants), ADR-106 (float32 frame storage), ADR-028 action-LTR reprojection / ADR-029 frame-LTR (orientation-adjacent; out of scope here), and the provider coord-contract comments in `spadl/{gradientsports,skillcorner,kloppy,_kloppy_coordinates,_sb_coordinates}.py`.
