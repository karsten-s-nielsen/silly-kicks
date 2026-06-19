# TF-23b — Geometric frame-LTR backstop on the native tracking adapters

| Field | Value |
|---|---|
| **Date** | 2026-06-18 (rev-5 2026-06-19) |
| **Status** | Approved; plan written + cross-session-reviewed. rev-5 adds the net `copy: bool=True` knob (plan-review #1, §3.4) so spec↔plan don't drift. |
| **Feature** | TF-23b |
| **Target version** | silly-kicks 4.34.0 |
| **ADR** | ADR-035 (new, concise) — references ADR-034 (parent), ADR-031 (Gate D), ADR-029 (supersedes its no-refactor fence), ADR-019, ADR-010 |
| **Branch** | `pr-s99-tf23b-geometric-backstop` |

## 1. Problem

The native tracking adapters `silly_kicks.tracking.gradientsports.convert_to_frames`
and `silly_kicks.tracking.sportec.convert_to_frames` orient each period from a
**caller-supplied flag** via `direction.home_attacks_right_per_period(home_team_start_left,
home_team_start_left_extratime)`. The per-period flip is only as correct as those flags.

The live GS-ET incident (2026-06-13) was a consumer passing a **wrong**
`home_team_start_left_extratime` placeholder → periods 3/4 reversed → every
geometry-derived ET tracking feature mis-oriented. The current regression test
`tests/tracking/test_adapter_extra_time_orientation.py::test_wrong_extra_time_flag_reverses_p3_p4`
*documents* this vulnerability (asserts the wrong flag reverses ET).

ADR-034 (TF-23, 4.33.0) shipped the flag-free geometric orienter
`tracking.orient_frames_to_ltr_by_geometry`, which reads orientation from the **data**
(per-period home-GK median x → point-reflect any period whose home GK sits on the
attacking half), is **idempotent**, and is provider-agnostic. ADR-034 explicitly parked
"the geometric net on the GS native adapter" as **TF-23b**, naming it the gate that lets
the luxury-lakehouse retire its own `correct_frames_to_home_ltr` backstop entirely:

> "The interim net stays only as an idempotent GS-ET backstop until **TF-23b** (the
> geometric net on the GS native adapter) lets it be deleted entirely. The
> provider-conditional net must not ossify — TF-23b is the gate."

Separately, ADR-031 lists **"IDSSE-ET handedness (Gate D)"** as an open item. The sportec
adapter shares the identical flag-flip pattern and the same ET vulnerability, so the same
backstop closes Gate D.

## 2. Goal

Wire `orient_frames_to_ltr_by_geometry` into **both** native tracking adapters
(`gradientsports` + `sportec`) as an always-on **idempotent backstop**, so a wrong/absent
ET flag self-corrects from GK geometry. Scope decided with the owner: **GS + sportec**
(closes ADR-031 Gate D), **backstop** (keep the flag-flip; the net only corrects), not a
replacement of the flag path.

This revision applies an **owner gold-standard / best-practice directive** (2026-06-19):
prefer the architecturally-correct shape even where it means a deliberate breaking change.
Three things follow from that directive (each detailed below):

1. **Single source of truth for the orientation tail** — extract the per-period flag flip +
   attacking-direction label + geometric backstop into one shared
   `direction.finalize_orientation(...)` that both adapters call (§3.3). This **supersedes the
   ADR-029 "adapters are NOT refactored through the helper" fence** — that fence's stated
   reason was "zero gain (primitives already shared)", which is now obsolete: this PR would
   otherwise add a *third* inline copy of the tail, and the 2026-06-09 dtype-safe `is_home` fix
   already had to be applied twice in lockstep. The supersession is recorded in ADR-035.
2. **Policy injected, not re-implemented** — give the net an
   `on_missing_home: Literal["raise","warn"]="raise"` parameter (§3.4); adapters pass
   `"warn"` to preserve their warn-don't-raise contract instead of re-implementing the net's
   zero-home guard. Default `"raise"` keeps every existing direct caller byte-identical
   (ADR-019 preserved).
3. **The net never orients PSO** — restrict the net's flip loop to `_LTR_KNOWN_PERIODS`
   (§3.5). Orienting a period-5 (penalty-shootout) frame by GK geometry is *meaningless*
   (both teams attack one end), so the correct behavior is to never attempt it — for **any**
   caller, not to preserve the garbage flip behind a default. This is a deliberate public-net
   behavior change (Hyrum), reflected in the CHANGELOG.

Non-goals: events converters (the SPADL `spadl/{gradientsports,sportec,metrica}.py` flip
math is untouched — the net is tracking-only); the kloppy gateway (already oriented — it does
not call `finalize_orientation`); a **package-exported** public API surface
(`finalize_orientation` is module-public-not-exported, matching the existing
`compute_attacking_direction` precedent; `on_missing_home` is an additive default-preserving
parameter, not a new entry point).

## 3. Design

### 3.1 Shape overview

Two collaborating changes in `silly_kicks/tracking/direction.py`, plus a collapse of both
adapters onto the shared helper:

- **`orient_frames_to_ltr_by_geometry`** (the net) gains `on_missing_home` (§3.4) and skips
  period 5 in its flip loop (§3.5).
- **`direction.finalize_orientation(...)`** (new, §3.3) owns the full orientation tail
  (ET guard → per-period flag flip → period-gated attacking-direction label → geometric
  backstop). One insertion point for the backstop, one definition of the dtype-safe
  `is_home` mask, computed once.
- **`sportec.convert_to_frames` / `gradientsports.convert_to_frames`** keep only their
  provider-specific coordinate construction (`x_centered + 52.5`, `y_centered + 34.0`) and
  call `finalize_orientation(...)`.

### 3.2 Why this is correct and safe

The whole "no-op / byte-identical / no-retrain on the correct-flag path" claim rests on two
adapter facts (a)+(b), verified against 4.33.0 source — they are load-bearing, so the §5.5
DGX no-op proof is a hard ship gate on them:

- **(a) The flag-flip's "correct" target convention IS the net's (home attacks +x).** Each
  adapter flips exactly the periods where the home team attacks left, so home always ends
  attacking +x: GS `gradientsports.py:134-137` (`home_rtl_periods = {p ... if not
  attacks_right}; out.loc[flip_mask, "x"] = 105.0 - x; "y"] = 68.0 - y`), sportec
  `sportec.py:143-146` (identical). The net flips iff home-GK **median** x > 52.5
  (`direction.py:254`). Same convention, same point reflection (`x→105−x, y→68−y`). A
  correctly-flag-flipped period has home-GK median x < 52.5 → net no-op (one reflection vs
  two = identity); a wrongly-flipped period (median > 52.5) → one more reflection → corrected.
- **(b) `team_attacking_direction` is set period-INDEPENDENT, not flag-derived per-period.**
  Both adapters set `home & isin([1,2,3,4]) → "ltr"`, `away & isin([1,2,3,4]) → "rtl"` (GS
  `:151-153`, sportec `:162-164`) — the label is the post-orientation invariant "home attacks
  +x", gated only on *known period*, NOT on the flag. The net leaves the already-populated
  label untouched (its `isna().all()` guard, `direction.py:274`). So after a geometric
  correction the label's claim becomes *true* for the corrected period — no label/coord
  inconsistency. (Were the label set per-period from the flag, a wrong flag would leave a wrong
  label the net does not fix; it is not, so this hazard does not exist.)
- **Runs for both conventions.** The net produces the home-attacks-right (`absolute_frame`)
  representation; `play_left_to_right` (for `ltr`) is applied on top afterward. So the backstop
  runs unconditionally inside `finalize_orientation`, before the convention branch — both
  outputs are corrected.
- **Velocities.** The net flips `vx`/`vy` when present; at this point preprocessing has not run
  and they are absent from the schema — nothing to flip; the later `derive_velocities` operates
  on the corrected coordinates.
- **Report is unaffected by the moved insertion point.** The backstop now runs on `out`
  **before** `final` is built + dtype-coerced (vs the v1 spec's after). Safe: the net adds/drops
  no rows and changes no NaN-ness, and `final` is rebuilt from `out`'s columns, so
  `nan_rate_per_column`, coverage, and `ball_out_seconds` are byte-identical regardless of
  ordering. Bonus: the pre-coercion/post-coercion `is_home` drift surface flagged in review
  concern 2 disappears — one mask, computed once in the helper.
- **Policy at the edge via injection (not a re-implemented guard).** The net **raises** on a
  zero-home-match by default (ADR-019: mis-orienting is worse than failing). The adapters
  currently **warn**. Rather than guard the call with a copy of the net's raise condition,
  `finalize_orientation` passes `on_missing_home="warn"` so the net itself emits the warning and
  returns the frame un-oriented (the flag-flip result stands). Single definition of "can't
  anchor", at one seam.

### 3.3 `finalize_orientation` — the shared tail (concern 1)

New module-public function in `direction.py` (no leading underscore, **not** added to
`tracking/__init__.__all__` — matching `compute_attacking_direction`, which is module-public,
unexported, and carries no `Examples` block yet passes the package-exported-only Examples gate;
a solid docstring with an `Examples` block is still written for quality / National Park):

```python
def finalize_orientation(
    out: pd.DataFrame,
    *,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None,
    source: str,
    game_id: Any = None,
    on_missing_home: Literal["raise", "warn"] = "warn",
) -> pd.DataFrame:
    """Shared orientation tail for the sportec + gradientsports native tracking adapters.

    Single source of truth for the ET guard, the per-period flag flip, the post-flip
    period-gated ``team_attacking_direction`` label, and the TF-23b geometric backstop.
    Expects ``out`` to already carry canonical ``x``/``y`` (105x68 m) plus ``team_id``,
    ``period_id``, ``is_ball``, ``is_goalkeeper``, ``game_id``. **Returns a NEW frame and does
    not mutate the input** (copy-at-entry — clean value semantics for a reusable helper). The
    output is in home-attacks-right (absolute) convention; the caller applies
    ``play_left_to_right`` afterward for ``output_convention="ltr"``.

    The geometric backstop (``orient_frames_to_ltr_by_geometry``) self-corrects any period
    whose home GK sits on the attacking half — e.g. a wrong ``home_team_start_left_extratime``
    placeholder. It is idempotent, so on a correct-flag match it is a byte-identical no-op.
    ``on_missing_home="warn"`` (the adapter default) preserves the adapters' warn-don't-raise
    contract without re-implementing the net's zero-home condition.
    """
    out = out.copy()  # P2: clean value semantics — never mutate the caller's frame
    require_et_direction(out["period_id"], home_team_start_left_extratime, source=source)

    flips = home_attacks_right_per_period(home_team_start_left, home_team_start_left_extratime)
    home_rtl_periods = {p for p, attacks_right in flips.items() if not attacks_right}
    flip_mask = out["period_id"].isin(home_rtl_periods).to_numpy()
    out.loc[flip_mask, "x"] = _PITCH_LENGTH_M - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = _PITCH_WIDTH_M - out.loc[flip_mask, "y"]

    out["team_attacking_direction"] = None
    is_player = (~out["is_ball"].astype(bool)).to_numpy(dtype=bool)
    is_home = ids_match(out["team_id"], home_team_id).fillna(False).to_numpy(dtype=bool)
    is_known = out["period_id"].isin(_LTR_KNOWN_PERIODS).to_numpy(dtype=bool)
    out.loc[is_player & is_home & is_known, "team_attacking_direction"] = "ltr"
    out.loc[is_player & ~is_home & is_known, "team_attacking_direction"] = "rtl"

    if is_player.any():  # all-ball frame: nothing to anchor; skip the net entirely
        out = orient_frames_to_ltr_by_geometry(
            out,
            home_team_id=home_team_id,
            source=source,
            game_id=game_id,
            on_missing_home=on_missing_home,
        )
    return out
```

Behavior-preserving micro-cleanups folded in (called out in the PR so review isn't surprised):
- `[1, 2, 3, 4]` literal → `_LTR_KNOWN_PERIODS` (same tuple, now single-sourced with the net).
- The zero-home warn now comes from the net (via `on_missing_home="warn"`) instead of the
  adapter's bespoke string — **the warning text changes** (see §4 + test note §5.6). No existing
  test asserts the old string (a grep of `tests/` finds only a code comment), so this breaks no
  current test.

Each adapter collapses (~30-line block at sportec `:134-164` / GS `:122-153`) to:

```python
out = raw_frames.copy()
out["x"] = out["x_centered"] + 52.5
out["y"] = out["y_centered"] + 34.0

out = direction.finalize_orientation(
    out,
    home_team_id=home_team_id,
    home_team_start_left=home_team_start_left,
    home_team_start_left_extratime=home_team_start_left_extratime,
    source=f"{_PROVIDER_NAME} convert_to_frames",
    game_id=(out["game_id"].iloc[0] if len(out) else None),
    on_missing_home="warn",
)
# unchanged below: speed / speed_source / confidence / visibility / source_provider /
# is_goalkeeper_source; build `final`, dtype-coerce, report, output_convention branch, preprocess
```

GS is identical except the provider name. `game_id` is confirmed present in **both** adapters'
`EXPECTED_INPUT_COLUMNS` (gs `:37`, sportec `:39`), so `out["game_id"].iloc[0]` is valid. The
next orientation change (the kind that needed the 2026-06-09 dtype-safe `is_home` fix applied
twice) now touches **one** function.

### 3.4 `on_missing_home` — injected zero-home policy (concern 2)

`orient_frames_to_ltr_by_geometry` gains `on_missing_home: Literal["raise","warn"] =
"raise"`. (P1: `"skip"` was dropped — no caller needs silent un-orientation; speculative
generality is debt per `feedback_speculative_api_surface_is_debt` + CLAUDE.md "no dead code".)
**Only** the zero-home branch changes; the **raise condition itself stays
byte-identical** (`if not bool((is_player & is_home).any())`) — I deliberately do **not** add
`is_player.any() and` to the net's condition (the reviewer's variant), so the `"raise"` default
is *truly* byte-identical for direct callers, including on an all-ball frame (still raises, as
today). The adapter path never reaches the net with an all-ball frame anyway — the helper gates
the call with `if is_player.any():` — so the unification the reviewer proposed buys nothing while
changing a degenerate default. Gold standard ≠ gratuitous edge-behavior change.

```python
from typing import Any, Literal  # add Literal

# inside orient_frames_to_ltr_by_geometry, replacing the bare raise:
if not bool((is_player & is_home).any()):
    msg = (
        f"orient_frames_to_ltr_by_geometry: home_team_id={home_team_id!r} matched ZERO "
        f"player rows ({source} game={game_id})"
    )
    if on_missing_home == "raise":
        raise ValueError(msg + " --- refusing to guess orientation.")
    if on_missing_home == "warn":
        warnings.warn(msg + " --- orientation left as-is.", stacklevel=2)
        return out  # return the (flag-flipped) copy untouched, before the flip loop
```

The net also gains a `copy: bool = True` knob (plan review #1): default `True` keeps the input-never-
mutated contract for direct/lakehouse callers (byte-identical), while `finalize_orientation` — which
already owns a fresh copy-at-entry — passes `copy=False` so a `convert_to_frames` call does **two**
full-frame copies, not three (adapter copy → finalize copy → net copy), on tracking-scale data
(~1M+ rows/match). Additive, default-preserving, same flavor as `on_missing_home`.

`"warn"` returns **before** the flip loop, so no coordinates change → exactly "the flag-flip
result stands" (§3.2). Default `"raise"` ⇒ every existing direct caller (the lakehouse TF-23
builders) is byte-identical — no migration; ADR-019 pure-raise preserved.

### 3.5 Period 5 (PSO) — net never orients shootouts (concern 3)

`orient_frames_to_ltr_by_geometry` today iterates **every** period present and flips on home-GK
median x (`direction.py:250`); only the `team_attacking_direction` label fill is restricted to
`_LTR_KNOWN_PERIODS=(1,2,3,4)` (`:274-277`). In a penalty shootout both teams attack one end, so
the home-GK median x is an anomalous orientation anchor → the net can spuriously flip period 5.
The **flag path never flips period 5** (`home_attacks_right_per_period[5] = True`,
`direction.py:122`), so a correct-flag PSO match would otherwise become a non-no-op purely from
meaningless shootout geometry. GS WC2022 contains shootouts (the final, etc.).

**Fix (one line):** restrict the net's flip loop to `_LTR_KNOWN_PERIODS`
(`direction.py:250` → iterate `_LTR_KNOWN_PERIODS` ∩ periods-present instead of all
`.unique()` periods), making the flip scope equal the already-restricted label scope. PSO
orientation is undefined and left as the input left it.

**This is a deliberate public-net behavior change (Hyrum), not "byte-identical except adapters":**
`orient_frames_to_ltr_by_geometry` is public and is also called by the TF-23
`tracking.skillcorner` / `tracking.metrica` builders (the authority path, where the net is the
*only* orientation step). After this change, any period-5 frames in **any** caller's data — GS,
sportec, SkillCorner, Metrica — are no longer flipped by the net. Handling:

- **Correct behavior, gold standard.** Orienting PSO by GK geometry is meaningless; the net
  should not attempt it for anyone. We do **not** preserve the garbage flip behind a default
  (the conservative-but-not-gold parametrize alternative was explicitly rejected under the owner
  directive).
- **Practical impact nil.** PSO frames are excluded from all geometric feature analysis. State
  this in the ADR/CHANGELOG.
- **Scope we can enumerate.** silly-kicks can enumerate period-5 presence in the **pining
  corpus** (GS×64 + IDSSE×7) via the DGX G1 run (§5.5) — that list is recorded in ADR-035. We
  **cannot** verify lakehouse-production SkillCorner/Metrica PSO presence; the CHANGELOG +
  ADR-035 flag the cross-repo retrain so the lakehouse assesses its own SC/Metrica PSO frames
  when it adopts ≥4.34.0.
- **Byte-identical for the TF-23 goldens** (no period 5 in that data), so it does not disturb the
  committed TF-23 fixtures.

Guard: a PSO regression test in the net's own `tests/tracking/test_orient_by_geometry.py`
(a period-5 frame with an attacking-end home GK is left un-flipped); existing
idempotency/orientation cases stay green.

### 3.6 Backstop, not replacement (owner decision, recorded)

Keeping the flag-flip and layering the net on top is strictly safer than orienting purely
geometrically:

- A period with **no GK in frame** falls back to the flag-flip result (correct if the flag is
  correct) instead of being left as raw per-period-absolute.
- `home_team_start_left` / `home_team_start_left_extratime` and the `require_et_direction` guard
  stay meaningful (no vestigial parameters).
- The change is purely additive on the correct-flag path (byte-identical), minimizing blast
  radius.

### 3.7 GK anchor reliability + anchor-context asymmetry

Both adapters carry **native** `is_goalkeeper` (`is_goalkeeper_source = "native"`): GS from the
roster join (`add_gradientsports_player_ids`, `position_group_type == "GK"`), sportec from the
DFL roster. The net anchors on the home-GK **median** x over the whole period — a median sits
firmly in the GK's own half even for a sweeper-keeper, so false-flips on correctly-oriented real
data are not a concern. Same anchor TF-23 validated for the metrica/skillcorner builders.

**Anchor-context asymmetry (note for ADR-035).** The native adapters convert the **full match**,
so the median anchor runs over a whole period (~tens of thousands of frames) and the
sweeper-keeper/corner transient cannot move the median out of the GK's half — no per-batch guard
is needed here. The opposite of the luxury-lakehouse TF-23 path, which feeds the builders
~250-frame (~10 s) windows where a transient *can* flip the deepest player, and therefore needs a
cross-batch GK/orientation-consistency guard. Same anchor, different consumption pattern → guard
downstream, not here. One line in ADR-035 so the asymmetry is on record.

## 4. Blast radius (verified against the actual fixtures)

| Test | Effect | Action |
|---|---|---|
| `golden_{gs,sportec}_tracking_rt.parquet` via `test_rt_no_regress.py` | Correct-flag period-1 fixtures → home-GK median < 52.5 → **net no-op**; `finalize_orientation` is a pure refactor of the same flip+label ops → **byte-identical** | None expected; confirm empirically (plan task 1) |
| `test_adapter_extra_time_orientation.py::test_positive_extra_time_orientation` | Correct flag → net no-op → home GK x=5 all periods | Unchanged (passes) |
| `test_adapter_extra_time_orientation.py::test_wrong_extra_time_flag_reverses_p3_p4` | Wrong flag now **self-corrects** (net flips ET back) → home GK x=5, not 100 | **Invert** → `test_wrong_extra_time_flag_self_corrects_via_geometry` |
| `test_et_guard_parity.py::test_all_converters_et_orientation_reflects_with_flag` | **Tracking** cases no longer reflect under flag negation (both flags self-correct); **events** unchanged | **Split**: tracking self-corrects; events reflect |
| `test_et_guard_parity.py::test_all_converters_raise_same_message_shape_on_et_without_flag` | ET guard fires inside `finalize_orientation` before the net; unchanged | None |
| `test_real_et_roundtrip.py` (sportec_tracking flag=True; events) | flag=True → net no-op → bounds OK; events untouched | None |
| Zero-home warn **text** | Now the net's message (`"...matched ZERO player rows (<source> game=<id>) --- orientation left as-is."`) instead of the adapter's `"(id dtype vs frame team_id mismatch?)"` | No existing test asserts it; new §5.4 test asserts substring `"matched ZERO player"` |

The net is **tracking-only**: all SPADL events converters (`spadl/*.py`) are untouched, so their
ET reflection behavior is preserved — this is why the parity test splits along the tracking/events
axis.

## 5. Test plan (TDD — red first)

### 5.1 Behavioral (synthetic, committed, all CI legs)

`tests/tracking/test_adapter_extra_time_orientation.py`:
- **Invert** the wrong-flag test → `test_wrong_extra_time_flag_self_corrects_via_geometry`
  (both adapters, parametrized): a wrong ET flag → home GK at x≈5 / away GK at x≈100 in **all
  four periods**. The synthetic fixture already places the home GK at x_centered = ±47.5 (deep).
- Keep `test_positive_extra_time_orientation` (correct flag → x≈5 everywhere): proves the no-op.
- Add an `output_convention="absolute_frame"` variant of the self-correction test (net runs
  before the convention branch — both conventions corrected; home GK at x≈5 all periods).

### 5.2 Net unit coverage of `on_missing_home` (concern 2)

`tests/tracking/test_orient_by_geometry.py`:
- `"raise"` (default) still raises on zero-home-match (byte-identical to today, incl. all-ball).
- `"warn"` emits `UserWarning` + returns the frame **un-oriented** (no coord change).
- **Net PSO guard** (§3.5): a period-5 frame with an attacking-end home GK is left **un-flipped**;
  the periods-1–4 idempotency/orientation cases stay green (byte-identical).

### 5.2b Direct unit test of `finalize_orientation` (concern P6)

`tests/tracking/test_finalize_orientation.py` (new): a small direct unit test of the shared
helper — correct-flag flip + period-gated label + net-no-op on a deep-GK frame, and the
wrong-flag self-correction — so a refactor regression localizes to the helper instead of
surfacing only as an adapter golden diff. Also asserts copy-at-entry (input frame unchanged
after the call). Cheap; gold-standard / National Park.

### 5.3 Cross-provider parity (synthetic, committed)

`tests/regressions/extratime/test_et_guard_parity.py`:
- Split `test_all_converters_et_orientation_reflects_with_flag`:
  - **events** (`sportec_events`, `gs_events`, `metrica_events`): `xl + xr == 105` (reflect) —
    unchanged contract.
  - **tracking** (`sportec_tracking`, `gs_tracking`): flag=True vs flag=False produce the **same**
    orientation (self-correct), i.e. `np.allclose(xl, xr)` on the finite ET coords.
- The raise-parity test is unchanged (guard fires pre-net, inside `finalize_orientation`).

### 5.4 Contract preservation

- Zero-home-match still **warns, does not raise** (via `on_missing_home="warn"`): a small fixture
  with a `home_team_id` matching no player asserts a `UserWarning` and a returned frame (no
  exception). Assert on the stable substring `"matched ZERO player"` (the full string changed,
  §4).

### 5.5 Native-GK self-correction in CI (concern 4) — regenerate `gs_et` from pining, geometric ground truth

**Fixture facts (verified against the committed repo + the lakehouse provenance clarification
2026-06-19).**
- The committed `tests/regressions/extratime/gs_et/frames.parquet` carries columns
  `match_id, period, frame_num, period_elapsed_time, team_side, is_ball, jersey_num, x, y, z` —
  **no `is_goalkeeper`, no roster, no `player_id`** (the `gs_et/` section of
  `tests/regressions/extratime/README.md`: "none delivered"; `test_real_et_roundtrip.py`
  *synthesizes* a roster and arbitrarily flags the first jersey per team as GK). So the current
  committed test does **not** exercise native GK.
- `match_id=10517`, `home_team_id=364`. The README's "A-League" competition label is a
  **maintainer typo** — 10517 is a **Gradient Sports WC2022 knockout ET match** (conclusive: GS =
  PFF FC WC2022 across ~10 test files; A-League tracking in this ecosystem is SkillCorner, not GS;
  `test_calibrate_cli.py:65` parses `gradientsports:10517`; the README's own audit table = the GS
  WC2022 ET set). Provider + match_id are correct; only the competition word is wrong. **Since we
  regenerate this file, correct the label in-place** to "Gradient Sports WC2022 knockout ET match"
  with a one-line note — no longer an open flag.

**What concern 4 needs, and where it's already met.** The net consumes `is_goalkeeper` identically
regardless of how the column was labeled (it medians `is_home & is_goalkeeper` home players). The
**synthetic ET fixture in §5.1 already sets `is_goalkeeper=True` on the deep GKs**
(`test_adapter_extra_time_orientation.py:33,50-51`), committed, **all CI legs** — so the net's
native-GK *consumption code path* (the new TF-23b surface) is **already a permanent CI gate**. The
residual gap is **real-data** native GK (messy sweeper-keeper positions), added below + validated
at scale by §5.7 G1/G2/G3.

**Regenerate `gs_et` from pining with native GK (the owner "regenerate now" deliverable).**
Licensing is cleared (owner 2026-06-19): committing GS-*derived* test data is fine; only whole
matches/datasets to HF are off-limits — a one-period ET slice is fine. Since 10517 is WC2022 it is
in the **GS×64 pining corpus** (one of the 3 ET-tracking matches), so regenerate the *same*
documented fixture (match 10517, P3) **on the DGX** via the pining roster join
(`_loader_pining.py::_build_gradientsports` → `add_gradientsports_player_ids`), which produces
native `is_goalkeeper` (`is_goalkeeper_source="native"`). Commit the regenerated slice (coords +
team membership + native `is_goalkeeper`, **no whole dataset, no extra restricted fields**) + the
regen script, and drop the synthesized-roster path from the test. (Plan task confirms 10517 ∈
pining at build start; if absent, fall back to another of the 3 GS ET-tracking matches.)

**Ground truth = geometry, NOT the placeholder flag.** Do **not** anchor the test on
`meta.home_team_start_left_extratime=True`: that bronze field is the **constant "true" placeholder
for all GS ET matches** — the exact GS-ET unreliability TF-23b exists to fix
(`reference_gs_et_flag_placeholder_unreliable`). For 10517 it happens to be geometrically correct,
so the test would pass, but it would document the *wrong* invariant. Instead: (1) define 10517's
correct P3 orientation **geometrically** (home GK median-x on the low-x half), (2) convert with
the **negated** ET flag, (3) assert the net recovers that geometric truth (home GK back on low-x;
per-`(team, period)` GK ends match; ET coords in SPADL bounds). Flag-independent, immune to the
placeholder problem, and exactly the property the backstop guarantees — self-justifying.

### 5.6 Feature-level closure (concern 5)

The incident was "every geometry-derived ET feature mis-oriented", and §5.1 asserts GK-x (the
direct orientation anchor) — a strong proxy but not the feature layer. Add **one**
coordinate-derived feature assertion to close the bug *class*: on the wrong-flag-vs-correct-flag
conversions, compute a cheap orientation-sensitive feature and assert the wrong-flag (now
self-corrected) output matches the correct-flag output on the ET periods.

**Feature-selection criterion (P4).** The backstop is a single rigid period-wide reflection
(`x→105−x, y→68−y` on every row), so a correct post-backstop GK-x already implies every
coordinate in that period is correctly reflected — a GK-dominated feature would just restate the
§5.1 GK-x check. The added value is specifically catching (a) a *non-uniform/partial* reflection
bug and (b) the symmetric-projection false-pass (`feedback_symmetry_test_insufficient_pin_ground_truth`).
To realize it, pick a feature derived from **multiple non-GK outfield players** — e.g.
`add_defensive_line`'s `defensive_line_x` (back-line outfielders) or a pitch-control value at a
fixed target — **not** one dominated by the GK. State this selection criterion in the test so it
is a genuinely independent pin, and use an asymmetric fixture (per
`feedback_symmetry_test_insufficient_pin_ground_truth`).

### 5.7 DGX empirical validation — **HARD SHIP GATE** (release blocker)

On the DGX (`ssh karsten@192.168.68.73`; pining cache `~/Development/silly-kicks/
xt_bandwidth_run/artifact_cache`, GS×64 + IDSSE×7, **real native rosters/GK** — the production
anchor). These are the only empirical validation of facts (a)+(b) (§3.2) and of the native-GK
anchor across all 71 matches, so they **block ship**:

- **G1 — No-op proof (the no-retrain claim).** The net is byte-identical to the pre-TF-23b
  conversion on every **correct-flag** match, across all 64 GS + 7 IDSSE matches.
  **Release blocker: if ANY correct-flag match is a non-no-op, do not ship — investigate**
  (Chesterton's Fence: a non-no-op means (a)/(b) is violated, a GK is mislabeled, or a
  period-5/anomaly path fired). Also records **which matches carry period-5 frames** — feeding
  the §3.5 enumerated scope.
- **G2 — GS self-correction on real native GK.** Convert a sample GS ET match with the
  deliberately-negated ET flag; the net restores the correct-flag output (validates the native
  anchor; now *also* committed as a CI gate via the §5.5 regenerated `gs_et` (match 10517)
  native-GK fixture).
- **G3 — IDSSE/sportec self-correction on real native GK (Gate D real-data closure).** Same
  negated-flag self-correction on a real IDSSE ET match (IDSSE×7 carry real rosters/GK). The
  **named ship gate for ADR-031 Gate D** — §5.1/5.3 cover sportec ET only synthetically. If no
  IDSSE match has ET frames in the corpus, that is recorded explicitly and Gate D closure is
  re-scoped (not silently claimed).
- The G1 changed-match list + max coordinate deltas + the period-5-match list are recorded in
  ADR-035 — that list IS the enumerated retrain scope (§6).

## 6. ADR-035 (concise)

New `docs/superpowers/adrs/ADR-035-geometric-backstop-native-adapters.md`:
- **Decision:** the native tracking adapters apply `orient_frames_to_ltr_by_geometry` as an
  always-on idempotent backstop, via a shared `direction.finalize_orientation`; backstop (not
  replacement); GS + sportec.
- **Supersedes the ADR-029 no-refactor fence — for the native adapters ONLY (P3).** ADR-029:55-56
  fenced "sportec/GS adapters **and kloppy gateway** are NOT refactored through the helper" on a
  "zero gain" rationale; that premise is dead for the native adapters (this change would add a
  third inline copy of the tail, and the dtype-safe `is_home` fix already shipped twice). The
  refactor is golden-safe (backstop no-op on correct-flag goldens; flip+label ops unchanged). The
  supersession is **scoped to the two native adapters**; the **kloppy gateway stays intentionally
  un-routed** (§2 non-goal — already oriented), so a future reader does not conclude kloppy should
  now adopt `finalize_orientation` and re-open a closed decision. Recorded so the override is seen
  as deliberate and bounded.
- **Policy injection.** `on_missing_home` (default `"raise"`, ADR-019 preserved); adapters pass
  `"warn"`. Replaces the re-implemented edge guard; one definition of "can't anchor".
- **Period-5 (PSO) policy = public-net behavior change.** The net never orients shootouts (flip
  loop restricted to `_LTR_KNOWN_PERIODS`), for **all** callers incl. the TF-23 SC/Metrica
  builders. PSO frames are excluded from geometric analysis (practical impact nil). Recorded so a
  future reader does not "fix" period 5 back into the loop.
- **Consequences:** closes ADR-031 **Gate D** (gated on G3, §5.7); realizes ADR-034's TF-23b gate,
  so the lakehouse can retire `correct_frames_to_home_ltr` **entirely** for all four tracking
  providers (its retrain trigger, not silly-kicks'); C4-free (count stays 28); no atomic mirror
  (converter, not aggregator); no default-xfn change.
- **Enumerated retrain scope** (replaces "byte-identical otherwise"): (1) ET frames of matches
  whose ET flag was wrong — bounded by GS WC2022 ET-**tracking** matches carrying the constant
  `homeTeamStartLeftExtraTime` placeholder. **≤3, not 5**: the `extratime/README.md` audit shows
  GS has 5 ET matches in *events* but only **3 in tracking**; the backstop is tracking-only
  (events converters untouched), so the events figure overcounts the tracking blast radius. Plus
  any wrong-flag IDSSE-ET. The **exact** set = the G1 non-no-op list, recorded here (authoritative
  — supersedes the estimate). (2) period-5 frames across all providers (incl. lakehouse SC/Metrica
  it must self-assess) — the G1 period-5-match list for the pining corpus is recorded here.
- **Anchor-context asymmetry** (§3.7): full-match anchor here needs no per-batch guard; the
  lakehouse 250-frame path does — one line.
- **Lazy cross-repo migration.** During the window the lakehouse runs BOTH the new adapter net
  AND its own `correct_frames_to_home_ltr`; double-application is safe because **both are
  idempotent**, so the lakehouse can retire its backstop lazily after adopting ≥4.34.0, not
  atomically.
- **Chesterton's Fence on the all-period loop** (concern 6): `git log -S` confirms the all-period
  flip loop was introduced in `cf7b29b` (TF-23, 4.33.0) with no period-5 test — an unscoped side
  effect of a PSO-free-providers PR, not an intentional fence. Recorded so it is not re-added.
- References ADR-034, ADR-031, ADR-029, ADR-019, ADR-010. No new attribution (reuses TF-23's
  ADR-053 promotion credit in NOTICE).

## 7. Version + release artifacts

- `pyproject.toml` + `silly_kicks/__init__.py` → `4.34.0`.
- `CHANGELOG.md` `### Changed`: the two adapters self-correct mis-oriented periods (retrain
  trigger; name GS + sportec + Gate D closure) **AND** the public-net change — `on_missing_home`
  param (additive, default-preserving) + the net no longer orients period-5 frames for any caller
  (incl. the TF-23 SkillCorner/Metrica builders; PSO frames are excluded from geometric analysis;
  the lakehouse self-assesses its SC/Metrica PSO retrain). Note the zero-home warn-text change.
- `TODO.md`: update Current-release header; move TF-23b out of "Research & Future Work"; note the
  lakehouse `correct_frames_to_home_ltr` deletion is now unblocked (cross-repo, not a silly-kicks
  TODO). Regenerate `gs_et` (match 10517, P3) from pining with native GK in-PR (§5.5) and correct
  the `extratime/README.md` competition label in-place ("A-League" typo → WC2022 knockout ET).
- `/final-review` before the single commit; CI green before any tag (owner-driven).

## 8. Risks

- **A real correct-flag match whose net is NOT a no-op** would indicate a GK-labeling defect or
  genuinely anomalous GK geometry. Mitigation: the DGX no-op proof (§5.7) over all 71 real
  matches; any non-no-op is investigated before ship (Chesterton's Fence).
- **Refactor regression.** `finalize_orientation` is a pure lift of the existing flip+label ops;
  the RT goldens (correct-flag, net no-op) are the byte-identity guard. Plan task 1 confirms the
  goldens empirically before any behavior change lands.
- **Fixture GK realism.** The net's native-GK consumption is already CI-covered by the §5.1
  synthetic (`is_goalkeeper`-bearing) fixture; real-data native GK is added in CI by regenerating
  `gs_et` (match 10517) from pining with native GK (§5.5) and validated at scale by §5.7 G1/G2/G3.
  The §5.5 test anchors on **geometry**, not the unreliable placeholder ET flag.

## 9. Review resolutions

### Round 1 (cross-session, 2026-06-18)
1. **Standalone ADR-035** — confirmed (Gate D closure + edge-policy warrant a discoverable ADR).
2. **(a)+(b) load-bearing facts** — verified against 4.33.0 source, quoted in §3.2; §5.7 G1 is a
   hard ship gate on them.
3. **Gate D real-data** — §5.5 (committed, native GK) + §5.7 G3 (real IDSSE-ET on DGX) close it;
   no synthetic-only closure.
4. **Period 5 (PSO)** — net flip loop skips period 5 (§3.5) with a PSO regression test.
5. **§5.5 native-vs-positional GK** — addressed by regenerating the fixture with native GK (see
   round 2 / concern 4).
6. **Retrain scope** — enumerated in §6.

### Round 2 (cross-session, 2026-06-18/19; owner gold-standard directive 2026-06-19)
1. **Concern 1 (triplication)** — **full** extraction to `direction.finalize_orientation` (§3.3);
   **supersedes the ADR-029 no-refactor fence** (premise obsolete; golden-safe). Owner-approved
   under the gold-standard directive.
2. **Concern 2 (re-implemented guard)** — `on_missing_home` param injected into the net (§3.4),
   default `"raise"`; adapters pass `"warn"`. Net raise *condition* kept byte-identical (no
   `is_player.any() and`); single `is_home` mask in the helper removes the pre/post-coercion drift.
3. **Concern 3 (period-5 public-API change)** — accepted as a deliberate gold-standard public-net
   change (the net should never orient PSO for anyone); documented in CHANGELOG for direct callers
   + TF-23 SC/Metrica builders; PSO-excluded-from-analysis stated; cross-repo SC/Metrica retrain
   flagged; pining-corpus period-5 set enumerated via G1. (Reversed the v1 "parametrize with
   backcompat default" idea — that preserved garbage behavior, the opposite of gold standard.)
4. **Concern 4 (native-GK no-op only on manual DGX)** — **regenerate `gs_et/` with native GK in
   this PR** (§5.5), making the production-anchor self-correction a permanent CI gate.
5. **Concern 5 (GK-x proxy vs feature)** — add one coordinate-derived feature-level assertion
   (§5.6).
6. **Concern 6 (Chesterton's Fence on the all-period loop)** — verified via `git log -S` (intro'd
   in `cf7b29b`, no period-5 test); recorded in ADR-035.
7. **Concern 7 (minors)** — single-match `game_id` access guarded (`if len(out)`); velocities
   confirmed absent at insertion; lazy idempotent cross-repo migration noted in ADR-035 (§6).

### Round 3 (cross-session review of rev-2, 2026-06-19; design APPROVED — polish only)
1. **P1 (dead `"skip"`)** — dropped; `on_missing_home: Literal["raise","warn"]` (§3.4 / §3.3).
2. **P2 (half-mutation)** — `finalize_orientation` copies at entry; clean value semantics, no
   input mutation (§3.3).
3. **P3 (partial supersession)** — ADR-035 scopes the ADR-029 override to the native adapters
   only; kloppy stays intentionally un-routed (§6).
4. **P4 (feature selection)** — §5.6 mandates a multi-non-GK-outfield-player feature (not
   GK-dominated) on an asymmetric fixture.
5. **P5 (licensing)** — resolved by the owner (2026-06-19): committing GS-*derived* test data is
   fine; only whole matches/datasets to HF are off-limits. §5.5's small slice complies.
6. **P6 (direct helper unit test)** — added (§5.2b).
- **Correction (supersedes Round-1 #3/#5 + Round-2 #4):** the committed `gs_et/` fixture has **no
  native `is_goalkeeper`/roster**; the net's native-GK *consumption* is already CI-covered by the
  §5.1 synthetic fixture. See the corrected §5.5.

### Round 4 (lakehouse provenance clarification, 2026-06-19)
1. **`gs_et` IS WC2022** — match 10517 is a Gradient Sports WC2022 knockout ET match; the
   README's "A-League" is a maintainer typo (GS = PFF FC WC2022; A-League = SkillCorner).
   Resolution: regenerate `gs_et` (10517, P3) **from pining** (it's in the GS×64 corpus) with
   native GK + **correct the label in-place** — no longer an open provenance flag. (§5.5)
2. **Geometric ground truth, not the flag** — the test must NOT anchor on
   `home_team_start_left_extratime=True` (the constant GS-ET placeholder TF-23b exists to fix;
   `reference_gs_et_flag_placeholder_unreliable`). Define 10517's P3 orientation geometrically
   (home GK median-x low half), negate the flag, assert the net recovers it. (§5.5)
3. **Retrain scope ≤3, not 5** — GS has 5 ET matches in *events* but only 3 in *tracking*; the
   backstop is tracking-only, so the events "5" overcounts. §6 fixed; G1 list is authoritative.

### Round 5 (plan review, 2026-06-19 — plan approved; refinements reflected back into the spec)
1. **Net `copy: bool=True` knob** (§3.4) — `finalize_orientation` already owns a copy-at-entry, so it
   passes `copy=False`; a `convert_to_frames` call drops from three full-frame copies to two on
   tracking-scale data. Additive, default-preserving (direct/lakehouse callers byte-identical). Public-
   net API addition alongside `on_missing_home` — recorded in CHANGELOG + ADR-035 too (plan-review R2).
2. Plan-only refinements (no spec change): a red-first test pins the `copy=False` contract (R1); the
   gs_et regen fails loud / derives `home_team_id` (R2); the ADR-019 `ids_match` comment is carried onto
   `finalize` (R3-of-round-1 / Chesterton); G1 uses strict `check_dtype=True` (R3).

**PR-S index confirmed:** `pr-s99` / **4.34.0** (PR-S98 was 4.33.0; owner delegated, locked).
