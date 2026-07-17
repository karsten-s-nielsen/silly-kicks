# Handoff — Soccermatics Pro tracking metrics vs silly-kicks (prior analysis, 2026-07-15)

> **What this is.** A prior analysis pass run on 2026-07-15 from a *different* project context (the
> GkNexus/Bounou workspace — its session memory is NOT visible from this repo, hence this file).
> It is provided as **handoff context to cross-check, not as settled fact**. You are expected to redo
> the analysis of the course material from your own reading. Treat every claim below as a hypothesis
> to verify — especially the capability map, which was a point-in-time read of **silly-kicks v4.46.0**
> and may already be stale.

---

## 1. The question asked

> "The latest course materials from Soccermatics Pro describe a number of tracking metrics.
> Investigate and consider which, if any, could potentially add value to silly-kicks if we were to add
> them (they do **not** have to be GK related)."

## 2. Method used (for reference — feel free to differ)

Six parallel research agents: five extracting the metrics taught in each module cluster from the
`.srt` transcripts, one mapping silly-kicks' *current* tracking-metric surface. The gap was then
judged as (course metric) minus (what silly-kicks already has), weighted by whether the ingested data
could actually support it.

## 3. Where the material lives

`D:\[Karsten]\Dropbox\[Coaching]\Twelve Football\Soccermatics Pro\`
**⚠ literal square brackets in the path** — PowerShell needs `-LiteralPath`; Python `glob` breaks
(brackets = char class) so use `Path`/`os.listdir` with direct strings. Every lesson has a whisper
`.srt` next to its `.mp4` (the readable source); some have slide PDFs (Read tool can't do PDFs here —
use `pypdf`/`docling`).

The tracking curriculum is modules **12–17** (all `.srt`-transcribed as of 7/15):
| Module | Covers |
|---|---|
| 12 Introduction to tracking data | data taxonomy; **SkillCorner data model** (the important one) |
| 13 Physical metrics | velocity/acceleration; speed zones, HSR, sprints, PSV-99 top speed |
| 14 Formations and tactics | avg positions, compactness, line height, convex hull |
| 15 Pitch control | Spearman's model (+ his guest masterclass); code walkthrough; **xT fusion** |
| 16 Scouting with tracking data | Barcelona zonal model; off-ball runs; defensive positioning; tracking features on xT |
| 17 Introduction to deep learning | Pegah Rahimian: GNN/CNN-LSTM/offline RL (mostly aspirational) |

## 4. The headline finding (verify this first)

**silly-kicks already implements most of the tracking curriculum, frequently beyond what the course
teaches.** So the answer was a short list, not a menu. If this is right, the interesting work is in
the two gaps below, not in re-building pitch control or team shape.

## 5. Capability map as found (silly-kicks v4.46.0, 2026-07-15 — RE-VERIFY)

**Key data fact:** silly-kicks ingests **full per-frame tracking, not just events**.
`tracking/skillcorner.py::convert_to_frames` reads `bronze.skillcorner_tracking` = every player x,y +
ball per frame, **10 Hz** for SkillCorner (25 Hz Sportec/Metrica). Canonical 20-col schema
`TRACKING_FRAMES_COLUMNS`, **SPADL-metric 105×68 bottom-left, LTR-normalised**, carrying `speed`,
`visibility`, `confidence`; linked to actions via `link_actions_to_frames(tol=0.2s)`. The separate
`dynamic_events.csv` (~294-col possession/passing-option/off-ball-run/pressing stream) is the *event*
layer, distinct from the frame layer. ⇒ the substrate supports essentially any tracking metric.

| Area | Status found | Where |
|---|---|---|
| Formation / team shape (= module 14) | **BUILT, beyond course** — centroid, convex-hull area, length/width, stretch index, defensive-line height, inter-line gaps (Ward), Delaunay **role-grid** | `tracking/_team_shape.py`, `_defensive_line.py`, `_shape_graph.py`, `_line_breaking.py` |
| Pitch control (= module 15) | **BUILT, beyond course** — Spearman + Fernández-Bornn + Voronoi w/ caching. **Module 15's "pitch control × xT" fusion IS silly-kicks' OBSO** (PPCF×transition×EPV) and `player_influence` (PC-share×xT) | `tracking/pitch_control/`, `_obso.py`, `_cover_shadows.py`, `_space_creation.py`, `_player_influence.py`, `_das.py` |
| Defensive positioning (module 16.1) | **Primitives BUILT** — 3-method pressure (Andrienko/Link/Bekkers), cover shadows/lane control, defensive line, GK zone-closing. Missing only the course's ±outcome-attribution *scoring* framing | `tracking/pressure.py`, `_cover_shadows.py`, `_gk_influence.py` |
| Off-ball runs (module 16.2) | **PARTIAL** — simple displacement/threshold detector only; **no run typology, no run-value model** (docstring itself disclaims sprint-intensity filtering) | `tracking/_off_ball_runs.py` |
| Tracking-context xT (module 16.3) | **PARTIAL/NARROW** — classic `xthreat/` is event-only (no tracking). Tracking-context xT exists **only** for the GK-distribution case (xT-GK v1/v2) | `xthreat/`, `tracking/_xt_gk.py`, `xtgk/` |
| **Physical / locomotor (= module 13)** | **ABSENT** — only instantaneous `speed`. No acceleration, no speed zones, no HSR/sprint counts, no distance-covered, no PSV-99. Savitzky-Golay velocity machinery is *one derivative short* | `tracking/preprocess/_velocity.py` |
| Possession value / xT-GK / GK stack | **BUILT**, most mature area (v2 self-reported NOT construct-validated) | `xtgk/`, ADR-036 |

## 6. Ranked recommendation reached (the actual handoff)

1. **⭐ Physical / locomotor metrics (module 13) — biggest clean gap, lowest risk.**
   Absent entirely; substrate already ingested and already smoothed to velocity. Course thresholds:
   running 15–20, **HSR 20–25, sprint >25 km/h**; sprint/HSR counts need a >1 s bout; accel events
   1.5–3 m/s² sustained ≥0.7 s; **PSV-99** = 99th-percentile of per-sprint top speeds (explicitly a
   bias-correction against tracking-error spikes); >12 m/s ⇒ treat as bad data.
   - **Caveat A (accuracy):** SkillCorner is **broadcast** tracking — off-camera players are
     NN-*predicted*, accurate to only **5–10 m** (module 12; David's own quote). ⇒ *total distance is
     the weakest* metric (undercounts off-camera; course flags this); **top speed / PSV-99 is the most
     robust**. Gate on the `visibility`/`confidence` columns silly-kicks already carries.
   - **Caveat B (fit):** this is athletic/load profiling — a **new genre** vs silly-kicks' stated
     mission ("classify and value on-ball actions using SPADL and VAEP"). Biggest *gap*, loosest
     *fit*. Worth a deliberate scope decision, not an automatic yes.
   - Synergy: an intensity layer would also upgrade the displacement-only off-ball-run detector.
2. **Typed & valued off-ball runs (module 16.2) — best strategic fit.**
   Extends the valuation mission to off-ball; reuses primitives already present (pitch control + xT +
   a run detector). New logic = the typology (target / disruptive / space-creation runs) and the
   course's **surface-max valuation** (value the run by the max of the pitch-control × xT surface for
   the space it opened, *not* the literal reception point). Bonus: SkillCorner's dynamic events
   already ship a **10-type run taxonomy**, so typing can come from the data rather than be re-derived.
3. **(Thin) general tracking-context xT / packing / pass-under-pressure (module 16.3)** — mostly
   already absorbed by HybridVAEP's action_context features; packing ≈ the existing line-breaking
   detector. Low marginal value.

**Judged not worth it:** module 14 (formations) and module 15 (pitch-control × xT fusion) — already
built beyond the course; module 17 (deep learning: GNN / CNN-LSTM / offline RL) — off-architecture for
a pure-function, zero-I/O library.

## 7. Where this analysis is most likely to be WRONG — cross-check these

- **Is physical really absent?** Grep for acceleration / sprint / HSR / distance-covered / top-speed /
  speed-zone tied to an actual metric (not just a docstring). Has anything landed since v4.46.0?
- **Is the off-ball run detector really untyped/unvalued?** Read `_off_ball_runs.py` end-to-end
  rather than trusting the summary.
- **Is packing genuinely covered by `_line_breaking.py`?** That equivalence was asserted, not proven.
- **Is OBSO genuinely the same thing as module 15 lesson 4's fusion?** Verify the formulation, not
  just the vibe.
- **The "beyond the course" claim** for formation/pitch control — spot-check against what the lessons
  actually teach before repeating it.
- **The genre-fit argument** against physical metrics is a *judgement*, not a fact. A different reading
  (silly-kicks as a general tracking-feature library, not strictly on-ball valuation) flips the ranking.

## 8. Status

**No code was written.** This was a consideration pass only; nothing was proposed to, or merged into,
the repo. Next step if pursued was a proper brainstorm/spec for whichever of (1) or (2) wins.
