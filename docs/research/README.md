# Research discipline — in-repo pointer (E1–E7 + the validity ladder)

The research under `docs/research/` follows a shared **research-discipline** standard: a pre-registration
checklist for building, validating, reviewing, and shipping quantitative football-analytics claims. The
**canonical, maintained copy is the `research-discipline` skill** (in the `mad-scientist-skills` plugin —
invoke it while designing or validating a metric). A session without that plugin, or a human reading a
committed spec, can resolve the shorthands below from this page; it is a stub, not a fork.

The specs and memos in this repo cite these by number (e.g. "closes E6/E2", "E7 only", "Part F multi-pass").

## Metric-design anti-patterns (E1–E7)

Pre-register a "no" to each before a metric is called validated.

- **E1 — Learning the provider's label instead of the construct.** If the supervised target *is* a vendor's
  event label (or it leaks in as a feature), you re-implemented the vendor. A provider label is a footnote
  sanity check, never the target or the definition.
- **E2 — Ignoring defensive block type.** Comparing a defensive quantity across teams/players without
  conditioning on block (low / medium / high) mixes regimes. Report within-block.
- **E3 — Validating only through immediate turnovers.** One binary, near-zero-base-rate outcome is thin. Use
  ≥2 channels: threat/danger reduction, forced-backward passes, delay, retention, second balls.
- **E4 — Treating a case study as sufficient evidence.** Clips illustrate; they do not validate. The evidence
  is the systematic, stratified analysis with intervals.
- **E5 — Ranking without opportunity normalisation.** Raw counts — and per-90 alone — leave an
  opportunity-volume confound. Normalise per opportunity (per opponent possession, per minute of relevant
  opponent possession, …).
- **E6 — One formula across all game phases.** Pooling across attacking phase (build-up / create / finish)
  hides phase-specific behaviour. Report per phase, or justify pooling.
- **E7 — Computing a metric on unobserved positions.** Broadcast tracking interpolates off-camera players; a
  metric averaged over interpolated positions measures the vendor's imputation model. Gate on the per-player
  detection flag, condition on where the ball is, and report the observability rate as a result, not a
  footnote. (Optical / in-stadium tracking is exempt — verify which you have.)

E2 and E6 are two faces of one failure (unconditioned pooling); test the two axes independently.

## The validity ladder

**construct → face → predictive → robustness** — report all four; do not skip to a single outcome-AUC.
Pre-register the direction and decision rule (effect floor, `n_min`, expected sign) before seeing the numbers.
Efficacy without an outcome label is legitimate — say so rather than invent a proxy (which reintroduces E3).

## Reviewing

A single review pass is a sample, not a verdict: for anything gating a commit/submission/claim, run **≥3
independent passes and report their agreement**; pin the reviewer model id + skill version; probe omissions.

## Full discipline

The complete standard — the deep validity ladder (ICC grain cross-checks, plant/scramble/ceiling controls,
bootstrap the unit), attribution/identifiability, the data-quality preflight, reproducibility/provenance, the
full multi-pass review protocol, and the pre-ship defect smell-test — lives in the `research-discipline`
skill. Attribution: the E1–E6 taxonomy is from Rahimian (2026); E7 and the review-reliability guidance are
in-house observability extensions. This repo quotes no restricted numbers and names no private data.
