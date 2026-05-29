# TF-24 — Tracking-defaults calibration harness

Optuna-TPE calibration of three engineering-choice tracking defaults against real multi-provider
tracking data. The harness **recommends** values + writes an auditable report; it does **not**
change the library constants (that is a separate "apply" PR after a real sweep).

Pure, provider-agnostic logic lives in `silly_kicks.calibration` (requires the `[calibration]`
extra: `pip install -e ".[calibration]"`). All I/O + orchestration is here in `scripts/`.

## Two stages

| Stage | Target params | Objective | Direction |
|-------|---------------|-----------|-----------|
| 1 | `infer_ball_carrier`: `tolerance_m`, `beta`, `gamma` | carrier accuracy `(inferred == actor).mean()` over carrier-actor actions, equal-provider-weight | maximize |
| 2 | `LinkParams.k3`, off-ball `pre_seconds`, `min_displacement_m` | augmented-VAEP held-out **Brier** (per-provider match-stratified CV, equal-provider-weight) | minimize |

Run Stage 1 first; it writes `carrier_best.json`, which Stage 2 consumes (`derive_team_in_possession`
→ DAS context). Stage 2 uses a **frozen exogenous xT artifact** (fit on a corpus disjoint from the
calibration matches) for train–serve-consistent, leak-free feature extraction.

## Environment

- `PINING_FOR_THE_DATA_TOKEN` — owner token (enables Gradient Sports). SkillCorner + IDSSE are
  public via the built-in public token, so no env var is needed for those.
- `PINING_API_URL` — override the pining base URL (defaults to the deployed instance).
- `DATABRICKS_HOST` / `DATABRICKS_HTTP_PATH` / `DATABRICKS_TOKEN` — only needed when
  `--xt-corpus-source databricks` (the `bronze.spadl_actions` xT corpus) or `--source databricks`.

## Usage

```bash
# Stage 1 — carrier accuracy (all three providers from pining)
python scripts/calibrate_tracking_defaults.py --stage 1 --source pining \
    --providers skillcorner idsse gradientsports \
    --n-trials 100 --store tc3_stage1.db
# -> writes carrier_best.json + calibration_report.{json,md}

# Stage 2 — augmented-VAEP Brier (consumes carrier_best.json)
python scripts/calibrate_tracking_defaults.py --stage 2 --source pining \
    --providers skillcorner idsse gradientsports \
    --n-trials 60 --store tc3_stage2.db \
    --carrier-best carrier_best.json --xt-artifact calibration_xt.npz

# Diagnostics — TF-25 provider-specific-defaults check (per-provider k3 sensitivity)
python scripts/calibrate_tracking_defaults.py --stage diagnostics --source pining \
    --providers skillcorner idsse gradientsports --store tc3_stage2.db
```

### xT corpus source (`--xt-corpus-source`)

- `pining` (default) — **id-space-safe**: fits xT on pining matches *held out* from the
  calibration set, in the same id space as the calibration loads. No mapping needed.
- `databricks` — fits on `bronze.spadl_actions` minus the calibration matches. Requires the bronze
  `game_id` space to **match** the pining match_ids; if it doesn't, the fit **fails closed**
  (refuses rather than silently leaking held-out matches into the grid). Provide the id mapping
  first.

Pre-fit the artifact once with `--xt-artifact <path>` so it is reused across runs and recorded in
the manifest (sha256 + corpus match-ID set + fit date).

## Outputs

- `carrier_best.json` (Stage 1) — the recommended carrier params.
- `<report-out>.json` / `<report-out>.md` (default `calibration_report.*`) — the ruthless result
  plus a **calibration manifest**: silly-kicks / ruthless / xgboost versions, git context,
  `n_trials`, seed, source, the per-provider match-ID list, the frozen-xT artifact identity, and
  the diagnostics (excluded providers, DAS-degradation counts). This is the trust anchor for the
  downstream "apply" PR's "Optuna-calibrated against \<fold\> on \<date\>" claim.

## Tests

- CI (no network): `pytest tests/calibration/ -m "not e2e"` — pure objectives, cache-equivalence,
  CV, gates, stubbed loaders.
- Real data (opt-in): `pytest tests/calibration/test_calibration_e2e.py -m e2e` — SkillCorner
  Stage-1 + Stage-2 on public pining (no token needed). Set `RUN_HEAVY_E2E=1` to add the IDSSE
  Stage-1 e2e (~419 MB DFL/Sportec XML download).
