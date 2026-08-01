"""Train the bundled GK-distribution completion model (xT-GK RAV; logistic, pure-numpy serve).

Mirrors scripts/train_xshot_occurrence.py at a fraction of the weight (logistic, observable
label, few features). The GREEN GATE is the NATIVE-ORIGIN POOLED out-of-fold calibration
(review #1): one AUC over ALL native-origin rows with a bootstrap-CI lower bound > 0.5, NOT a
per-fold mean over the small native minority. Bundled default is GS-only (review R1).

Usage:
    python scripts/train_gk_completion.py --providers gradientsports --max-per-provider 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _loader_pining import load_matches

from silly_kicks._calibration_metrics import ece as _ece
from silly_kicks._calibration_metrics import reliability_slope as _reliability_slope
from silly_kicks.tracking._gk_completion import (
    GK_COMPLETION_FEATURE_NAMES,
    GkCompletionModel,
    prepare_gk_completion_training_data,
)

_WEIGHTS_ROOT = Path(__file__).resolve().parent.parent / "silly_kicks" / "tracking" / "_gk_completion_weights"
_WEIGHTS_DIR = _WEIGHTS_ROOT / "default"
_SKILLCORNER_WEIGHTS_DIR = _WEIGHTS_ROOT / "skillcorner"
_N_NATIVE_FLOOR = 100
_GKPASS_AUC_FLOOR = 0.70  # D-S3: SkillCorner GK-pass held-out floor
_ECE_TOL = 0.10  # C1: calibration tolerance (expected calibration error)
_SLOPE_TOL = 0.25  # C1: reliability-slope within [1-tol, 1+tol]; NaN slope (degenerate) is not gated
# Corpus-identity probe tolerance (SK-91). The re-bundle SHIPS the committed coefficients (served =
# load(); only the gate fields are added), so this never affects the served weights -- it only decides
# abort-vs-attach by checking the fresh full-data probe still describes the SAME corpus+model. Byte
# identity (1e-9) is unachievable across an unrecorded original `tracking_limit`: the GS frame subset
# shifts the density feature by a hair (98% vs 96.3% finite -> coef move <=0.0056) even when the row
# set reproduces EXACTLY. A meaningful tolerance separates that float/tracking_limit noise from a real
# retrain (the SkillCorner data-drift retrain shifted coef ~0.47, ~9x this floor); SkillCorner's own
# frames load whole at any cap so it still matches to ~0.
_CORPUS_IDENTITY_ATOL = 0.05


def _extract(providers, max_per_provider, tracking_limit, *, shard_root):
    from scripts._driver import for_each, shard_path

    def _work(item):
        prov, mid, actions, frames, _home = item
        try:
            X, y, groups = prepare_gk_completion_training_data(actions, frames=frames)
        except ValueError as exc:  # a single near-degenerate match shouldn't kill the run
            # Returned as an EMPTY shard rather than re-raised: `for_each` would otherwise record it
            # as a FAILURE and three in a row would abort the pass, whereas this has always been a
            # skip. An empty shard also stops a resume re-deciding the same degenerate match.
            print(f"  {prov}/{mid}: skipped ({exc})", flush=True)
            return None
        X = X.copy()
        X["_y"] = y
        X["_group"] = groups
        return X

    res = for_each(
        load_matches(providers=providers, max_per_provider=max_per_provider, tracking_limit=tracking_limit),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=shard_root,
        # What determines a shard's CONTENT: the extractor and the frame depth it sees.
        # `--tracking-limit` is load-bearing here and NOT merely a cap -- SK-91 measured that a small
        # cap starves the SkillCorner derived-GK, over-flagging keepers and inflating the frame-derived
        # GK-pass domain (461 vs 542 on full frames). A capped run and a full run are DIFFERENT corpora
        # and must never share a generation.
        token_inputs={
            "extractor": "prepare_gk_completion_training_data",
            "tracking_limit": tracking_limit,
        },
        tag="gk_completion",
        label="match",
    )
    if res.failures:
        raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")

    # Combined from THIS PASS'S keys, not `_driver.reconcile`: no partition surface, so a
    # whole-generation read would fold in matches from a wider earlier run. See its docstring.
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
    if not parts:
        raise SystemExit("No usable training data.")
    return pd.concat(parts, ignore_index=True)


def _bootstrap_auc_ci(y, p, n_boot=2000, lo=2.5, seed=0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, lo)), float(np.percentile(aucs, 100 - lo))


def _per_type_gate_from_oof(oof: np.ndarray, y_all: np.ndarray, X_all) -> tuple[dict, dict]:
    """Per-type serve gate over the model's 3-way {goalkick, throw_in, other} partition (matches
    GkCompletionModel._base_rates). Returns (type_serve_mode, type_gate_metrics) from held-out OOF.
    A degenerate/insufficient bucket (AUC undefined or n < _GATE_N_MIN) -> base_rate via the shared
    serve_mode_from_lcb. Bucket masks use the feature columns is_goalkick / is_throw_in."""
    from sklearn.metrics import roc_auc_score

    from silly_kicks.tracking._gk_completion import serve_mode_from_lcb

    ok = np.isfinite(oof)
    is_gk = X_all["is_goalkick"].to_numpy() == 1.0
    is_ti = X_all["is_throw_in"].to_numpy() == 1.0
    buckets = {"goalkick": is_gk, "throw_in": is_ti, "other": ~(is_gk | is_ti)}
    serve_mode: dict[str, str] = {}
    metrics: dict[str, dict] = {}
    for name, b in buckets.items():
        m = b & ok
        n = int(m.sum())
        if n < 2 or len(np.unique(y_all[m])) < 2:
            auc = lcb = None  # degenerate (e.g. near-empty GK throw-in positive class)
        else:
            auc = float(roc_auc_score(y_all[m], oof[m]))
            lcb = float(_bootstrap_auc_ci(y_all[m], oof[m])[0])
        serve_mode[name] = serve_mode_from_lcb(lcb, n)
        metrics[name] = {"auc": auc, "lcb": lcb, "n": n}
        print(f"  [gate {name}] n={n} auc={auc} lcb={lcb} -> {serve_mode[name]}", flush=True)
    return serve_mode, metrics


def predictions_moved(old: dict, new: dict, *, probe_old, probe_new, atol: float = 1e-6) -> bool:
    """Did the SERVED probabilities change -- each model evaluated on the coordinates IT sees?

    Keyed on behaviour, not on parameter deltas, following the house pattern: `_chirality.py`
    fingerprints model OUTPUT on a fixed probe frame and `_feature_contract.py` fingerprints the
    feature vector on one. A parameter-delta rule is wrong in BOTH directions, and both are
    measured in ``tests/scripts/test_train_gk_completion.py``: a pure translation moves ``mean`` by
    metres while standardisation absorbs it exactly, so every served probability is identical and a
    max-over-arrays guard reads it as a retrain; and a change confined to standardisation moves
    served probabilities while the coefficients sit byte-still, which a coefficients-only guard
    reads as no change.

    TWO probes, not one. ``probe_old`` is the design matrix the committed model was fit on and
    ``probe_new`` the one the fresh fit was; they are the SAME array whenever the feature space did
    not move, which is the ordinary case. A single shared probe was drafted first and MEASURED
    unable to express the property -- it asks whether two functions agree on one input, when the
    question is whether the model behaves the same on the data each version actually sees.

    The probe should be the model's own design matrix, never synthetic noise: a probe that does not
    exercise the region a retrain changed would report "no movement" for a real retrain.
    """

    def _serve(w: dict, p) -> np.ndarray:
        z = (np.asarray(p, dtype=float) - np.asarray(w["mean"], dtype=float)) / np.asarray(w["std"], dtype=float)
        return 1.0 / (1.0 + np.exp(-(z @ np.asarray(w["coef"], dtype=float) + float(w["intercept"]))))

    return not np.allclose(_serve(old, probe_old), _serve(new, probe_new), atol=atol, rtol=0.0)


def _as_weights(m) -> dict:
    """The four served parameters of a `GkCompletionModel`, as `predictions_moved` wants them."""
    return {"coef": m._coef, "intercept": m._intercept, "mean": m._mean, "std": m._std}


def _superseded_coef(committed, feats) -> dict | None:
    """The coefficients this run replaces, recorded in metrics.json. ``None`` on a first bundle.

    A weights change is reviewable six months later only if the thing it replaced is written down;
    the artifact itself keeps no history, and git shows the npz as a binary blob.
    """
    if committed is None or committed._coef is None:
        return None
    return dict(zip(feats, [float(v) for v in committed._coef], strict=True))


def _validate_bundling_args(args) -> None:
    """Refuse the retrain cases whose right instrument the artifact format cannot supply.

    With no probe persisted beside the weights today, ``--feature-space moved`` ALWAYS refuses. That
    is the point: a loud refusal naming why the comparison is impossible beats silently answering
    the wrong question, which is what ``probe_old = probe_new = X_all`` would do.
    """
    if args.mode != "retrain":
        return
    if args.feature_space is None:
        raise SystemExit("--mode retrain requires --feature-space {unchanged,moved}; see --help.")
    if args.feature_space == "moved" and args.probe_old is None:
        raise SystemExit(
            "--feature-space moved requires --probe-old: the committed weights directory stores "
            "coef/intercept/mean/std but NOT a design matrix, so the pre-change feature space "
            "cannot be reconstructed from the artifact. Without it this guard would serve the "
            "committed model on coordinates it never saw, report a difference caused by the "
            "coordinate change alone, and stamp the artifact `retrained` when nothing behavioural "
            "moved -- the exact defect it exists to catch. Persist a probe sample beside the "
            "weights (ADR-011 follow-up) or re-run with the pre-change extractor."
        )


def _assert_retrain_moved_predictions(args, committed, model, X_all, feats) -> None:
    """`--mode retrain`: the fresh fit must SERVE differently from the committed weights."""
    if committed is None:
        return  # first-ever bundle: there is nothing to have moved away from
    probe_old = pd.read_parquet(args.probe_old)[feats] if args.probe_old else X_all
    if not predictions_moved(_as_weights(committed), _as_weights(model), probe_old=probe_old, probe_new=X_all):
        raise SystemExit(
            "RETRAIN produced the committed model's served predictions unchanged -- the input "
            "change never reached the model. Shipping this as a retrain would be a false claim. "
            f"(reason given: {args.reason!r})"
        )


def _train_skillcorner(args) -> int:
    """D-S1 GS-transfer re-measurement on the CORRECTED native label + the SkillCorner gate (D-S3/C1).

    Decides whether to bundle distinct SkillCorner weights or alias to the GS ``default``: the
    prior 0.50 non-transfer was on the WRONG (proxy) label and is void; this re-measures GS-transfer
    on the native-completion label (training is native-only via the F1/G1 filter). Per sub-domain
    (overall / goal-kick / GK-pass), reports SkillCorner-fit OOF vs GS-transfer AUC (+bootstrap LCB)
    and ECE. Gate: GK-pass AUC LCB > 0.70 AND ECE <= tol."""
    # Read, not re-checked: `main()` already REFUSED a dirty tree before dispatching here, so
    # this only recovers the same facts for the artifact. A direct call (the smoke test) still
    # gets truthful values rather than a missing key.
    from scripts._provenance import git_provenance

    run_prov = git_provenance()

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cache = getattr(args, "cache_features", None)
    if cache and Path(cache).exists():
        print(f"=== loading cached SkillCorner features from {cache} ===", flush=True)
        df = pd.read_parquet(cache)
    else:
        print("=== extracting SkillCorner GK distributions (native-label-filtered, F1/G1) ===", flush=True)
        df = _extract(
            ["skillcorner"], args.max_per_provider, args.tracking_limit, shard_root=Path(args.shard_dir) / "skillcorner"
        )
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
            print(f"  cached features -> {cache}", flush=True)
    feats = GK_COMPLETION_FEATURE_NAMES
    X_all = df[feats]
    y_all = df["_y"].to_numpy(int)
    groups = df["_group"].to_numpy()
    is_gk = df["is_goalkick"].to_numpy() == 1.0
    print(
        f"N={len(df)} base_rate={y_all.mean():.3f} matches={len(np.unique(groups))} "
        f"goalkick={int(is_gk.sum())} gk_pass={int((~is_gk).sum())}",
        flush=True,
    )

    # SkillCorner-fit out-of-fold
    oof = np.full(len(df), np.nan)
    n_splits = max(2, min(5, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits).split(X_all, y_all, groups):
        if len(np.unique(y_all[tr])) < 2:
            continue
        m = GkCompletionModel().fit(X_all.iloc[tr], pd.Series(y_all[tr]))
        oof[te] = m.predict_proba(X_all.iloc[te])
    ok = np.isfinite(oof)

    # GS-transfer: the bundled GS default served on the SkillCorner rows
    gs = GkCompletionModel.from_variant("default")
    gs_p = gs.predict_proba(X_all)

    def _report(name, mask):
        m = mask & ok
        n = int(m.sum())
        if n < 10 or len(np.unique(y_all[m])) < 2:
            print(f"  [{name}] n={n}: insufficient/degenerate -- skipped", flush=True)
            return None
        sc_auc = roc_auc_score(y_all[m], oof[m])
        sc_lo, _ = _bootstrap_auc_ci(y_all[m], oof[m])
        gs_auc = roc_auc_score(y_all[m], gs_p[m])
        gs_lo, _ = _bootstrap_auc_ci(y_all[m], gs_p[m])
        sc_ece, gs_ece = _ece(y_all[m], oof[m]), _ece(y_all[m], gs_p[m])
        sc_slope, gs_slope = _reliability_slope(y_all[m], oof[m]), _reliability_slope(y_all[m], gs_p[m])
        print(
            f"  [{name}] n={n} base={y_all[m].mean():.3f} | SC-fit AUC={sc_auc:.3f}(LCB {sc_lo:.3f}) "
            f"ECE={sc_ece:.3f} slope={sc_slope:.2f} | GS-transfer AUC={gs_auc:.3f}(LCB {gs_lo:.3f}) "
            f"ECE={gs_ece:.3f} slope={gs_slope:.2f}",
            flush=True,
        )
        return dict(
            n=n,
            base=float(y_all[m].mean()),
            sc_auc=float(sc_auc),
            sc_lcb=float(sc_lo),
            sc_ece=float(sc_ece),
            sc_slope=float(sc_slope),
            gs_auc=float(gs_auc),
            gs_lcb=float(gs_lo),
            gs_ece=float(gs_ece),
            gs_slope=float(gs_slope),
        )

    print("\n=== D-S1 GS-transfer re-measurement (corrected native label) ===", flush=True)
    rep = {
        "overall": _report("overall", np.ones(len(df), bool)),
        "goalkick": _report("goalkick", is_gk),
        "gk_pass": _report("gk_pass", ~is_gk),
    }

    # Decision (D-S1): alias to GS ONLY if GS actually transfers (clears the floor + is no worse than
    # the SC fit); else bundle the SC model when it clears the POINT floor + calibration (LCB + n are
    # REPORTED as uncertainty, m3 -- the floor is the point estimate, not the conservative LCB, which
    # on a few-hundred-row sample would reject a clearly-better model). Aliasing to a worse-than-chance
    # GS transfer is never correct.
    gkp = rep["gk_pass"]
    if gkp is None:
        decision = "alias_gs_insufficient"
    else:

        def _slope_ok(s):  # NaN (degenerate, single occupied bin) is not gated
            return (not np.isfinite(s)) or abs(s - 1.0) <= _SLOPE_TOL

        gs_transfers = gkp["gs_auc"] >= _GKPASS_AUC_FLOOR and gkp["gs_ece"] <= _ECE_TOL and _slope_ok(gkp["gs_slope"])
        sc_usable = gkp["sc_auc"] >= _GKPASS_AUC_FLOOR and gkp["sc_ece"] <= _ECE_TOL and _slope_ok(gkp["sc_slope"])
        if gs_transfers and gkp["gs_auc"] >= gkp["sc_auc"]:
            decision = "alias_gs"  # GS transfers and is no worse -> no distinct weights
        elif sc_usable:
            decision = "bundle_skillcorner"  # SC clears floor + calibrated; GS doesn't transfer
        elif gkp["sc_auc"] > gkp["gs_auc"]:
            decision = "bundle_skillcorner_below_floor"  # neither ideal, but SC >> GS -> ship SC, flag
        else:
            decision = "alias_gs_below_floor"
    print(f"\nDECISION: {decision}  (GK-pass floor {_GKPASS_AUC_FLOOR}, ECE tol {_ECE_TOL})", flush=True)

    bundled = False
    # Bound OUTSIDE the branch: on a no-bundle decision nothing is loaded, and the metrics dict
    # below still has to answer "what did this supersede" with a definite None.
    committed_before = None
    if decision.startswith("bundle_skillcorner"):
        # `model` = fresh full-data fit, used as the CORPUS-IDENTITY PROBE only. The SERVED artifact is
        # the committed model (its bytes), so the OOF gate provably describes the served model AND
        # coef stay byte-identical (spec v3 §5 + the additive-only re-bundle check). Re-fit is NEVER
        # persisted on a re-bundle.
        model = GkCompletionModel().fit(X_all, pd.Series(y_all))
        sm, gm = _per_type_gate_from_oof(oof, y_all, X_all)
        try:
            committed = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)  # committed coef = the served bytes
        except FileNotFoundError:
            committed = None
        committed_before = committed
        if committed is None:
            served = model  # first-ever bundle: nothing committed to preserve -> ship the fresh fit
        elif args.mode == "rebundle":
            # KEPT on parameters, deliberately -- see the block comment on _CORPUS_IDENTITY_ATOL.
            # fit()/load() populate the coef arrays above; narrow off Optional for the type checker.
            assert model._coef is not None and model._mean is not None and model._std is not None  # noqa: S101
            assert committed._coef is not None and committed._mean is not None and committed._std is not None  # noqa: S101
            np.testing.assert_allclose(model._coef, committed._coef, atol=_CORPUS_IDENTITY_ATOL)
            np.testing.assert_allclose([model._intercept], [committed._intercept], atol=_CORPUS_IDENTITY_ATOL)
            np.testing.assert_allclose(model._mean, committed._mean, atol=_CORPUS_IDENTITY_ATOL)
            np.testing.assert_allclose(model._std, committed._std, atol=_CORPUS_IDENTITY_ATOL)
            served = committed
        else:
            _assert_retrain_moved_predictions(args, committed, model, X_all, feats)
            served = model  # a retrain SHIPS the fresh fit
        served.shipped_variant = "skillcorner"
        served.provider_list = ["skillcorner"]
        served._type_serve_mode, served._type_gate_metrics = sm, gm
        served.save(_SKILLCORNER_WEIGHTS_DIR)
        reloaded = GkCompletionModel.load(_SKILLCORNER_WEIGHTS_DIR)
        np.testing.assert_allclose(served.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)
        bundled = True
        print(f"SAVED skillcorner weights -> {_SKILLCORNER_WEIGHTS_DIR}", flush=True)
    else:
        print(
            "Not bundling distinct SkillCorner weights; from_variant('skillcorner') resolves to the "
            "gs 'default' (alias -- the GS native-completion model transfers, or is the best-available).",
            flush=True,
        )

    metrics = {
        "variant": "skillcorner",
        # ADR-052: the artifact records WHICH CODE produced it. `--allow-dirty` permits a dev
        # run; the flag survives into the artifact rather than living in someone's memory.
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "mode": args.mode,
        "reason": args.reason,
        "superseded_coef": _superseded_coef(committed_before, feats),
        "decision": decision,
        "bundled": bundled,
        "n_rows": len(df),
        "n_matches": len(np.unique(groups)),
        "subdomains": rep,
        "gkpass_auc_floor": _GKPASS_AUC_FLOOR,
        "ece_tol": _ECE_TOL,
        "slope_tol": _SLOPE_TOL,
    }
    out_dir = _SKILLCORNER_WEIGHTS_DIR if bundled else _WEIGHTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "metrics.json" if bundled else "skillcorner_remeasurement.json"
    (out_dir / fname).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / fname}\nDONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["gradientsports"])
    ap.add_argument("--max-per-provider", type=int, default=64)
    ap.add_argument(
        "--tracking-limit",
        type=int,
        default=None,
        help="frames loaded per match; default None = FULL match (SK-91). The GK-completion model "
        "REQUIRES full frames to reproduce the bundled weights: a small cap (the old generic 200) "
        "starves the SkillCorner derived-GK -> it over-flags GKs in some matches and inflates the "
        "frame-derived GK-pass domain, and collapses the GS density feature. Pass a small int only "
        "for a quick dev smoke, never to (re-)bundle.",
    )
    ap.add_argument("--variant", default="default", choices=["default", "skillcorner"])
    ap.add_argument(
        "--mode",
        required=True,
        choices=["rebundle", "retrain"],
        help="REQUIRED, no default. `rebundle` re-attaches fresh gate metadata to the COMMITTED "
        "weights and asserts the fresh fit still reproduces them. `retrain` ships the fresh fit and "
        "asserts the SERVED PREDICTIONS moved -- a retrain that reproduces the old behaviour means "
        "the input change never reached the model, and shipping it as 'retrained on X' is a false "
        "claim. Neither is reachable by accident.",
    )
    ap.add_argument(
        "--reason",
        required=True,
        help="REQUIRED. Why this run is bundling -- recorded verbatim in metrics.json. A weights "
        "change with no stated cause is unreviewable six months later.",
    )
    ap.add_argument(
        "--feature-space",
        choices=["unchanged", "moved"],
        default=None,
        help="REQUIRED with --mode retrain. `unchanged` = more data or new hyperparameters, so both "
        "models are validly served on the same design matrix. `moved` = a geometry/coordinate "
        "correction changed the raw features, so the committed model must be served on the "
        "PRE-change matrix or the comparison is meaningless. There is no safe default: the two "
        "choices need opposite instruments.",
    )
    ap.add_argument(
        "--probe-old",
        default=None,
        help="parquet of the pre-change design matrix; required for --feature-space moved.",
    )
    ap.add_argument("--cache-features", default=None, help="parquet path to cache/reuse extracted features (owner-run)")
    ap.add_argument(
        "--shard-dir",
        default="gk_completion_shards",
        help=(
            "dir for the resumable per-match extraction shards. This driver writes its weights to a "
            "hardcoded path and has no --output-dir, so the shards need a home of their own; a "
            "capped --tracking-limit run lands in a different generation (see _extract)."
        ),
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Train from a modified working tree. The run still records run_tree_dirty=true in "
        "metrics.json -- the hatch permits a dev run, it never launders the fact.",
    )
    args = ap.parse_args()
    _validate_bundling_args(args)

    # FIRST, before any corpus work. This trainer writes BUNDLED weights, and an artifact whose
    # provenance is unknown is one nobody can reproduce or audit later. ADR-052 enrolled all five
    # trainers at once, deliberately: a partial roll-out is how the same rule failed twice before.
    from scripts._provenance import git_provenance, require_clean_tree

    run_prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    if args.variant == "skillcorner":
        return _train_skillcorner(args)

    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    print(f"=== extracting GK distributions ({args.providers}) ===", flush=True)
    df = _extract(
        args.providers, args.max_per_provider, args.tracking_limit, shard_root=Path(args.shard_dir) / "default"
    )
    feats = GK_COMPLETION_FEATURE_NAMES
    X_all = df[feats]
    y_all = df["_y"].to_numpy(int)
    groups = df["_group"].to_numpy()
    native = df["origin_source"].to_numpy() == "native"
    print(
        f"N={len(df)}  base_rate={y_all.mean():.3f}  native={native.sum()} ({native.mean():.0%})  "
        f"matches={len(np.unique(groups))}",
        flush=True,
    )

    # ---- out-of-fold predictions ----
    oof = np.full(len(df), np.nan)
    n_splits = max(2, min(5, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits).split(X_all, y_all, groups):
        if len(np.unique(y_all[tr])) < 2:
            continue
        m = GkCompletionModel().fit(X_all.iloc[tr], pd.Series(y_all[tr]))
        oof[te] = m.predict_proba(X_all.iloc[te])
    ok = np.isfinite(oof)

    # ---- GREEN GATE: native-origin pooled calibration (review #1) ----
    nat = native & ok
    n_native = int(nat.sum())
    yb, pb = y_all[nat], oof[nat]
    base_brier = brier_score_loss(yb, np.full(n_native, yb.mean()))
    native_auc = roc_auc_score(yb, pb)
    ci_lo, ci_hi = _bootstrap_auc_ci(yb, pb)
    native_brier = brier_score_loss(yb, pb)
    density_finite = float(np.isfinite(df["dest_defender_density"].to_numpy()).mean())
    label_split = {"fail": int((y_all == 0).sum()), "success": int((y_all == 1).sum())}

    print("\n=== native-origin pooled out-of-fold gate ===", flush=True)
    print(
        f"  n_native={n_native}  AUC={native_auc:.4f}  CI95=[{ci_lo:.4f},{ci_hi:.4f}]  "
        f"Brier={native_brier:.4f} (base {base_brier:.4f})",
        flush=True,
    )
    print(f"  density_finite_rate={density_finite:.0%}  label_split={label_split}", flush=True)

    gate = {
        "n_native_ge_floor": n_native >= _N_NATIVE_FLOOR,
        "auc_ci_lower_gt_chance": ci_lo > 0.5,
        "brier_lt_base_rate": native_brier < base_brier,
    }
    print(f"  gate: {gate}", flush=True)
    if not all(gate.values()):
        print("GATE FAILED -- not shipping.", flush=True)
        return 1

    # ---- final fit on ALL kept rows (native + imputed) ----
    # `model` = corpus-identity PROBE; the SERVED artifact is the committed model + the OOF gate, so
    # coef stay byte-identical and the gate provably describes the served model (spec v3 §5).
    model = GkCompletionModel().fit(X_all, pd.Series(y_all))
    sm, gm = _per_type_gate_from_oof(oof, y_all, X_all)
    try:
        committed = GkCompletionModel.load(_WEIGHTS_DIR)
    except FileNotFoundError:
        committed = None
    if committed is None:
        served = model  # first-ever bundle: nothing committed to preserve
    elif args.mode == "rebundle":
        # KEPT on parameters, deliberately -- see the block comment on _CORPUS_IDENTITY_ATOL.
        # fit()/load() populate the coef arrays above; narrow off Optional for the type checker.
        assert model._coef is not None and model._mean is not None and model._std is not None  # noqa: S101
        assert committed._coef is not None and committed._mean is not None and committed._std is not None  # noqa: S101
        np.testing.assert_allclose(model._coef, committed._coef, atol=_CORPUS_IDENTITY_ATOL)
        np.testing.assert_allclose([model._intercept], [committed._intercept], atol=_CORPUS_IDENTITY_ATOL)
        np.testing.assert_allclose(model._mean, committed._mean, atol=_CORPUS_IDENTITY_ATOL)
        np.testing.assert_allclose(model._std, committed._std, atol=_CORPUS_IDENTITY_ATOL)
        served = committed
    else:
        # A retrain SHIPS the fresh fit, and must prove the fresh fit BEHAVES differently.
        _assert_retrain_moved_predictions(args, committed, model, X_all, feats)
        served = model
    served.shipped_variant = "default"
    served.provider_list = list(args.providers)
    served._type_serve_mode, served._type_gate_metrics = sm, gm
    served.save(_WEIGHTS_DIR)
    reloaded = GkCompletionModel.load(_WEIGHTS_DIR)
    np.testing.assert_allclose(served.predict_proba(X_all), reloaded.predict_proba(X_all), atol=1e-9)

    metrics = {
        "mode": args.mode,
        "reason": args.reason,
        "superseded_coef": _superseded_coef(committed, feats),
        "n_rows": len(df),
        "n_native": n_native,
        "base_rate": float(y_all.mean()),
        "native_auc": native_auc,
        "native_auc_ci95": [ci_lo, ci_hi],
        "native_brier": native_brier,
        "base_rate_brier": base_brier,
        "density_finite_rate": density_finite,
        "label_split": label_split,
        "providers": list(args.providers),
        # ADR-052: the artifact records WHICH CODE produced it. `--allow-dirty` permits a dev
        # run; the flag survives into the artifact rather than living in someone's memory.
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "coef": dict(zip(feats, model._coef.tolist(), strict=True)),  # type: ignore[reportOptionalMemberAccess]
    }
    (_WEIGHTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSAVED bundled default -> {_WEIGHTS_DIR}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
