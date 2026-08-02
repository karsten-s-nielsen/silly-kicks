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


def _corpus_taxonomy(providers: list[str], max_per_provider: int | None) -> tuple[str, bool]:
    """The ADR-038 label for the artifact this run will write, plus its all-public verdict.

    Keyed on the manifest ``visibility`` field, **never** the provider name: the 98 owner-tier
    SkillCorner matches carry provider ``skillcorner`` and are non-redistributable, so a
    provider-name allowlist would label a restricted run ``public`` -- the exact defect ADR-038
    deleted ``_PUBLIC_PROVIDERS`` to prevent.

    Derived from the REQUESTED corpus (``select_match_ids``), not the extracted one, for the same
    reason that helper gives: ``load_matches`` may drop a match at runtime. It also makes the label
    identical on a fresh extraction and on a ``--cache-features`` resume, which never sees the keys.

    FAIL-CLOSED on an empty corpus: ``ndarray.all()`` is vacuously True on an empty array, which
    would let a zero-match run claim ``public``.
    """
    sys.path.insert(0, "scripts")
    from _corpus import artifact_label, assert_public_corpus, is_public_row
    from _loader_pining import match_visibility, select_match_ids

    pairs = select_match_ids(providers=providers, max_per_provider=max_per_provider)
    vis = match_visibility(providers)
    # Subset gate, unconditional: nothing unregistered may call itself public (a LICENSING failure).
    assert_public_corpus(vis)
    all_public = bool(
        len(pairs)
        and is_public_row(
            providers=np.asarray([p for p, _ in pairs]),
            match_ids=np.asarray([m for _, m in pairs]),
            visibility=vis,
        ).all()
    )
    return artifact_label(providers=set(providers), all_public=all_public), all_public


def _extract(providers, max_per_provider, tracking_limit, *, shard_root, cache_dir=None):
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
        load_matches(
            providers=providers,
            max_per_provider=max_per_provider,
            tracking_limit=tracking_limit,
            cache_dir=cache_dir,
        ),
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

    TWO probes, not one. ``probe_old`` carries the PRE-change feature space and ``probe_new`` the
    post-change one; they are the SAME array whenever the feature space did not move, which is the
    ordinary case. A single shared probe was drafted first and MEASURED unable to express the
    property -- it asks whether two functions agree on one input, when the question is whether the
    model behaves the same on the data each version actually sees.

    ``probe_old`` is NOT "the matrix the committed weights were fit on" in the archaeological sense,
    and reading it that way cost this cycle ~50 minutes of corpus compute. The comparison below is
    ELEMENT-WISE, so the two probes must be ROW-ALIGNED. A historical training matrix from the
    original fit (1666 rows) against the current corpus (~3491) raises
    ``ValueError: operands could not be broadcast together`` -- but do NOT rely on that: it is an
    accident of those particular numbers. A **1-row probe BROADCASTS** against any corpus and returns
    a verdict SILENTLY (measured). `_assert_retrain_moved_predictions` therefore compares row counts
    explicitly before calling this. The correct probe is the SAME corpus as the fresh fit, extracted
    under the pre-change geometry -- which is also why the sentence above says "the SAME array
    whenever the feature space did not move": only a same-corpus extraction can ever be that array.
    The question is about
    SERVING (does what production emits change), not about fitting provenance.

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

    ``--feature-space moved`` refuses only when ``--probe-old`` is ABSENT (the weights directory
    stores no design matrix, so it cannot be reconstructed from the artifact). An earlier draft said
    this ALWAYS refuses, which was already wrong when written. A loud refusal naming why the
    comparison is impossible beats silently answering the wrong question, which is what
    ``probe_old = probe_new = X_all`` would do.

    NOT CLAIMED HERE: how the shipped 4.73.0 bundles were produced. `feature_space` and `probe_old`
    are recorded in `metrics.json` from this release ONWARD, but both bundles were produced at
    4b15365, before that recording landed, so neither artifact can attest to its own invocation. The
    corpus and reason are in the two MODEL_CARD.md files; the invocation is not checkable from the
    artifact, and saying so is better than an unverifiable assurance in a docstring.
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
            "moved -- the exact defect it exists to catch. Re-extract the probe with the pre-change "
            "extractor over THIS SAME corpus: it must be row-aligned with this run's design matrix, "
            "which the next guard checks. (ADR-011's follow-up is sometimes paraphrased as "
            "'persist a probe SAMPLE beside the weights' -- do not read it that way here: a sample "
            "has the wrong row count, and a 1-row one would broadcast and answer silently.)"
        )


def _assert_retrain_moved_predictions(args, committed, model, X_all, feats) -> None:
    """`--mode retrain`: the fresh fit must SERVE differently from the committed weights.

    ROW ALIGNMENT IS CHECKED HERE, EXPLICITLY, and the reason is a measured near-miss. The comparison
    inside `predictions_moved` is element-wise, and it was believed that a mismatched probe therefore
    "raises rather than answering the wrong question". That is true of the shape this cycle actually
    hit (1666 historical rows against a 3491-row corpus -> ValueError) and FALSE in general: numpy
    BROADCASTS a 1-row probe against any corpus and returns a verdict silently. ADR-052's own
    follow-up says to "persist a fixed probe SAMPLE beside the weights" -- and a sample of one row is
    exactly the shape that slips through. Relying on the broadcast error was relying on an accident
    of the numbers involved.
    """
    if committed is None:
        return  # first-ever bundle: there is nothing to have moved away from
    probe_old = pd.read_parquet(args.probe_old)[feats] if args.probe_old else X_all
    if len(probe_old) != len(X_all):
        raise SystemExit(
            f"--probe-old has {len(probe_old)} rows but this run's design matrix has {len(X_all)}. "
            "The two probes are compared ELEMENT-WISE, so they must describe the SAME rows: "
            "probe_old is this corpus under the PRE-change geometry, not a historical training "
            "matrix and not a sample. A broadcastable mismatch (notably a 1-row probe) would not "
            "even raise -- it would answer, silently and wrongly."
        )
    if not predictions_moved(_as_weights(committed), _as_weights(model), probe_old=probe_old, probe_new=X_all):
        raise SystemExit(
            "RETRAIN produced the committed model's served predictions unchanged -- the input "
            "change never reached the model. Shipping this as a retrain would be a false claim. "
            f"(reason given: {args.reason!r})"
        )


def _assert_rebundle_reproduces(model, committed) -> None:
    """`--mode rebundle`: the fresh fit must reproduce the committed parameters.

    Kept on PARAMETERS, deliberately -- see the block comment on `_CORPUS_IDENTITY_ATOL`.

    This exists as a named function for one reason: the bare `np.testing.assert_allclose` it replaces
    raised an `AssertionError` carrying a float diff and no remedy, at the end of a corpus pass that
    costs half an hour. The drift it reports is nearly always the SAME situation -- the feature space
    moved under a re-bundle -- and the correct response is a specific command, so the failure says so
    rather than leaving the operator to rediscover it. Both call sites were byte-identical, so this
    also stops them drifting apart.

    TOLERANCE SEMANTICS DIFFER FROM THE ORIGINAL, DELIBERATELY. `assert_allclose` applies
    `atol + rtol * |desired|` with a default `rtol=1e-7`; this compares max-absolute-drift against
    `_CORPUS_IDENTITY_ATOL` alone. At the scales involved that is a distinction without a difference
    -- the coefficients are O(1), so the relative term contributes ~1e-7 against a 0.05 floor -- and
    an absolute-only rule is what the block comment on `_CORPUS_IDENTITY_ATOL` actually describes.
    Recorded because dropping a default silently is how a tolerance stops meaning what its comment
    says.

    TWO of `assert_allclose`'s defaults are dropped, not one. The second is `equal_nan=True`, and it
    is a SEMANTIC divergence rather than a numeric one: the original ACCEPTED a NaN -- or a
    same-signed inf -- at the SAME index on BOTH sides, measured on all four parameters. That is
    exactly the case the non-finite block below aborts on, so on that side this function is STRICTER
    than what it replaced.
    """
    # fit()/load() populate the coef arrays; narrow off Optional for the type checker.
    assert model._coef is not None and model._mean is not None and model._std is not None  # noqa: S101
    assert committed._coef is not None and committed._mean is not None and committed._std is not None  # noqa: S101

    # SHAPE FIRST. `assert_allclose` reports a shape mismatch as a readable assertion; a raw
    # subtraction raises `ValueError: operands could not be broadcast together` out of the middle of
    # a corpus pass, naming numpy rather than the feature-count change that actually happened.
    pairs = (
        ("coef", np.asarray(model._coef), np.asarray(committed._coef)),
        ("mean", np.asarray(model._mean), np.asarray(committed._mean)),
        ("std", np.asarray(model._std), np.asarray(committed._std)),
    )
    for name, fresh, old in pairs:
        if fresh.size == 0 or old.size == 0:
            # (0,) == (0,) passes the shape check below and then `np.max` on an empty array raises a
            # bare numpy ValueError naming neither the model nor the parameter. A degenerate artifact
            # is not "reproduces the committed model" either, so abort with the reason.
            raise SystemExit(
                f"REBUNDLE aborted: `{name}` is EMPTY (fresh {fresh.shape}, committed {old.shape}). "
                "One of these artifacts has no parameters -- a degenerate fit or a truncated file. "
                "Investigate before shipping anything."
            )
        if fresh.shape != old.shape:
            raise SystemExit(
                f"REBUNDLE aborted: `{name}` has shape {fresh.shape} but the committed weights have "
                f"{old.shape}. The FEATURE SET changed, which no tolerance can reconcile -- the two "
                "models do not describe the same input space. A re-bundle is meaningless here; "
                "re-fit and ship the fresh model."
            )

    drift = {name: float(np.max(np.abs(fresh - old))) for name, fresh, old in pairs}
    # `intercept` is a scalar, so it has no shape to check -- but it DOES need the same None guard the
    # three arrays get above, or `abs(None - x)` raises a TypeError instead of naming the problem.
    if model._intercept is None or committed._intercept is None:
        raise SystemExit(
            "REBUNDLE aborted: `intercept` is None on "
            f"{'the fresh fit' if model._intercept is None else 'the committed weights'} -- the "
            "artifact is incomplete, so 'reproduces the committed model' cannot be answered."
        )
    drift["intercept"] = float(abs(model._intercept - committed._intercept))

    # NON-FINITE DRIFT ABORTS UNCONDITIONALLY: part regression guard, part STRENGTHENING.
    #
    # REGRESSION half -- the `max(...)`/`<=` form below is order-dependent under NaN: every comparison
    # against NaN is False, so `max` keeps whichever key it happened to be holding and a NaN sitting
    # anywhere but first is silently discarded. MEASURED: a ONE-SIDED NaN (fresh fit NaN against
    # finite committed weights) in `intercept`, `mean` or `std` was ACCEPTED here, while the
    # `np.testing.assert_allclose` calls this replaced rejected it in all four positions.
    #
    # STRENGTHENING half -- that comparison holds only one-sided, and an earlier draft of this comment
    # overstated it. `assert_allclose` defaults to `equal_nan=True` and treats matched same-signed
    # infs as equal, so a NaN/inf at the SAME index on BOTH sides -- precisely what the abort message
    # below names, a committed artifact carrying NaN met by a fresh fit that reproduces it -- was
    # ACCEPTED by the original in all four positions too. MEASURED both directions. So this function
    # is STRICTER than what it replaced on that side, not equal to it; do not "restore parity" with
    # `assert_allclose` on the way past. NaN weights mean a degenerate fit, which must never be waved
    # through as "reproduces the committed model", whichever side carries them.
    nonfinite = sorted(k for k, v in drift.items() if not np.isfinite(v))
    if nonfinite:
        raise SystemExit(
            f"REBUNDLE aborted: non-finite drift in {nonfinite}. Either the fresh fit or the "
            "committed weights contain NaN/inf, so 'reproduces the committed model' is not a "
            "question that can be answered. Investigate the fit before shipping anything."
        )

    worst = max(drift, key=lambda k: drift[k])
    if drift[worst] <= _CORPUS_IDENTITY_ATOL:
        return

    raise SystemExit(
        "REBUNDLE aborted: the fresh fit does not reproduce the committed weights.\n"
        + "".join(f"  max |{k}| drift = {v:.6g}\n" for k, v in drift.items())
        + f"  tolerance = {_CORPUS_IDENTITY_ATOL} (exceeded by {worst!r})\n"
        "\n"
        "A re-bundle SHIPS THE COMMITTED WEIGHTS, so production would then serve them against "
        "whatever features the current extractor produces. If the extractor moved -- a geometry or "
        "coordinate correction, a provider fix, a corpus change -- that is a train/serve skew, and "
        "re-bundling would hide it behind a byte-identical model.json.\n"
        "\n"
        "If the feature space MOVED, the right action is a retrain, which ships the fresh fit:\n"
        "  --mode retrain --feature-space moved --probe-old <pre-change design matrix>.parquet "
        '--reason "<what moved>"\n'
        "The probe is the SAME corpus as this run, extracted at the commit just BEFORE the change "
        "under test -- not the vintage the committed weights were originally fit on, and not a "
        "sample. The two probes are compared element-wise, so they must be ROW-ALIGNED; the trainer "
        "checks that explicitly, because numpy does not always: a 1-row probe BROADCASTS against any "
        "corpus and answers silently rather than raising.\n"
        "\n"
        "If the feature space did NOT move, this drift is real corpus drift and needs investigating "
        "before anything is shipped -- do not widen the tolerance to make it pass."
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
            ["skillcorner"],
            args.max_per_provider,
            args.tracking_limit,
            shard_root=Path(args.shard_dir) / "skillcorner",
            cache_dir=args.cache_dir,
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
        # coef stay byte-identical (spec v3 S5 + the additive-only re-bundle check). Re-fit is NEVER
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
            _assert_rebundle_reproduces(model, committed)
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

    sc_label, sc_all_public = _corpus_taxonomy(["skillcorner"], args.max_per_provider)
    metrics = {
        "variant": "skillcorner",
        # ADR-038: the tier this artifact was fit on, derived from the manifest ship mask rather
        # than assigned by hand. Previously absent from this trainer entirely, so a defaulted
        # --max-per-provider 64 run could pull 54 restricted SkillCorner matches into a
        # distributable artifact with nothing refusing it or labelling the result.
        "artifact_label": sc_label,
        "all_public": sc_all_public,
        # ADR-052: the artifact records WHICH CODE produced it. `--allow-dirty` permits a dev
        # run; the flag survives into the artifact rather than living in someone's memory.
        "run_commit": run_prov["commit"],
        "run_tree_dirty": run_prov["dirty"],
        "run_tree_state": run_prov["tree_state"],
        "mode": args.mode,
        "reason": args.reason,
        # CORPUS BOUNDS, recorded IN the artifact. An unrecorded cap is indistinguishable from a
        # full run and silently biases every number beside it -- the failure this cycle hit on the
        # ADR-028 RC4 measurement, where `tracking_limit=3000` went unrecorded and halved a headline
        # figure. `--feature-space` and `--probe-old` ride along because both are load-bearing for
        # the retrain verdict: `moved` without a probe is refused, and the probe decides what
        # `predictions_moved` actually compared. Only the probe's BASENAME is stored -- an absolute
        # home-directory path is noise in a committed artifact.
        "max_per_provider": args.max_per_provider,
        "tracking_limit": args.tracking_limit,
        "feature_space": args.feature_space,
        "probe_old": Path(args.probe_old).name if args.probe_old else None,
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
        "--cache-dir",
        default=None,
        help="Persist downloaded pining artifacts under CACHE_DIR/{provider}/{match_id}/ and reuse "
        "them on later runs over the same corpus. Default None re-downloads every run (~24-90 s per "
        "match). Deliberately NOT part of the shard token_inputs: it caches DOWNLOADS, not extracted "
        "features, so it cannot change a shard's content.",
    )
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
        args.providers,
        args.max_per_provider,
        args.tracking_limit,
        shard_root=Path(args.shard_dir) / "default",
        cache_dir=args.cache_dir,
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
    # coef stay byte-identical and the gate provably describes the served model (spec v3 S5).
    model = GkCompletionModel().fit(X_all, pd.Series(y_all))
    sm, gm = _per_type_gate_from_oof(oof, y_all, X_all)
    try:
        committed = GkCompletionModel.load(_WEIGHTS_DIR)
    except FileNotFoundError:
        committed = None
    if committed is None:
        served = model  # first-ever bundle: nothing committed to preserve
    elif args.mode == "rebundle":
        _assert_rebundle_reproduces(model, committed)
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

    default_label, default_all_public = _corpus_taxonomy(list(args.providers), args.max_per_provider)
    metrics = {
        "mode": args.mode,
        "reason": args.reason,
        # CORPUS BOUNDS, recorded IN the artifact. An unrecorded cap is indistinguishable from a
        # full run and silently biases every number beside it -- the failure this cycle hit on the
        # ADR-028 RC4 measurement, where `tracking_limit=3000` went unrecorded and halved a headline
        # figure. `--feature-space` and `--probe-old` ride along because both are load-bearing for
        # the retrain verdict: `moved` without a probe is refused, and the probe decides what
        # `predictions_moved` actually compared. Only the probe's BASENAME is stored -- an absolute
        # home-directory path is noise in a committed artifact.
        "max_per_provider": args.max_per_provider,
        "tracking_limit": args.tracking_limit,
        "feature_space": args.feature_space,
        "probe_old": Path(args.probe_old).name if args.probe_old else None,
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
        # ADR-038: the tier this artifact was fit on, derived from the manifest ship mask rather
        # than assigned by hand. For the default (Gradient Sports) corpus this is "full" -- the
        # most restricted tier -- where previously no label was recorded at all. Owner decision
        # 2026-08-02: ship it. Those coefficients already ship, so the label documents an existing
        # situation rather than changing what is distributed.
        "artifact_label": default_label,
        "all_public": default_all_public,
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
