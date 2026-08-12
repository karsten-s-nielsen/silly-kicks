"""Stage-2 augmented-VAEP held-out Brier objective (ruthless CachedObjective; spec §4).

patch_params = {k3, pre_seconds, min_displacement_m}. The expensive enrichment is invariant across
these (prepared once per match); only link_zones pressure + off-ball runs re-run per trial. xT is a
frozen exogenous artifact (no leak). XGBoost is pinned deterministic so the fast path equals the
full recompute to 1e-9 (assert_cache_equivalence).

Examples
--------
>>> from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective
>>> # obj = AugmentedVaepBrierObjective(fold=fold, xt=frozen.xt,
>>> #     carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ruthless import Direction, penalty_metrics
from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._cv import cv_standard_error, match_cv_splits
from silly_kicks.calibration._features import (
    ALL_FEATURES,
    enrich_full,
    enrich_invariant,
    patch_trial_columns,
)
from silly_kicks.calibration._gates import (
    PENALTY_K,
    default_feature_variances,
    h1_penalty_fires,
    signal_sanity,
)
from silly_kicks.tracking import DAS_SOURCE_UNSCOREABLE_CALL

if TYPE_CHECKING:
    from silly_kicks.calibration._xt import FrozenXt

_DEFAULT_PARAMS = {"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0}


def _vaep_labels(actions: pd.DataFrame) -> pd.DataFrame:
    """scores + concedes labels (10-action window), aligned 1:1 with actions."""
    from silly_kicks.spadl import add_names
    from silly_kicks.vaep.labels import concedes, scores

    named = add_names(actions)
    out = pd.DataFrame(index=actions.index)
    out["scores"] = scores(named, nr_actions=10)["scores"].to_numpy()
    out["concedes"] = concedes(named, nr_actions=10)["concedes"].to_numpy()
    return out


def _xgb_classifier(seed: int):
    """Fully-pinned deterministic XGBoost (C1/L3): the 1e-9 cache gate rides on this.

    Fixed seed + single thread + hist + subsample/colsample EXPLICITLY 1.0 (defend against a
    future default change) => identical features give identical Brier.
    """
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        tree_method="hist",
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


def _provider_cv(
    X: pd.DataFrame, y_scores: np.ndarray, y_concedes: np.ndarray, mids: np.ndarray, seed: int
) -> tuple[float | None, float]:
    """Per-provider CV: per-fold Brier = mean(scores_brier, concedes_brier); return (mean, SE).

    M4: splits are computed ONCE; a fold is skipped for BOTH labels if EITHER label is single-class
    in train — so the two labels stay fold-aligned (no zip-misalignment). NaN is passed through to
    XGBoost (M3 — never fillna(0); deterministic under n_jobs=1). Returns (None, nan) if no usable
    fold (caller treats as no-signal).
    """
    from sklearn.metrics import brier_score_loss

    fold_briers: list[float] = []
    for train_idx, test_idx in match_cv_splits(mids):
        if len(np.unique(y_scores[train_idx])) < 2 or len(np.unique(y_concedes[train_idx])) < 2:
            continue  # skip this fold for BOTH labels (keeps them aligned)
        x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
        per_label = []
        for y in (y_scores, y_concedes):
            model = _xgb_classifier(seed)
            model.fit(x_tr, y[train_idx])  # NaN passthrough (no fillna)
            probs = model.predict_proba(x_te)[:, 1]
            per_label.append(float(brier_score_loss(y[test_idx], probs)))
        fold_briers.append(float(np.mean(per_label)))
    if not fold_briers:
        return None, float("nan")
    return float(np.mean(fold_briers)), cv_standard_error(fold_briers)


class _Invariant:
    """Prepared invariant: per-(provider, match) base actions/links/labels + penalty anchors."""

    def __init__(self) -> None:
        self.bases: dict[str, list[dict]] = {}  # provider -> [{frames, raw_actions, base, links, labels, ...}]
        self.kept_providers: list[str] = []  # signal-sanity survivors (R7: fixed for the study)
        self.default_variances: dict[str, float] = {}
        self.default_brier: float = 0.25  # fallback before computed
        self.das_degraded: dict[str, int] = {}  # provider -> n matches with degraded DAS (M8)


class AugmentedVaepBrierObjective:
    """ruthless ``CachedObjective`` — minimize equal-provider-weight held-out Brier.

    Examples
    --------
    Build the Stage-2 augmented-VAEP Brier objective for a fold (driven by a ruthless strategy,
    e.g. ``from ruthless.strategies.optuna_ import OptunaStrategy``)::

        from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective

        obj = AugmentedVaepBrierObjective(
            fold=fold, xt=frozen.xt,
            carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42,
        )
    """

    patch_params = frozenset({"k3", "pre_seconds", "min_displacement_m"})

    def __init__(self, *, fold: dict[str, list[tuple]], xt: FrozenXt, carrier_params: dict, seed: int = 42) -> None:
        # `xt` is the frozen calibration artifact (grid + provenance); the inner ExpectedThreat is
        # unwrapped (self._xt.xt) where the feature functions consume it. Taking the FrozenXt here —
        # not the bare grid — keeps the CLI passing ONE object to both this objective and the report
        # manifest, and lets pyright reject a bare ExpectedThreat at the call site.
        self._fold = fold
        self._xt = xt
        self._carrier_params = carrier_params
        self._seed = seed
        self.diagnostics: dict = {}  # surfaced into the manifest (M1/M8)

    # ---- helpers ---------------------------------------------------------
    def _assemble(self, per_match: list[dict]) -> dict:
        """Concat one provider's matches into {X (ALL_FEATURES, NaN kept), y_scores, y_concedes, mids}."""
        x_parts, y_s, y_c, mids = [], [], [], []
        for e in per_match:
            x = e["X"][ALL_FEATURES]  # NaN passthrough (M3) — no fillna(0)
            x_parts.append(x)
            y_s.append(e["labels"]["scores"].to_numpy())
            y_c.append(e["labels"]["concedes"].to_numpy())
            mids.append(np.array([e["match_id"]] * len(x)))
        return {
            "X": pd.concat(x_parts, ignore_index=True),
            "y_scores": np.concatenate(y_s),
            "y_concedes": np.concatenate(y_c),
            "mids": np.concatenate(mids),
        }

    def _score_features(self, per_provider: dict[str, dict], default_variances, default_brier) -> Metrics:
        """Shared scorer: H1 gate + per-provider CV + equal-provider-weight mean.

        ``per_provider`` already contains ONLY the signal-sanity-kept providers (filtered at
        prepare() time, R7) — the equal-weight denominator is fixed for the whole study here.
        """
        if not per_provider:
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)
        trial_x = pd.concat([a["X"] for a in per_provider.values()], ignore_index=True)
        if h1_penalty_fires(trial_x, default_variances):
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)

        provider_brier: dict[str, float] = {}
        provider_se: dict[str, float] = {}
        for provider, a in per_provider.items():
            mean_b, se = _provider_cv(a["X"], a["y_scores"], a["y_concedes"], a["mids"], self._seed)
            if mean_b is not None:
                provider_brier[provider] = mean_b
                provider_se[provider] = se
        if not provider_brier:
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)

        metrics: Metrics = {"brier": float(np.mean(list(provider_brier.values())))}
        for provider, b in provider_brier.items():
            metrics[f"brier__{provider}"] = b
            metrics[f"brier_se__{provider}"] = provider_se[provider]
        return metrics

    # ---- fast path -------------------------------------------------------
    def prepare(self) -> _Invariant:
        """Build the trial-independent invariant once (per-match enrichment + labels + anchors).

        Call this ONCE per fold, then reuse the returned invariant for every trial. It
        carries three things a trial must not recompute and must not disagree about: the
        expensive per-match enrichment, the H1 default-param anchors, and the kept-provider
        set that fixes the equal-weight denominator for the whole study.

        Examples
        --------
        The invariant is built once and threaded through every trial::

            obj = AugmentedVaepBrierObjective(
                fold=fold, xt=frozen.xt, carrier_params=carrier_params, seed=42
            )
            invariant = obj.prepare()
            for candidate in study.candidates():
                metrics = obj.evaluate_patch(invariant, candidate)

        Two side effects are worth knowing about, because both are decisions the harness
        makes FOR the study rather than per trial: a provider with no usable label signal is
        excluded loudly (recorded in ``obj.diagnostics["excluded_providers"]``), and a match
        whose DAS computation degraded is counted into
        ``obj.diagnostics["das_degraded"]`` and warned about.
        """
        inv = _Invariant()
        for provider, matches in self._fold.items():
            inv.bases[provider] = []
            inv.das_degraded[provider] = 0
            for match_idx, (actions, frames, home) in enumerate(matches):
                base, links = enrich_invariant(
                    actions=actions,
                    frames=frames,
                    xt=self._xt.xt,
                    carrier_params=self._carrier_params,
                )
                # M8 via the PUBLIC das_source provenance (ADR-043), not a private flag:
                # DAS_SOURCE_UNSCOREABLE_CALL is stamped on every row when the whole
                # per-match DAS computation degraded.
                if (base["das_source"] == DAS_SOURCE_UNSCOREABLE_CALL).any():
                    inv.das_degraded[provider] += 1
                inv.bases[provider].append(
                    {
                        "frames": frames,
                        "raw_actions": actions,
                        "base": base,
                        "links": links,
                        "labels": _vaep_labels(base),
                        "home": home,
                        "match_id": f"{provider}:{match_idx}",
                    }
                )
        # R7/M1: signal-sanity at PREPARE time (data-determined, FIXED for the whole study). A
        # provider with no usable label signal (no match has both classes in either label) is
        # loudly EXCLUDED, never silently averaged in. The equal-weight denominator is set here.
        signal = {
            p: float(
                sum(1 for e in entries if e["labels"]["scores"].nunique() > 1 or e["labels"]["concedes"].nunique() > 1)
            )
            for p, entries in inv.bases.items()
        }
        kept, excluded = signal_sanity(signal, min_value=1.0)
        inv.kept_providers = kept
        self.diagnostics["excluded_providers"] = excluded
        # Default-param pass: anchors for the H1 penalty (M2: SAME mean(scores,concedes) as scoring).
        default_per_provider = self._build_per_provider(inv, _DEFAULT_PARAMS, use_full=False)
        if default_per_provider:
            all_default_x = pd.concat([a["X"] for a in default_per_provider.values()], ignore_index=True)
            inv.default_variances = default_feature_variances(all_default_x)
            default_briers = []
            for a in default_per_provider.values():
                mean_b, _se = _provider_cv(a["X"], a["y_scores"], a["y_concedes"], a["mids"], self._seed)
                if mean_b is not None:
                    default_briers.append(mean_b)
            if default_briers:
                inv.default_brier = float(np.mean(default_briers))
        # M8: surface DAS degradation (loud + manifest), never silent.
        total_degraded = sum(inv.das_degraded.values())
        if total_degraded:
            warnings.warn(
                f"DAS degraded on {total_degraded} match(es): {inv.das_degraded}",
                UserWarning,
                stacklevel=2,
            )
        self.diagnostics["das_degraded"] = dict(inv.das_degraded)
        return inv

    def _build_per_provider(self, inv: _Invariant, params: dict, *, use_full: bool) -> dict[str, dict]:
        """Assemble per-provider features. use_full=True => enrich_full (independent monolith, H1);
        else patch the cached invariant base."""
        per_provider: dict[str, dict] = {}
        providers = inv.kept_providers or list(inv.bases)  # kept-only (set in prepare before this is called)
        for provider in providers:
            entries = inv.bases[provider]
            per_match = []
            for e in entries:
                if use_full:
                    x_actions = enrich_full(
                        actions=e["raw_actions"],  # ORIGINAL SPADL actions — genuine from-scratch (H1)
                        frames=e["frames"],
                        xt=self._xt.xt,
                        home_team_id=e["home"],
                        carrier_params=self._carrier_params,
                        **params,
                    )
                else:
                    x_actions = patch_trial_columns(
                        base_actions=e["base"],
                        frames=e["frames"],
                        links=e["links"],
                        home_team_id=e["home"],
                        **params,
                    )
                per_match.append({"X": x_actions, "labels": e["labels"], "match_id": e["match_id"]})
            per_provider[provider] = self._assemble(per_match)
        return per_provider

    def evaluate_patch(self, invariant: _Invariant, candidate: Candidate) -> Metrics:
        """Cheap per-trial Brier: patch only the 2 trial-varying steps on the cached invariant.

        The fast path, and the one a sweep actually runs. It re-runs link_zones pressure and
        off-ball runs on the cached base; everything else is reused. The returned metrics
        carry the equal-weight mean under ``"brier"`` plus a per-provider
        ``"brier__<provider>"`` / ``"brier_se__<provider>"`` breakdown, so a mean that moved
        because of one provider is visible rather than averaged away.

        Examples
        --------
        Score a candidate against a prepared invariant::

            invariant = obj.prepare()
            metrics = obj.evaluate_patch(invariant, candidate)
            metrics["brier"]  # equal-weight mean over the KEPT providers

        The result must equal :meth:`evaluate` on the same candidate to 1e-9 — that
        equivalence is what makes the cache sound, and ``assert_cache_equivalence`` checks
        it rather than trusting it. If a trial flattens a tuned feature, the H1 gate returns
        a finite penalty Brier instead: a degenerate trial is steered away, never raised on.
        """
        params = {k: float(candidate.params[k]) for k in ("k3", "pre_seconds", "min_displacement_m")}
        per_provider = self._build_per_provider(invariant, params, use_full=False)  # CACHED base + patch
        return self._score_features(per_provider, invariant.default_variances, invariant.default_brier)

    # ---- full path (Objective port) — INDEPENDENT monolith (H1) ----------
    def evaluate(self, candidate: Candidate) -> Metrics:
        """Full from-scratch Brier via the independent monolith (the Objective port; H1).

        The slow path, and the ORACLE the fast path is checked against. It re-derives every
        feature from the original SPADL actions via ``enrich_full`` — deliberately sharing
        no cached intermediate with :meth:`evaluate_patch`, because a shared intermediate
        would make the two paths agree by construction and the 1e-9 equivalence check would
        prove nothing.

        Examples
        --------
        Verify the cache rather than trusting it::

            fast = obj.evaluate_patch(obj.prepare(), candidate)["brier"]
            slow = obj.evaluate(candidate)["brier"]  # independent recompute
            assert abs(fast - slow) < 1e-9

        Use this for the cache-equivalence gate and for spot checks, not for a sweep: it
        re-runs the whole per-match enrichment for every call.
        """
        invariant = self.prepare()  # anchors + per-match frames/actions/labels (prepare is deterministic)
        params = {k: float(candidate.params[k]) for k in ("k3", "pre_seconds", "min_displacement_m")}
        per_provider = self._build_per_provider(invariant, params, use_full=True)  # enrich_full, no cache
        return self._score_features(per_provider, invariant.default_variances, invariant.default_brier)
