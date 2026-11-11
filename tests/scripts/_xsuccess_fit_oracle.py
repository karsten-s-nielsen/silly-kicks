"""The oracle of the xsuccess `fit` stage (HFU fit-units spec 3.3, A-R3).

`oracle_cross_val_metrics` and `oracle_fit_work` are `scripts/train_xsuccess.py`'s `cross_val_metrics` and
`_fit_work` as they were at `bb0c601c` (4.140.0), copied VERBATIM, with two changes only: the two function names,
and `XSuccessModel` is bound to `OracleXSuccessModel` (the 4.140.0 model code). `oracle_fit_work` takes its serial
branch here (a test context has no InnerPool). The new `fit` stage (36 units and a reduce) must give the same
bundle bytes. Do not edit the bodies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import _open_data_stages as od
from scripts._stages import FileSet
from scripts.train_xsuccess import _cv_splits, _oof_metrics, render_model_card


def _fit_map(*_a, **_k) -> tuple[dict, dict, str]:  # the old InnerPool path: never reached without a pool
    raise AssertionError("the oracle runs the serial branch only")


def oracle_cross_val_metrics(pooled: pd.DataFrame, *, params: dict | None, family: str = "xgboost") -> dict:
    """GroupKFold-by-match out-of-fold AUC / Brier / calibration-in-the-large + per-type reliability."""
    from tests.xsuccess._fit_oracle import OracleXSuccessModel as XSuccessModel

    y, splits = _cv_splits(pooled)
    oof = np.full(len(y), np.nan)
    if not splits:
        oof = XSuccessModel().fit(pooled, family=family, params=params).predict_success(pooled)
    else:
        import time

        from scripts._study_shared import print_fold

        t0 = time.monotonic()
        for k, (tr, te) in enumerate(splits, start=1):
            m = XSuccessModel().fit(pooled.iloc[tr], family=family, params=params)
            oof[te] = m.predict_success(pooled.iloc[te])
            print_fold("xsuccess", k, len(splits), t0)
    return _oof_metrics(pooled, y, oof)


def oracle_fit_work(key, ctx):
    """The out-of-fold metrics at the best parameters, the pooled fit and the bundle (model, card, metrics)."""
    import tempfile

    from scripts._corpus_identity import open_data_identity
    from tests.xsuccess._fit_oracle import OracleXSuccessModel as XSuccessModel

    pooled, family = ctx.upstream("extract"), ctx.params["family"]
    if not len(pooled):
        raise SystemExit("no on-ball rows collected from the corpus; nothing to fit")
    best = ctx.upstream("hpo")["best_params"]
    # The folds and the final fit on the InnerPool (HFU spec 16.2); a width-1 (escalated) job runs the serial
    # composition in-process, as the hpo stage does -- the same result either way.
    if ctx.pool is not None and ctx.pool.workers > 1 and _cv_splits(pooled)[1]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            metrics, files, feature_set = _fit_map(ctx.pool, pooled, family=family, params=best, work=Path(tmp))
    else:
        metrics = oracle_cross_val_metrics(pooled, params=best, family=family)
        model = XSuccessModel().fit(pooled, family=family, params=best)
        feature_set = model.feature_set
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            files = {f.name: f.read_bytes() for f in sorted(Path(tmp).iterdir()) if f.is_file()}
    # The bundled metadata.json carries the training provenance (spec S5.5, ADR-052/056); it is not in SHA256SUMS.
    meta = json.loads(files["metadata.json"])
    meta |= {"training_commit": ctx.prov["commit"], "run_tree_dirty": ctx.prov["dirty"]}
    files["metadata.json"] = (json.dumps(meta, indent=2) + "\n").encode("utf-8")
    refs = ctx.upstream("refs")
    out = {
        **metrics,
        "n_rows": len(pooled),
        "n_matches": int(pooled["game_id"].nunique()),
        "providers": "statsbomb-open " + ", ".join(f"(comp {c}, season {s})" for c, s in refs["competitions"]),
        "hpo_trials": ctx.params["hpo_trials"],
        **open_data_identity([r["match_id"] for r in refs["refs"]]),  # verify compares it with the lock (14.3)
        "hpo_best_params": best,
        "feature_set": feature_set,
        "training_commit": ctx.prov["commit"],
        "run_tree_dirty": ctx.prov["dirty"],
        "run_tree_state": ctx.prov.get("tree_state"),
        **od.pass_fields(ctx),
    }
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    files["metrics.json"] = json.dumps(out, indent=2, default=str).encode("utf-8")
    print(json.dumps(out, indent=2, default=str))
    return FileSet(files)
