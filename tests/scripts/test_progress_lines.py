"""Progress lines for long single units (HFU spec 14.8, Change 7g; launch-3 finding F18): one line per
finished study trial, a timed census-bootstrap line on both paths, and a reduce end line."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import scripts._stages as S
from scripts._crossed_icc import bootstrap_icc_and_power
from scripts._stages import LocalStore, Param, Pipeline, Stage, int_key, make_context, run_unit
from tests.scripts._stage_harness import PROV


def _study(tmp_path, n_trials: int = 4) -> dict:
    from scripts.train_xshot_occurrence import _hpo_once

    rng = np.random.default_rng(0)
    n = 240
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 0.8).astype(int).to_numpy()
    groups = np.repeat([f"g{i}" for i in range(8)], n // 8)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return _hpo_once(
        X, y, groups, tmp_path, "single", n_trials,
        objective_inputs={"driver": "t", "args": {}, "match_ids": []},
        prov={"commit": "c" * 40, "dirty": False, "tree_state": "clean"},
        negative_subsample=None, seed=42,
    )  # fmt: skip


def test_study_prints_one_line_per_trial(tmp_path, capsys):
    _study(tmp_path / "a", n_trials=4)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] study ")]
    assert len(lines) == 4, lines
    for k, ln in enumerate(lines, start=1):
        assert re.fullmatch(rf"\[progress\] study single trial {k}/4 best=\S+ in \d+\.\d s", ln), ln
    bests = [float(re.search(r"best=(\S+)", ln).group(1)) for ln in lines]  # type: ignore[union-attr]
    assert bests == sorted(bests, reverse=True)  # logloss: the best so far never rises


def test_study_observer_keeps_best_params(tmp_path, monkeypatch):
    """The observer is a sink only (ruthless's documented contract): the best parameters are equal with and
    without it."""
    import scripts._study_shared as SS

    with_observer = _study(tmp_path / "a")
    monkeypatch.setattr(SS, "TrialPrinter", lambda *_a, **_k: None)
    assert _study(tmp_path / "b") == with_observer


def _toy_design(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    d = rng.integers(0, 40, n)
    t = (d // 4 + rng.integers(0, 2, n)) % 12
    y = rng.normal(size=n) + 0.3 * (d % 5)
    return y, d.astype(str), t.astype(str)


def test_bootstrap_progress_both_paths(capsys):
    y, d, t = _toy_design()
    kw = {"n_boot": 30, "alpha": 0.05, "effect_size": 0.05, "rng_seed": 7, "progress_s": 0.0}
    serial = bootstrap_icc_and_power(y, d, t, **kw)
    serial_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] bootstrap ")]

    def rmap(task, *, on_done):  # chunks of 10 replicates, completed out of order
        fn, items = task
        keys = sorted(items)
        out = {}
        for chunk in (keys[20:], keys[:10], keys[10:20]):
            done = {k: fn(*items[k]) for k in chunk}
            out.update(done)
            on_done(done)
        return out

    mapped = bootstrap_icc_and_power(y, d, t, replicate_map=rmap, **kw)
    map_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] bootstrap ")]
    assert mapped == serial
    assert (
        len(serial_lines) == 30
        and serial_lines[-1] == f"[progress] bootstrap 30/30 convergent={serial['n_boot_effective']}"
    )
    assert [ln.split()[2] for ln in map_lines] == ["10/30", "20/30", "30/30"]
    assert map_lines[-1].endswith(f"convergent={serial['n_boot_effective']}")


def test_bootstrap_progress_is_throttled_and_ends_with_one_line(capsys):
    """Throttled lines, then always one final ``n/n`` line, never a repeat of the last throttled one (spec §15.5)."""
    y, d, t = _toy_design()
    res = bootstrap_icc_and_power(y, d, t, n_boot=20, alpha=0.05, effect_size=0.05, rng_seed=7, progress_s=3600.0)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] bootstrap ")]
    assert lines == [f"[progress] bootstrap 20/20 convergent={res['n_boot_effective']}"]


def test_reduce_done_line(tmp_path, capsys):
    pipeline = Pipeline(
        name="reduceline",
        module=__name__,
        params=(Param("n", int, 2),),
        stages=(
            Stage(
                name="s",
                units=lambda ctx: [int_key(i) for i in range(int(ctx.params["n"]))],
                work=lambda key, ctx: pd.DataFrame({"k": [key]}),
                reduce=lambda results, ctx: S.combine_frames(list(results.values()), ids=()),
                params=("n",),
            ),
        ),
        output="s",
    )
    ctx = make_context(pipeline, {"n": 2}, PROV, store=LocalStore(tmp_path), data_root=None)
    for key in (int_key(0), int_key(1)):
        run_unit(ctx, pipeline.stage("s"), key, worker="t")
    capsys.readouterr()
    S.reduce_stage(ctx, "s")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[reduce] ")]
    assert len(lines) == 1 and re.fullmatch(r"\[reduce\] s done in \d+\.\d s", lines[0]), lines


def test_trial_printer_counts_this_call(capsys):
    """A resumed study continues the global trial numbers; the printer counts the events of THIS call."""
    from ruthless import ProgressEvent

    from scripts._study_shared import TrialPrinter

    p = TrialPrinter("t", 3, metric="logloss")
    for number, v in ((7, 0.5), (8, 0.7), (9, 0.4)):
        p(ProgressEvent(number=number, candidate=None, metrics={"logloss": v}, state="complete"))  # type: ignore[arg-type]
    lines = capsys.readouterr().err.splitlines()
    assert [ln.split(" in ")[0] for ln in lines] == [
        "[progress] study t trial 1/3 best=0.5",
        "[progress] study t trial 2/3 best=0.5",
        "[progress] study t trial 3/3 best=0.4",
    ]


def test_trial_printer_optuna_callback_counts_a_failed_trial(capsys):
    """The optuna ``study.optimize(callbacks=...)`` form prints the same line; a failed trial (no value) is
    counted and leaves ``best`` unchanged (launch-4 finding F21)."""
    from types import SimpleNamespace

    from scripts._study_shared import TrialPrinter

    p = TrialPrinter("xsuccess", 3, metric="logloss")
    for v in (0.5, None, 0.4):
        p.optuna_callback(None, SimpleNamespace(value=v))
    assert [ln.split(" in ")[0] for ln in capsys.readouterr().err.splitlines()] == [
        "[progress] study xsuccess trial 1/3 best=0.5",
        "[progress] study xsuccess trial 2/3 best=0.5",
        "[progress] study xsuccess trial 3/3 best=0.4",
    ]


def _xsuccess_pooled(n: int = 400, seed: int = 0) -> pd.DataFrame:
    import silly_kicks.spadl.config as cfg

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 105, n)
    y = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(3.0 - 0.04 * (105.0 - x))))
    return pd.DataFrame(
        {
            "game_id": np.repeat(np.arange(8), n // 8),
            "type_id": cfg.actiontype_id["pass"],
            "bodypart_id": cfg.bodypart_id["foot"],
            "start_x": x,
            "start_y": rng.uniform(0, 68, n),
            "end_x": x,
            "end_y": rng.uniform(0, 68, n),
            "result_id": np.where(y, cfg.result_id["success"], cfg.result_id["fail"]),
            "time_seconds": rng.uniform(0, 3000, n),
            "period_id": 1,
        }
    )


def test_native_fit_stages_print_fold_lines(capsys):
    """HFU spec 15.3: the native pass_completion ``fit`` CV loop prints one fold line per fold. (The xsuccess
    ``fit`` is 36 stage units since the HFU fit-units cycle: each prints its `[unit]` line, below.)"""
    from scripts.train_pass_completion import cross_val_metrics

    cross_val_metrics(_xsuccess_pooled())
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] pass_completion fold ")]
    assert len(lines) == 5, lines
    for k, ln in enumerate(lines, start=1):
        assert re.fullmatch(rf"\[progress\] pass_completion fold {k}/5 in \d+\.\d s", ln), ln


def test_an_xsuccess_fit_unit_prints_its_unit_lines(tmp_path, capsys):
    """HFU fit-units spec 3.2: the `[progress] xsuccess fold` and `[progress] xsuccess fit` lines are gone with
    the serial and the InnerPool fit; a `fit` unit prints the `[unit]` start and done lines of every stage unit."""
    import scripts.train_xsuccess as T
    from scripts import _stages as s
    from tests.scripts._stage_harness import PROV, params_from_args

    ctx = s.make_context(
        T.PIPELINE, params_from_args(T.PIPELINE, []), PROV, store=s.LocalStore(tmp_path), data_root=None
    )
    s.write_result(ctx.store, f"{ctx.generation('extract')}/_output", _xsuccess_pooled(n=120))
    s.write_result(ctx.store, f"{ctx.generation('hpo')}/_output", {"best_params": None})
    capsys.readouterr()
    assert s.run_unit(ctx, T.PIPELINE.stage("fit"), "fold0-cal0", worker="w0") == "done"
    err = capsys.readouterr().err
    assert "[unit] fit/fold0-cal0 start" in err
    assert re.search(r"\[unit\] fit/fold0-cal0 done in \d+\.\d s", err), err
    assert "[progress] xsuccess" not in err


def test_xsuccess_hpo_prints_one_line_per_trial(capsys):
    """The xsuccess HPO (a direct optuna study) prints one ``[progress] study xsuccess`` line per trial (F21)."""
    pytest.importorskip("xgboost")
    from scripts.train_xsuccess import _hpo_params

    _hpo_params(_xsuccess_pooled(), 3)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[progress] study ")]
    assert len(lines) == 3, lines
    for k, ln in enumerate(lines, start=1):
        assert re.fullmatch(rf"\[progress\] study xsuccess trial {k}/3 best=\S+ in \d+\.\d s", ln), ln


def test_xsuccess_hpo_progress_keeps_best_params(monkeypatch):
    """The callback is a sink: the best parameters are equal with and without it."""
    pytest.importorskip("xgboost")
    import scripts._study_shared as SS
    from scripts.train_xsuccess import _hpo_params

    with_printer = _hpo_params(_xsuccess_pooled(), 3)

    class _Silent:
        def __init__(self, *_a, **_k) -> None:
            pass

        def optuna_callback(self, study, trial) -> None:
            pass

    monkeypatch.setattr(SS, "TrialPrinter", _Silent)
    assert _hpo_params(_xsuccess_pooled(), 3) == with_printer
