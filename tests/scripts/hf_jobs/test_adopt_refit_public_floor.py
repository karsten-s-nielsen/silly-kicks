"""Task 4 (adopt-for-refit seam): the refit public floor falls back to the fresh acceptance gate when the
COMMITTED bundle fails its own contract load (a fingerprint drift — the state a refit exists to replace).

The pre-check uses the SUBPROCESS `load_check` (adopt.py:268, HFJOBS_PYTHON) — the SAME env the head-to-head
loads+crashes in (headtohead.py:244) — so an in-process verdict cannot diverge. Gated to `mode == "refit"`;
a rematerialize public arm must never take the fallback (anti-goal §4). Never auto-pass."""

from __future__ import annotations

import types

from scripts.hf_jobs import adopt as ad

_GATES_PASS = {
    "enough_usable_folds": True,
    "pr_auc_gt_base_rate": True,
    "brier_lt_base_rate_brier": True,
    "log_loss_lt_uniform": True,
}
_GATES_FAIL = {**_GATES_PASS, "brier_lt_base_rate_brier": False}  # the xcross-public shape


def _spec():
    return types.SimpleNamespace(certification_for=lambda tier: "public")


def _inst():
    return types.SimpleNamespace(job="xshot_occurrence", tier="public", variant="default", args=())


def _ref():
    return types.SimpleNamespace(kind="wheel", id="a1", path="silly_kicks/tracking/_xshot_weights/default")


def _certify(monkeypatch, tmp_path, *, mode, gates, load_result, run_spy, loadcheck_calls=None):
    def load_check(*a, **k):
        if loadcheck_calls is not None:
            loadcheck_calls.append(1)
        return load_result

    monkeypatch.setattr(ad, "load_check", load_check)
    monkeypatch.setattr(ad, "_eval_corpus_args", lambda inst, parent, **k: ())  # head-to-head path reachable
    return ad._certify(_spec(), _inst(), _ref(), tmp_path, {"acceptance": gates}, {}, run=run_spy, mode=mode)  # pyright: ignore[reportArgumentType]


def test_drifted_committed_floor_uses_fresh_acceptance_pass(monkeypatch, tmp_path):
    runs: list = []
    out = _certify(
        monkeypatch,
        tmp_path,
        mode="refit",
        gates=_GATES_PASS,
        load_result="drift: fingerprint mismatch",
        run_spy=lambda *a, **k: runs.append(1),
    )
    assert out["ok"] is True
    assert "baseline" in out.get("reason", "").lower()
    assert runs == []  # the head-to-head is NOT run against an unloadable committed bundle


def test_drifted_committed_floor_fails_on_bad_fresh(monkeypatch, tmp_path):
    runs: list = []
    out = _certify(
        monkeypatch,
        tmp_path,
        mode="refit",
        gates=_GATES_FAIL,
        load_result="drift: fingerprint mismatch",
        run_spy=lambda *a, **k: runs.append(1),
    )
    assert out["ok"] is False  # never auto-pass: a failing fresh acceptance fails the floor
    assert runs == []


def test_loadable_committed_uses_headtohead(monkeypatch, tmp_path):
    runs: list = []
    _certify(
        monkeypatch,
        tmp_path,
        mode="refit",
        gates=_GATES_PASS,
        load_result=None,  # committed loads clean
        run_spy=lambda *a, **k: runs.append(1) or types.SimpleNamespace(returncode=1, stderr="stop"),
    )
    assert runs  # a loadable committed bundle → the head-to-head CLI IS invoked (fallback NOT taken)


def test_rematerialize_drifted_committed_does_not_fallback(monkeypatch, tmp_path):
    runs: list = []
    lc_calls: list = []
    _certify(
        monkeypatch,
        tmp_path,
        mode="rematerialize",
        gates=_GATES_PASS,
        load_result="drift",
        run_spy=lambda *a, **k: runs.append(1) or types.SimpleNamespace(returncode=1, stderr="stop"),
        loadcheck_calls=lc_calls,
    )
    assert lc_calls == []  # _certify does NOT consult load_check for a fallback in rematerialize mode
    assert runs  # it takes the head-to-head path as today (anti-goal §4: no refit fallback)
