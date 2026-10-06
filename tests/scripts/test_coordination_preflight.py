"""ADR-069 Layer 2 for the TF-58 raw-artifact corpus (spec 8.3; owner ruling 2026-10-03).

Every TF-58 corpus pass loads, BEFORE any compute, one detection-aware match per provider in its slice through the
pass's own loader and refuses -- with the ADR-069 native-rebuild remedy -- when that match's ``visibility`` flag was
discarded. A loader path that throws the flag away is caught at the start of the pass, never deep in it (the ADR-069
incident failed an hour in). The per-match ``detected_mask`` trap stays for a single match's hole.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _fake_corpus import SpyLoader, make_loaded, make_ref  # noqa: E402

from scripts._coordination_corpus import visibility_preflight  # noqa: E402
from tests.coordination._fixtures import make_coordination_match  # noqa: E402
from tests.scripts._script_population import coordination_corpus_drivers  # noqa: E402

# DERIVED, not hand-listed (review A-54; ADR-056): the same `run_params_token` population the worker-combine token
# guards use, single-sourced in `_script_population`. A new TF-58 corpus driver is covered automatically.
_DRIVERS = coordination_corpus_drivers()


def _skillcorner(match_id: str, *, discarded: bool):
    frames = make_coordination_match(seconds=10.0, hz=10.0, provider="skillcorner")
    if discarded:
        frames["visibility"] = None  # the kloppy gateway's signature (ADR-069)
    return make_loaded("skillcorner", match_id, frames=frames)


def test_a_discarded_detection_flag_is_refused_before_the_pass():
    loader = SpyLoader({("skillcorner", "m1"): _skillcorner("m1", discarded=True)})
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        visibility_preflight([make_ref("skillcorner", "m1")], loader)


def test_one_match_per_detection_aware_provider_and_none_for_fully_observed():
    loader = SpyLoader({("skillcorner", m): _skillcorner(m, discarded=False) for m in ("m1", "m2")})
    refs = [make_ref("sportec", "s1"), make_ref("skillcorner", "m1"), make_ref("skillcorner", "m2")]
    visibility_preflight(refs, loader)
    assert [key for key, _kw in loader.calls] == [("skillcorner", "m1")]


def test_a_match_that_fails_to_load_is_skipped_for_the_next():
    # a download/parse failure is a per-match problem the pass records; the pre-flight checks the next match instead
    loader = SpyLoader(
        {("skillcorner", "m2"): _skillcorner("m2", discarded=True)},
        fail={("skillcorner", "m1")},
    )
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        visibility_preflight([make_ref("skillcorner", "m1"), make_ref("skillcorner", "m2")], loader)
    assert [key for key, _kw in loader.calls] == [("skillcorner", "m1"), ("skillcorner", "m2")]


def test_the_probe_is_bounded_to_attempts_loads_per_provider():
    # The justification of its ADR-052 Rule C exemption (_UNSHARDED_LOOP_EXEMPT): a bounded probe, never a corpus
    # pass. After `attempts` failed loads of a provider it stops; the per-match trap stays that provider's guard.
    refs = [make_ref("skillcorner", f"m{i}") for i in range(1, 6)]
    loader = SpyLoader({}, fail={("skillcorner", f"m{i}") for i in range(1, 6)})
    with pytest.warns(UserWarning, match="could not verify"):  # every probed match failed -> unverified, warned (nit)
        visibility_preflight(refs, loader, attempts=3)
    assert [key for key, _kw in loader.calls] == [("skillcorner", "m1"), ("skillcorner", "m2"), ("skillcorner", "m3")]


def test_a_provider_whose_probed_matches_all_fail_is_WARNED_not_silent():
    # nit: all attempts of a detection-aware provider failing to load leaves its visibility flag unverified at the pass
    # boundary -- a gap that must be recorded (a warning), not only printed per match.
    loader = SpyLoader({}, fail={("skillcorner", "m1"), ("skillcorner", "m2")})
    with pytest.warns(UserWarning, match=r"could not verify \['skillcorner'\]"):
        visibility_preflight([make_ref("skillcorner", "m1"), make_ref("skillcorner", "m2")], loader, attempts=2)


def test_a_fully_verified_provider_does_not_warn():
    # the other side: when the provider IS verified, no unverified-gap warning fires.
    import warnings as _w

    loader = SpyLoader({("skillcorner", "m1"): _skillcorner("m1", discarded=False)})
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning would fail here
        visibility_preflight([make_ref("skillcorner", "m1")], loader)


def _calls(fn: ast.FunctionDef, name: str) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    )


@pytest.mark.parametrize("driver", _DRIVERS)
def test_every_corpus_pass_runs_the_preflight_before_its_for_each(driver):
    # Registry gate (ADR-056 style, the population DERIVED): every function that opens the corpus (corpus_source) and
    # walks it (for_each) runs visibility_preflight between the two. `main`'s --list-matches path never walks it.
    tree = ast.parse((_SCRIPTS / f"{driver}.py").read_text(encoding="utf-8"))
    passes = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and _calls(fn, "corpus_source") and _calls(fn, "for_each")
    ]
    assert passes  # non-vacuity: the driver has corpus passes
    for fn in passes:
        pre = _calls(fn, "visibility_preflight")
        assert pre, f"{driver}.{fn.name} walks the corpus without the Layer-2 pre-flight"
        assert _calls(fn, "corpus_source")[0] < pre[0] < _calls(fn, "for_each")[0], f"{driver}.{fn.name}"


def test_a_pass_refuses_before_any_compute(monkeypatch, tmp_path):
    # behaviour, end to end on one pass: the D3 metrics pass meets a discarded flag and never reaches for_each
    import validate_team_coordination as d3

    loader = SpyLoader({("skillcorner", "m1"): _skillcorner("m1", discarded=True)})
    monkeypatch.setattr(d3, "corpus_source", lambda args: ([make_ref("skillcorner", "m1")], loader))
    monkeypatch.setattr("scripts._driver.for_each", lambda *a, **kw: pytest.fail("the pass computed"))
    args = SimpleNamespace(
        out=str(tmp_path),
        providers=("skillcorner",),
        match_ids_json=None,
        corpus_json=None,
        max_matches=None,
        cache_dir=None,
        token=None,
        in_package_params=True,
        derivation=None,
        calibration=None,
    )
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        d3._pass_metrics(args, {"commit": "0" * 40, "dirty": False, "tree_state": "clean"})
