"""TF-19 sign-off package: the Layer 2 spells pass feeding the ATT power leg."""

from __future__ import annotations

import json

import pytest

import scripts.build_layer2_spells as mod  # bare import: tests/scripts/ has NO __init__.py


def _write(dest, name, payload):
    (dest / f"manifest_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_treated_prevalence_is_reported_at_CORPUS_scope(tmp_path):
    """A rare treatment is invisible per match -- one match can legitimately contain zero treated
    spells -- but it decides the entire power curve, because cluster resamples drawn from a corpus
    with almost no treated units carry a single treatment class and are not estimable at all."""
    _write(tmp_path, "p0", {"n_matches": 8, "n_spells": 4000, "n_treated": 10})
    _write(tmp_path, "p1", {"n_matches": 8, "n_spells": 6000, "n_treated": 30})
    got = mod._aggregate_manifests(tmp_path)
    assert got["n_spells"] == 10000
    assert got["n_treated"] == 40
    assert got["treated_prevalence"] == pytest.approx(0.004)


def test_prevalence_of_an_empty_corpus_is_None_not_a_zero_division(tmp_path):
    """None says "no spells were built"; 0.0 would say "spells were built and none were treated"."""
    got = mod._aggregate_manifests(tmp_path)
    assert got["treated_prevalence"] is None
    assert got["n_spells"] == 0


def test_a_ZERO_prevalence_corpus_reports_0_not_None(tmp_path):
    """The other side of the test above: spells exist, none treated. That is a real, reportable
    finding (the design never fires) and must be distinguishable from "nothing ran"."""
    _write(tmp_path, "p0", {"n_matches": 4, "n_spells": 500, "n_treated": 0})
    got = mod._aggregate_manifests(tmp_path)
    assert got["treated_prevalence"] == 0.0
    assert got["n_spells"] == 500


def test_out_is_required_unless_listing_matches(monkeypatch, capsys):
    """`--list-matches` writes no artifact, so it is the one mode exempt from --out AND from the
    clean-tree requirement. Every other invocation must name where the shards go."""
    monkeypatch.setattr("sys.argv", ["build_layer2_spells.py"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "--out is required" in str(e.value)


def _install_spells_corpus(monkeypatch, keys=(("gradientsports", "m1"),), *, fail=(), exclude=None):
    """Point the driver's loader seam at a fake corpus (spec section 4.5: refs + load_match)."""
    import pandas as pd
    from _fake_corpus import SpyLoader, install_fake_corpus, make_loaded, make_ref

    import scripts._loader_pining as loader

    spells = pd.DataFrame({"Z": [0, 1], "r": [1.0, 2.0], "theta": [0.1, 0.2]})
    refs = [make_ref(p, m) for p, m in keys]
    served = {(p, m): make_loaded(p, m, home_team_id="5") for p, m in keys}
    spy = SpyLoader(served, fail=fail, exclude=exclude)
    install_fake_corpus(monkeypatch, loader, refs=refs, loader=spy)

    import silly_kicks.causal as causal
    import silly_kicks.causal._confounders as conf

    monkeypatch.setattr(causal, "build_opportunities", lambda *a, **k: spells.copy())
    monkeypatch.setattr(causal, "layer2_config", lambda *a, **k: object())
    monkeypatch.setattr(conf, "join_layer2_confounders", lambda sp, **k: sp)
    return spy


def test_main_walks_a_match_end_to_end_and_writes_its_shard(tmp_path, monkeypatch):
    """Executes the real control flow with the refs+load_match seam (spec section 4.5).

    This exists because a refactor moved the per-match shard write onto a `tag` bound AFTER the
    loop -- a NameError on the first match, invisible to compile checks, to lint, and to every
    unit test of the helpers. Only running `main()` catches that shape.
    """
    import sys

    _install_spells_corpus(monkeypatch)
    monkeypatch.setattr(mod, "_aggregate_manifests", lambda dest: {"n_matches": 1, "n_spells": 2, "n_treated": 1})
    monkeypatch.setattr(sys, "argv", ["build_layer2_spells.py", "--out", str(tmp_path), "--allow-dirty"])

    mod.main()

    # `rglob`, not a hard-coded path: since the `_driver` migration the shard lives inside a
    # GENERATION directory (`shards/<token>/…`) whose name is a digest of the declared inputs.
    shards = list((tmp_path / "shards").rglob("gradientsports__m1.parquet"))
    assert shards, "the per-match shard was never written"
    assert (tmp_path / "layer2_spells.parquet").is_file()
    assert not list(tmp_path.glob("**/*.tmp*")), "atomic temp file left behind"


def test_resume_does_not_reload_a_finished_match(tmp_path, monkeypatch):
    """resume-before-load (spec section 4.3): a match whose shard exists is never handed to `load`."""
    import sys

    spy = _install_spells_corpus(monkeypatch)
    monkeypatch.setattr(mod, "_aggregate_manifests", lambda dest: {"n_matches": 1, "n_spells": 2, "n_treated": 1})
    monkeypatch.setattr(sys, "argv", ["build_layer2_spells.py", "--out", str(tmp_path), "--allow-dirty"])

    mod.main()
    n_after_first = len(spy.calls)
    assert n_after_first == 1
    mod.main()
    assert len(spy.calls) == n_after_first, "resume re-loaded a match whose shard already existed"


# The ASCII contract for this driver is enforced at SOURCE level (stricter than docstrings alone)
# by the derived gate in test_build_gkdv_arm_values.py, which also asserts this module is NOT on the
# pre-existing-debt list. Duplicating a weaker check here would just be a second thing to maintain.
