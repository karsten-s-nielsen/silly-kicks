"""The RC4 measurement driver's seam integration, exercised without pining access.

`measure()` needs an owner token and a real match, so the driver's `run()` -- the ADR-052 `for_each`
integration, the one-row-frame contract, the shard round-trip and the `distinct_labels` JSON
encode/decode -- would otherwise ship having never executed. That is the gap this file closes: the
provider measurement is stubbed, everything around it is real.

Why it matters here specifically: this driver exists BECAUSE the original RC4 measurement was an
ad-hoc pass whose numbers could not be re-derived. A committed replacement that has never run would
reproduce the same problem one level up.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "measure_rc4_orientation.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("measure_rc4_orientation", _DRIVER)
    assert spec is not None and spec.loader is not None, f"could not load {_DRIVER}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FAKE = {
    "skillcorner": {
        "match_id": "1886347",
        "n_frames": 43458,
        "player_rows": 956076,
        "unlabelled_fraction": 1.0,
        "distinct_labels": [],
        "n_actions": 1197,
        "n_flip_true": 0,
        "flip_true_fraction": 0.0,
        "orientation_warnings": 1,
    },
    "idsse": {
        "match_id": "DFL-MAT-J03WMX",
        "n_frames": 145967,
        "player_rows": 3211274,
        "unlabelled_fraction": 0.0,
        "distinct_labels": ["ltr", "rtl"],
        "n_actions": 1363,
        "n_flip_true": 718,
        "flip_true_fraction": 0.5267791636096845,
        "orientation_warnings": 0,
    },
}

_PROV = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}


def test_run_round_trips_every_field_through_the_shard_seam(tmp_path, monkeypatch):
    """The real `for_each` + parquet round trip, with only the provider measurement stubbed."""
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))

    record = mod.run(
        label="prefix",
        tracking_limit=None,
        cache_dir=None,
        shard_dir=str(tmp_path / "shards"),
        run_prov=_PROV,
    )

    assert set(record) == {"_provenance", "skillcorner", "idsse"}
    for provider, expected in _FAKE.items():
        assert record[provider] == expected, f"{provider} did not survive the shard round trip"


def test_distinct_labels_survives_as_a_LIST_not_a_json_string(tmp_path, monkeypatch):
    """The one field that needs encoding, asserted on both shapes.

    `distinct_labels` is a list, and a parquet row wants scalars, so `run()` json-encodes it on the
    way in and decodes on the way out. A silent failure here yields the STRING '["ltr","rtl"]' in the
    artifact, which still looks plausible in a JSON file -- exactly the class of defect that is only
    caught by asserting the type, not the presence.
    """
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))

    record = mod.run(
        label="postfix", tracking_limit=None, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV
    )

    assert record["idsse"]["distinct_labels"] == ["ltr", "rtl"]
    assert isinstance(record["idsse"]["distinct_labels"], list)
    # and the empty case, which is what the pre-fix SkillCorner side actually produces
    assert record["skillcorner"]["distinct_labels"] == []


def test_provenance_block_carries_the_ADR052_vocabulary(tmp_path, monkeypatch):
    """`run_commit` / `run_tree_dirty` / `run_tree_state`, plus the corpus bound.

    The shipped artifacts predate this driver and use the bare key `commit`; the whole point of
    committing a producer was to converge on ADR-052's names, so assert them rather than assume.
    """
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))

    record = mod.run(label="prefix", tracking_limit=3000, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV)

    prov = record["_provenance"]
    assert prov["run_commit"] == "0" * 40
    assert prov["run_tree_dirty"] is False
    assert prov["run_tree_state"] == "clean"
    # The CAP must be recorded even when set -- an unrecorded one is the defect this driver exists for.
    assert prov["tracking_limit"] == 3000
    assert prov["max_per_provider"] == 1


def test_a_second_run_RESUMES_from_the_shards_instead_of_re_measuring(tmp_path, monkeypatch):
    """The property adopting `for_each` was supposed to buy, asserted rather than assumed."""
    mod = _load_driver()
    calls: list[str] = []

    def _counting(provider, **_kw):
        calls.append(provider)
        return dict(_FAKE[provider])

    monkeypatch.setattr(mod, "measure", _counting)
    shard_dir = str(tmp_path / "shards")
    first = mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=shard_dir, run_prov=_PROV)
    assert sorted(calls) == ["idsse", "skillcorner"]

    calls.clear()
    second = mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=shard_dir, run_prov=_PROV)
    assert calls == [], f"re-measured {calls} instead of resuming from shards"
    assert second == first, "the resumed run did not reproduce the first"


def test_a_FAILING_provider_raises_rather_than_writing_a_partial_artifact(tmp_path, monkeypatch):
    """A half-measured artifact is worse than none: it looks complete and cites two providers."""
    mod = _load_driver()

    def _boom(provider, **_kw):
        raise RuntimeError(f"{provider} exploded")

    monkeypatch.setattr(mod, "measure", _boom)
    with pytest.raises((RuntimeError, Exception)):
        mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV)


def test_the_stub_is_not_secretly_doing_the_work(tmp_path, monkeypatch):
    """Non-vacuity: if `run()` ignored `measure` entirely, every test above would still pass."""
    mod = _load_driver()
    sentinel = {**_FAKE["idsse"], "n_flip_true": 999999}
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(sentinel))

    record = mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV)
    assert record["skillcorner"]["n_flip_true"] == 999999
    assert record["idsse"]["n_flip_true"] == 999999


def test_the_shard_frame_is_ONE_ROW_per_provider(tmp_path, monkeypatch):
    """ADR-052 D7: the work -> tidy frame contract. A multi-row frame would silently drop rows,
    because `run()` reads `frame.iloc[0]`."""
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))
    shard_dir = tmp_path / "shards"
    mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(shard_dir), run_prov=_PROV)

    shards = list(shard_dir.rglob("*.parquet"))
    assert len(shards) == 2, f"expected one shard per provider, found {len(shards)}"
    for s in shards:
        assert len(pd.read_parquet(s)) == 1, f"{s.name} is not one row"


def test_the_artifact_serialises_to_json(tmp_path, monkeypatch):
    """`run()`'s output is written with `json.dumps`; a numpy scalar leaking through would raise
    there rather than here, at the end of a corpus pass."""
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))
    record = mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV)
    reparsed = json.loads(json.dumps(record))
    assert reparsed["idsse"]["flip_true_fraction"] == pytest.approx(0.5267791636096845)


def test_a_LOAD_failure_RAISES_rather_than_becoming_artifact_DATA(tmp_path, monkeypatch):
    """The failure that actually happens -- no token, no network -- through the REAL `measure()`.

    Its sibling above stubs `measure` itself, so it is STRUCTURALLY BLIND to a `measure` that
    swallows its own load error. That is precisely what shipped, and it was measured: `measure`
    caught `Exception` and returned an error dict, `_work` wrapped it in an ordinary one-row frame,
    `for_each` wrote it as a healthy shard, `res.failures` stayed empty, `run()` returned normally,
    and `main()` would have written `{"skillcorner": {"error": ...}}` over the committed artifact at
    the DEFAULT `--out-dir` and exited 0.

    Worse, the error shard made `already_done()` true forever: every resume reported
    `skip (shard exists)` and re-published the memoized error, recoverable only by deleting a 16-hex
    generation directory by hand.
    """
    mod = _load_driver()

    def _boom(**_kw):
        raise RuntimeError("PINING_TOKEN not set")

    fake = types.ModuleType("_loader_pining")
    fake.load_matches = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_loader_pining", fake)

    with pytest.raises(RuntimeError, match=r"provider\(s\) failed"):
        mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(tmp_path / "s"), run_prov=_PROV)


def test_a_failed_provider_leaves_NO_shard_so_a_resume_REDOES_it(tmp_path, monkeypatch):
    """The ADR-052 property the swallow inverted: a failure must not be memoized as done."""
    mod = _load_driver()

    def _boom(**_kw):
        raise RuntimeError("transient network blip")

    fake = types.ModuleType("_loader_pining")
    fake.load_matches = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_loader_pining", fake)

    shard_dir = tmp_path / "s"
    with pytest.raises(RuntimeError):
        mod.run(label="prefix", tracking_limit=None, cache_dir=None, shard_dir=str(shard_dir), run_prov=_PROV)

    assert list(shard_dir.rglob("*.parquet")) == [], "a failure was memoized as a completed shard"


def test_a_DIFFERENT_commit_does_not_reuse_the_previous_runs_shards(tmp_path, monkeypatch):
    """`run_commit` is in `token_inputs`, and it is load-bearing rather than decorative.

    This driver's entire subject is that the CODE differs between its two sides. A token of
    {measurement, tracking_limit, label} captures no code at all, so two runs at different commits
    under the same label would share a generation: the second would silently REUSE the first's
    shards while stamping its own `run_commit` into the artifact -- the false-provenance failure
    this driver exists to prevent, reintroduced one level down.

    Both directions are asserted: the same commit must still RESUME (or adopting the seam bought
    nothing), and a different commit must re-measure.
    """
    mod = _load_driver()
    seen: list[str] = []

    def _counting(provider, **_kw):
        seen.append(provider)
        return dict(_FAKE[provider])

    monkeypatch.setattr(mod, "measure", _counting)
    shard_dir = str(tmp_path / "s")
    at_a = {"commit": "a" * 40, "dirty": False, "tree_state": "clean"}
    at_b = {"commit": "b" * 40, "dirty": False, "tree_state": "clean"}

    mod.run(label="postfix", tracking_limit=None, cache_dir=None, shard_dir=shard_dir, run_prov=at_a)
    seen.clear()

    mod.run(label="postfix", tracking_limit=None, cache_dir=None, shard_dir=shard_dir, run_prov=at_a)
    assert seen == [], "same commit should have resumed from shards"

    mod.run(label="postfix", tracking_limit=None, cache_dir=None, shard_dir=shard_dir, run_prov=at_b)
    assert sorted(seen) == ["idsse", "skillcorner"], "a different commit reused stale shards"


def test_a_CAPPED_run_REFUSES_to_overwrite_the_committed_artifacts(tmp_path, monkeypatch):
    """`--tracking-limit` + the default `--out-dir` must abort BEFORE any corpus work.

    The committed artifacts are full-frame measurements that everything cites; a cap DEPRESSES
    `flip_true_fraction`, so a capped run writing to the same directory would replace them with
    lower bounds. `tracking_limit` is recorded either way, so this is not about hiding the cap -- it
    is about not silently overwriting the cited values with weaker ones.

    The refusal is asserted to happen BEFORE `measure()` is ever called: it originally sat after
    `run()`, so an operator paid for the whole corpus pass and was refused afterwards, which is a
    worse version of no refusal.
    """
    mod = _load_driver()
    called: list[str] = []
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: called.append(provider) or dict(_FAKE[provider]))
    # `--allow-dirty` only gets PAST the provenance refusal, which correctly fires first on a
    # modified tree. That ordering is worth knowing in itself: provenance, then the cap, then any
    # corpus work -- both refusals land before the expensive part.
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure_rc4_orientation.py", "--label", "prefix", "--tracking-limit", "3000", "--allow-dirty"],
    )

    with pytest.raises(SystemExit, match="CAPPED measurement"):
        mod.main()
    assert called == [], "the corpus pass ran before the refusal"


def test_an_UNCAPPED_run_is_not_refused(tmp_path, monkeypatch):
    """Non-vacuity: the guard must gate on the CAP, not on the output directory alone."""
    mod = _load_driver()
    monkeypatch.setattr(mod, "measure", lambda provider, **_kw: dict(_FAKE[provider]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_rc4_orientation.py",
            "--label",
            "prefix",
            "--out-dir",
            str(tmp_path),
            "--allow-dirty",
            "--shard-dir",
            str(tmp_path / "sh"),
        ],
    )
    assert mod.main() == 0
    assert (tmp_path / "prefix_measurement.json").exists()
