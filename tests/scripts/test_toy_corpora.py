"""ADR-128: the toy corpora are byte-deterministic and use the staged layout an HF job reads."""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from scripts._toy_corpora import TOY_GENERATION, build_flat_corpus, build_pergame_corpus


def _digest(root) -> dict[str, str]:
    files = (p for p in sorted(root.rglob("*")) if p.is_file())
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_flat_corpus_is_byte_deterministic_and_staged_shaped(tmp_path):
    a = build_flat_corpus(tmp_path / "a", n_games=3, providers=("skillcorner", "sportec"), seed=4)
    b = build_flat_corpus(tmp_path / "b", n_games=3, providers=("skillcorner", "sportec"), seed=4)
    assert _digest(a) == _digest(b)
    names = set(_digest(a))
    assert f"shards/{TOY_GENERATION}/skillcorner__100.parquet" in names
    assert "_actions/sportec__101.parquet" in names and "home_teams.json" in names
    other = build_flat_corpus(tmp_path / "c", n_games=3, providers=("skillcorner", "sportec"), seed=5)
    assert _digest(other) != _digest(a)


def test_pergame_corpus_is_byte_deterministic(tmp_path):
    a = build_pergame_corpus(tmp_path / "a", n_games=2, provider="gradientsports")
    b = build_pergame_corpus(tmp_path / "b", n_games=2, provider="gradientsports")
    assert _digest(a) == _digest(b)
    assert {"gradientsports-100/frames.parquet", "gradientsports-101/shots.parquet"} <= set(_digest(a))


# ---- fixture fidelity, measured on what the builders WRITE (Reviewer B CCHC-IMPL-15) --------------------


class _LocalFs:
    """`HfFileSystem.open` over a local toy dataset, so the toy is measured by the SAME `measure_id_kinds`
    that measured the staged data into the lock."""

    def __init__(self, root):
        self.root = root

    def open(self, path, mode="rb"):
        rel = path.split("@local/", 1)[1]
        p = self.root / rel
        if not p.is_file():
            raise FileNotFoundError(rel)
        return open(p, mode)


def _toy_rec(root):
    from scripts._driver_sources import staged_flat_refs, staged_pergame_refs

    if (root / "shards").is_dir():
        refs = staged_flat_refs(root.parent, root.name)
        return {"kind": "flat_tc3", "generation": TOY_GENERATION, "units": [{"stem": r.stem} for r in refs]}
    refs = staged_pergame_refs(root.parent, root.name)
    return {"kind": "pergame", "units": [{"stem": r.stem, "source_provider": r.source_provider} for r in refs]}


def _campaign_datasets_by_module() -> dict[str, set[str]]:
    from scripts._toml import load_toml
    from scripts.hf_jobs.registry import spec_for

    out: dict[str, set[str]] = {}
    for job in load_toml(pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml"))["jobs"]:
        spec = spec_for(job["job"])
        module = getattr(spec, "pipeline", "") or getattr(spec, "module", "")
        out.setdefault(module, set()).add(job["dataset"])
    return out


def _staged_modules() -> list[str]:
    from scripts._toy_corpora import TOY_BUILDERS
    from scripts.hf_jobs import datasets as D

    flat = {n for n, d in D.declared().items() if d.kind != "statsbomb_open"}
    return sorted(m for m, ds in _campaign_datasets_by_module().items() if m in TOY_BUILDERS and ds & flat)


@pytest.mark.parametrize("module", _staged_modules())
def test_every_toy_writes_the_locked_id_kinds_and_mixes_them_where_the_campaign_does(module, tmp_path):
    from scripts._toy_corpora import PROVIDER_ID_KIND, build_toy_for
    from scripts.hf_jobs import datasets as D

    locked = {}
    for name in _campaign_datasets_by_module()[module]:
        locked.update(D.locked(name).get("provider_id_kinds") or {})
    data = build_toy_for(module, tmp_path / "data")
    written: dict[str, str] = {}
    for root in sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith(".")):
        written.update(D.measure_id_kinds(root.name, _toy_rec(root), repo="toy", revision="local", fs=_LocalFs(root)))
    assert written, f"{module}: the toy wrote no staged unit"
    for provider, kind in written.items():
        assert kind == PROVIDER_ID_KIND[provider], (module, provider, kind)  # what the builder really wrote
    if len(set(locked.values())) > 1:  # the campaign's data mixes id kinds: the toy must, too
        assert set(written.values()) >= set(locked.values()), (module, written, locked)
