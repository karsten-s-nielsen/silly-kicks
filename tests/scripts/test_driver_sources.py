"""Neutral loader port (ADR-052 D14 / ADR-128): refs read the lock (HF) or the local tree (dev/toy)."""

from __future__ import annotations

import ast
import pathlib

from scripts import _driver_sources as S
from scripts._toy_corpora import build_flat_corpus, build_pergame_corpus


def test_flat_refs_list_units_and_their_files(tmp_path):
    # A toy dataset name is never in datasets.lock.json: the lock answers for a recorded name.
    root = build_flat_corpus(tmp_path / "toyflat", n_games=3, providers=("skillcorner",))
    refs = S.staged_flat_refs(tmp_path, "toyflat")
    assert [r.stem for r in refs] == sorted(r.stem for r in refs)
    gen = next((root / "shards").iterdir()).name
    assert refs[0].files == (f"shards/{gen}/{refs[0].stem}.parquet", f"_actions/{refs[0].stem}.parquet")
    assert refs[0].source_provider == "skillcorner" and refs[0].key == (refs[0].provider, refs[0].game)


def test_pergame_refs_skip_dot_directories(tmp_path):
    build_pergame_corpus(tmp_path / "toyprobe", n_games=2, provider="gradientsports")
    (tmp_path / "toyprobe" / ".cache").mkdir()  # what `hf download --local-dir` leaves behind
    (tmp_path / "toyprobe" / ".cache" / "frames.parquet").write_bytes(b"")
    refs = S.staged_pergame_refs(tmp_path, "toyprobe")
    assert len(refs) == 2 and all(not r.stem.startswith(".") for r in refs)


def test_driver_sources_carries_no_visibility_concept():
    src = pathlib.Path(S.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "scripts._loader_pining" not in mods and "scripts._corpus" not in mods


def test_loader_pining_still_exports_the_moved_names():
    from scripts import _loader_pining as P

    assert P.LoadedMatch is S.LoadedMatch and P.resolve_cache_dir is S.resolve_cache_dir
    assert P.CORPUS_CACHE_ENV == S.CORPUS_CACHE_ENV
