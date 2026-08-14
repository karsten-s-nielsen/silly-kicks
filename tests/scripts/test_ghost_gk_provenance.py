"""Corpus-provenance metadata block (spec 2026-07-20 §6). Providers + counts only, no split.

The save()/load() plumbing lives in silly_kicks.tracking._ghost_gk (landed with the
parameters-only change); this module pins the CONTRACT: what is recorded, and what must never be.
"""

import json

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit_small() -> GhostGkModel:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((300, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 300).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 300), "gk_y": rng.uniform(25, 45, 300)})
    m = GhostGkModel(n_estimators=20)
    m.fit(X, labels)
    return m


def test_provenance_recorded_when_supplied(tmp_path):
    m = _fit_small()
    m.corpus_provenance = {"providers": ["gradientsports", "sportec"], "n_games": 71, "n_rows": 300}
    m.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["corpus_provenance"]["n_games"] == 71
    assert meta["corpus_provenance"]["providers"] == ["gradientsports", "sportec"]
    # NEVER a per-match id list and NEVER a public/restricted split (owner decision 2026-07-20)
    assert "match_ids" not in meta["corpus_provenance"]
    assert "visibility" not in meta["corpus_provenance"]


def test_provenance_round_trips_through_load(tmp_path):
    m = _fit_small()
    m.corpus_provenance = {"providers": ["gradientsports"], "n_games": 5, "n_rows": 300}
    m.save(tmp_path / "m")
    reloaded = GhostGkModel.load(tmp_path / "m")
    assert reloaded.corpus_provenance == {"providers": ["gradientsports"], "n_games": 5, "n_rows": 300}


def test_provenance_null_when_absent(tmp_path):
    m = _fit_small()
    m.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["corpus_provenance"] is None


# --------------------------------------------------------------------------------------------
# The stamped platform must not depend on the operator remembering a flag.


def test_training_platform_defaults_to_the_detected_platform():
    """Ghost was the ONLY trained model whose machine identity was operator-supplied: xShot and
    xCross both record `platform.platform()` unconditionally, while ghost's `--training-platform`
    defaulted to None. The flag was in fact forgotten during the ADR-050 re-fit."""
    from scripts.train_ghost_gk import resolve_training_platform

    prov = {"platform": "Linux-6.11.0-aarch64-with-glibc2.39", "machine": "aarch64"}
    assert resolve_training_platform(None, prov) == "Linux-6.11.0-aarch64-with-glibc2.39"


def test_an_explicit_label_still_wins():
    """The override exists for a human-meaningful card label; detection must not clobber it."""
    from scripts.train_ghost_gk import resolve_training_platform

    prov = {"platform": "Linux-6.11.0-aarch64-with-glibc2.39", "machine": "aarch64"}
    assert resolve_training_platform("dgx-spark-aarch64", prov) == "dgx-spark-aarch64"


def test_the_stamped_platform_is_never_None():
    """An absent platform and an unknown one are the same useless value downstream."""
    from scripts.train_ghost_gk import resolve_training_platform

    assert resolve_training_platform(None, {"platform": "Windows-11", "machine": "AMD64"})
    assert resolve_training_platform("", {"platform": "Windows-11", "machine": "AMD64"}) == "Windows-11"
