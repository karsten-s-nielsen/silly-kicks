"""Trainer-side guards (ADR-050): feature-cache identity, corpus fingerprints, provider fail-fast.

These live in ``scripts/``, which has NO ``__init__.py`` -- so trainers are loaded by file path and
their own imports use the bare ``from _cache import ...`` form.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"could not load scripts/{name}.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ghost_cache_token_is_derived_from_the_geometry_constants():
    """A hand-bumped literal goes stale INSIDE the re-fit cycle it protects: extract features, flip
    the constant, re-run -- and the second run silently reuses the first run's 40.3 features while
    stamping a 20.16 contract. Deriving the token from the constants auto-invalidates on the flip
    with zero discipline required."""
    import silly_kicks.spadl.config as _spc

    t = _load("train_ghost_gk")
    before = t.cache_token()
    original = _spc.penalty_area_half_width
    try:
        # Flip the CANONICAL source -- ghost's own `_PENALTY_AREA_*` constants were deleted when it
        # migrated onto it (ADR-050 §6), and `cache_token` followed them there.
        _spc.penalty_area_half_width = 20.15
        assert t.cache_token() != before, "token must change when the box constant changes"
    finally:
        _spc.penalty_area_half_width = original


def test_corpus_fingerprint_distinguishes_corpora():
    """The whole point: a changed corpus must MISS. The constant 'schema-v2' token cannot."""
    # NOTE the import form: `scripts/` has NO __init__.py, so `from scripts._cache import ...` is a
    # ModuleNotFoundError. This is the established idiom (tests/scripts/test_cache_schema.py).
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from _cache import corpus_fingerprint

    a = corpus_fingerprint([("gs", "1", "public"), ("gs", "2", "public")])
    b = corpus_fingerprint([("gs", "1", "public")])
    c = corpus_fingerprint([("gs", "2", "public"), ("gs", "1", "public")])  # order-insensitive
    assert a != b
    assert a == c


@pytest.mark.parametrize(
    ("name", "argv"),
    [
        ("train_xshot_occurrence", ["--dataset", "d"]),
        ("train_xcross_attempt", ["--dataset", "d", "--probe-dataset", "p"]),
    ],
)
def test_the_stage_trainers_key_the_extraction_on_the_live_corpus_identity(name, argv):
    """ADR-128: the whole-corpus feature cache and its fingerprint are gone (they replaced a constant
    ``"schema-v2"`` token, ADR-050). The CORPUS now keys the extract stage's TOKEN (`uses_dataset`), so a
    changed corpus is a new generation -- never a constant token, and never a fingerprint of a second copy
    of the selection rule."""
    import ast
    import importlib

    from scripts import _stages
    from tests.scripts._stage_harness import params_from_args

    m = importlib.import_module(f"scripts.{name}")
    src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assigned = {
        t.id
        for node in ast.parse(src).body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "_CACHE_FINGERPRINT" not in assigned
    assert m.PIPELINE.stage("extract").uses_dataset
    params = params_from_args(m.PIPELINE, [*argv, "--arms-json", "{}"])
    _, a = _stages.compute_tokens(m.PIPELINE, params, "c", {"name": "d", "revision": "r1"})
    _, b = _stages.compute_tokens(m.PIPELINE, params, "c", {"name": "d", "revision": "r2"})
    assert a["extract"] != b["extract"], "a changed corpus must be a new extract generation"


def test_selection_is_single_sourced_between_fingerprint_and_extraction():
    """``select_match_ids`` and ``load_matches`` must apply the SAME allowlist/cap rule. Two copies
    would let the fingerprint describe a corpus the extraction never loaded -- the cache would then
    be validated against a fiction."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from _loader_pining import _wanted_for_provider

    manifest = ["a", "b", "c", "d"]
    assert _wanted_for_provider(manifest, "gs", None, None) == ["a", "b", "c", "d"]
    assert _wanted_for_provider(manifest, "gs", None, 2) == ["a", "b"]
    assert _wanted_for_provider(manifest, "gs", {"gs": ["c", "d"]}, None) == ["c", "d"]
    assert _wanted_for_provider(manifest, "gs", {"gs": ["c", "d"]}, 1) == ["c"]
    # a provider absent from the allowlist falls back to the full manifest, not to empty
    assert _wanted_for_provider(manifest, "sk", {"gs": ["c"]}, None) == ["a", "b", "c", "d"]


def test_unclassified_provider_fails_BEFORE_any_fitting(monkeypatch):
    """Spy the fit: without this the test passes on the pre-existing mid-run raise and proves
    nothing. The value of this change is entirely in WHEN it fires."""
    import pytest as _pytest

    import silly_kicks.tracking._ghost_gk as gg

    calls = {"n": 0}
    monkeypatch.setattr(gg.GhostGkModel, "fit", lambda self, *a, **k: calls.__setitem__("n", 1))

    t = _load("train_ghost_gk")
    with _pytest.raises(ValueError, match="provider"):
        t.validate_corpus_providers(["gradientsports", "not_a_provider"])
    assert calls["n"] == 0


def test_validate_provider_is_shared_not_duplicated():
    """Two copies of the membership set drift the moment a provider is added -- silently, because
    an unclassified provider only surfaces deep inside a run.

    Pins the SINGLE SOURCE: `validate_provider` lives in `_provider_visibility` (moved out of
    `_ghost_gk` under the id_compat clean-break precedent), so this test imports from there."""
    import pytest as _pytest

    import silly_kicks.tracking._provider_visibility as pv

    assert callable(pv.validate_provider)
    pv.validate_provider("gradientsports")
    with _pytest.raises(ValueError, match="provider"):
        pv.validate_provider("nope")


def test_keeper_detection_mask_still_rejects_an_unknown_provider():
    """The extraction of `validate_provider` must be behaviour-preserving at the original site --
    moving a check earlier must not mean removing it from where it already was."""
    import pandas as pd
    import pytest as _pytest

    from silly_kicks.tracking._ghost_gk import keeper_detection_mask

    with _pytest.raises(ValueError, match="provider"):
        keeper_detection_mask(pd.Series([True, False]), provider="nope")


# ---------------------------------------------------------------------------
# Layer 2 (ADR: detection-aware visibility guardrails): the ghost-trainer consume-time pre-flight
# fails on a detection-aware shard whose `visibility` was discarded, BEFORE any extraction/fit --
# via parquet null_count metadata, so a MIXED corpus is caught, not just a systematic one.
# Co-located here (review S2) beside test_unclassified_provider_fails_BEFORE_any_fitting.
# ---------------------------------------------------------------------------


def _write_shard(path, *, provider, visibility):
    import pandas as pd

    n = len(visibility) if visibility is not None else 2
    data = {"source_provider": [provider] * n, "x": [float(i) for i in range(n)]}
    if visibility is not None:
        data["visibility"] = visibility
    pd.DataFrame(data).to_parquet(path)
    return path


def test_validate_corpus_visibility_raises_before_fit_on_all_null_skillcorner(tmp_path, monkeypatch):
    """Review B3: an all-null-visibility SkillCorner shard fails the pre-flight, and fit is never
    reached (spy call-count 0)."""
    import pandas as pd
    import pytest as _pytest

    import silly_kicks.tracking._ghost_gk as gg

    shard = _write_shard(
        tmp_path / "skillcorner__m1.parquet",
        provider="skillcorner",
        visibility=pd.array([None, None], dtype="boolean"),
    )

    calls = {"n": 0}
    monkeypatch.setattr(gg.GhostGkModel, "fit", lambda self, *a, **k: calls.__setitem__("n", 1))

    t = _load("train_ghost_gk")
    with _pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        t.validate_corpus_visibility({shard: "skillcorner"})
    assert calls["n"] == 0


def test_validate_corpus_visibility_ok_on_native_skillcorner(tmp_path):
    shard = _write_shard(tmp_path / "skillcorner__m2.parquet", provider="skillcorner", visibility=[True, False, True])
    t = _load("train_ghost_gk")
    assert t.validate_corpus_visibility({shard: "skillcorner"}) is None


def test_validate_corpus_visibility_noop_for_fully_observed(tmp_path):
    # A fully-observed provider legitimately carries no `visibility` column -> no-op (not a raise).
    shard = _write_shard(tmp_path / "gradientsports__m3.parquet", provider="gradientsports", visibility=None)
    t = _load("train_ghost_gk")
    assert t.validate_corpus_visibility({shard: "gradientsports"}) is None


def test_validate_corpus_visibility_raises_when_column_absent_for_skillcorner(tmp_path):
    # M2 (consume side): a detection-aware shard that DROPPED the column is a discarding regression.
    shard = _write_shard(tmp_path / "skillcorner__m4.parquet", provider="skillcorner", visibility=None)
    import pytest as _pytest

    t = _load("train_ghost_gk")
    with _pytest.raises(ValueError, match="visibility"):
        t.validate_corpus_visibility({shard: "skillcorner"})


def test_validate_corpus_visibility_catches_a_mixed_corpus(tmp_path):
    """Review S3 / the load-bearing 'one question': reading EVERY detection-aware shard (not sampling
    one) catches a corpus where only a TAIL shard was kloppy-rebuilt."""
    import pandas as pd
    import pytest as _pytest

    good = _write_shard(tmp_path / "skillcorner__ok.parquet", provider="skillcorner", visibility=[True, False])
    bad = _write_shard(
        tmp_path / "skillcorner__bad.parquet",
        provider="skillcorner",
        visibility=pd.array([None, None], dtype="boolean"),
    )
    t = _load("train_ghost_gk")
    with _pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        t.validate_corpus_visibility({good: "skillcorner", bad: "skillcorner"})


def test_main_calls_validate_corpus_visibility_before_extraction():
    """Wiring + ordering: the trainer must call validate_corpus_visibility, and BEFORE the per-frame
    keeper_detection_mask extraction / the model fit -- else 'fires before extraction' is not
    actually guaranteed by the pre-flight function raising in isolation.

    ADR-128: the order is the stage DAG's. The `validate` stage's work calls the pre-flight, and every
    stage whose work extracts or fits needs `validate` (transitively), so the executor completes
    `validate` before any of them can start."""
    import ast

    t = _load("train_ghost_gk")
    # ADR-129: the reusable stages (validate/extract) moved to scripts/_ghost_gk_stages.py; parse both.
    funcs: dict = {}
    for src in ("train_ghost_gk.py", "_ghost_gk_stages.py"):
        for n in ast.walk(ast.parse((REPO / "scripts" / src).read_text(encoding="utf-8"))):
            if isinstance(n, ast.FunctionDef):
                funcs[n.name] = n

    def _calls(fn, pred):
        return any(isinstance(n, ast.Call) and pred(n) for n in ast.walk(funcs[fn]))

    def _is_visibility(n):
        return isinstance(n.func, ast.Name) and n.func.id == "validate_corpus_visibility"

    def _extracts_or_fits(n):
        return (isinstance(n.func, ast.Name) and n.func.id == "keeper_detection_mask") or (
            isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
        )

    assert _calls("_validate_work", _is_visibility), "the validate stage must call validate_corpus_visibility"

    def _needs_validate(name):
        return any(n == "validate" or _needs_validate(n) for n in t.PIPELINE.stage(name).needs)

    later = [st.name for st in t.PIPELINE.stages if _calls(st.work.__name__, _extracts_or_fits)]
    assert later, "expected keeper_detection_mask / .fit in the stage work functions"
    assert {"extract", "cv", "fit"} <= set(later)
    assert all(_needs_validate(n) for n in later), "validate_corpus_visibility must run BEFORE extraction/fit"


# ---------------------------------------------------------------------------
# Ghost-GK shard packing (ADR-052): one game's features + labels + side arrays as ONE tidy frame
# ---------------------------------------------------------------------------


def test_ghost_shard_pack_unpack_round_trips_features_labels_and_side_arrays():
    """The migration's load-bearing claim: sharding changes WHERE the training matrix is held, not
    WHAT it is.

    The deliberately-extra feature column is the point of the test, not padding: `_unpack`
    reconstructs the feature frame by REMOVAL, and a version that selected
    ``GHOST_GK_FEATURE_NAMES`` instead would pass every other assertion here while silently
    training a narrower model the day the extractor widens.
    """
    import pandas as pd

    t = _load("train_ghost_gk")
    feats = pd.DataFrame({"ball_x": [1.0, 2.0], "phase": [0, 1], "a_future_feature": [7.0, 8.0]})
    labs = pd.DataFrame({"gk_x": [3.0, 4.0], "gk_y": [5.0, 6.0]})

    packed = t._pack(feats, labs, game_id="g1", provider="skillcorner", keepers=pd.Series(["k1", "k2"]))
    out_f, out_l, groups, provs, keepers = t._unpack(packed)

    pd.testing.assert_frame_equal(out_f, feats)
    pd.testing.assert_frame_equal(out_l, labs)
    assert groups.tolist() == ["g1", "g1"]
    assert provs.tolist() == ["skillcorner", "skillcorner"]
    assert keepers.tolist() == ["k1", "k2"]
    # Dtype, not just contents: `keepers` feeds the keeper-grouped CV domain and `provider_labels`
    # is the stratification target, so a storage decision must not change what they see.
    assert keepers.dtype == object


def test_ghost_shard_packing_REFUSES_a_feature_that_shadows_a_reserved_column():
    import pandas as pd
    import pytest as _pytest

    t = _load("train_ghost_gk")
    labs = pd.DataFrame({"gk_x": [3.0], "gk_y": [5.0]})
    for shadow in ("_provider", "_lab_gk_x"):
        feats = pd.DataFrame({"ball_x": [1.0], shadow: [99.0]})
        with _pytest.raises(ValueError, match="collide"):
            t._pack(feats, labs, game_id="g", provider="p", keepers=pd.Series(["k"]))


def test_the_collision_GUARD_is_not_decorative():
    """Non-vacuity. Pack the same shadowed frame WITHOUT the guard and the feature is gone: the
    round trip drops it as if it were the side column it is named after, and the model is fitted on
    a silently narrower matrix. That is the failure the guard converts into a raise."""
    import pandas as pd

    t = _load("train_ghost_gk")
    feats = pd.DataFrame({"ball_x": [1.0], "_provider": [99.0]})
    labs = pd.DataFrame({"gk_x": [3.0], "gk_y": [5.0]})
    unchecked = pd.concat([feats, labs.add_prefix(t._LABEL_PREFIX)], axis=1).assign(
        _game_id="g", _provider="skillcorner", _keeper="k"
    )

    out_f, _, _, provs, _ = t._unpack(unchecked)

    assert "_provider" not in out_f.columns, "the shadowed feature survived -- rewrite this test"
    assert provs.tolist() == ["skillcorner"], "and it was read as the side column, not as a feature"


def test_the_shipped_feature_names_pass_the_collision_guard():
    """The other side of the guard: it must be silent on the real 26, or it is unusable."""
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    _load("train_ghost_gk")._assert_no_column_collision(GHOST_GK_FEATURE_NAMES)


def test_ghost_bias_means_are_reconstructed_from_summed_counters():
    """The selection-bias diagnostic is pooled as (sum, count) so it survives a partial resume.
    An empty population must report NaN, exactly as `np.mean([])` did -- never 0.0, which would
    read as a measured absence of bias."""
    import math

    t = _load("train_ghost_gk")
    assert t._mean_from_counters(10.0, 4) == 2.5
    assert math.isnan(t._mean_from_counters(0.0, 0))
