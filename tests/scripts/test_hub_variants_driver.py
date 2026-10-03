"""Hub smoke (combined-cycle-completion spec section 9): exact population, fail-closed load, finite scores."""

import math

import numpy as np
import pytest

from scripts import validate_hub_variants as H


def test_registry_covers_the_org_exactly():
    listed = set(H.HUB_REGISTRY)
    assert H.check_population(listed) is None
    with pytest.raises(SystemExit, match="unregistered"):
        H.check_population(listed | {"silly-kicks/new-model-v1"})
    with pytest.raises(SystemExit, match="missing"):
        H.check_population(listed - {"silly-kicks/ghost-gk-v1"})


def test_registry_classifies_nine_frame_geometry_repos():
    roles = [role for _cls, role in H.HUB_REGISTRY.values()]
    assert len(H.HUB_REGISTRY) == 10
    # spec 0.10 (rev 4 correction): ghost-gk-v1 is the Hub-only `full` variant, NOT a mirror.
    assert (roles.count("hf_only"), roles.count("mirror"), roles.count("event_only")) == (5, 4, 1)
    assert H.HUB_REGISTRY["silly-kicks/ghost-gk-v1"][1] == "hf_only"


def test_card_source_covers_the_registry_exactly():
    from scripts._hub_publish import CARD_SOURCE

    assert set(CARD_SOURCE) == set(H.HUB_REGISTRY)


def test_readme_matches_card_is_line_ending_blind_and_content_exact(tmp_path):
    from scripts._hub_publish import CARD_SOURCE

    repo = "silly-kicks/xsuccess-v1"
    card = tmp_path / CARD_SOURCE[repo]
    card.parent.mkdir(parents=True)
    card.write_bytes(b"# card\r\nline\r\n")  # a Windows (core.autocrlf=true) checkout
    hub = tmp_path / "hub_README.md"
    hub.write_bytes(b"# card\nline\n")
    assert H.readme_matches_card(repo, download=lambda r, f: str(hub), root=tmp_path) is True
    hub.write_bytes(b"# card\nother\n")
    assert H.readme_matches_card(repo, download=lambda r, f: str(hub), root=tmp_path) is False


def test_driver_frame_equals_the_test_frame():
    import pandas as pd

    from tests.test_bundled_models_load_on_float32_commit1 import _float32_canonical_frame

    a, b = H._float32_canonical_frame(), _float32_canonical_frame()
    pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])


@pytest.mark.parametrize(
    "cls_name", ["XShotOccurrenceModel", "XCrossAttemptModel", "GhostGkModel", "GhostOutfieldModel"]
)
def test_score_fn_runs_on_the_bundled_variants(cls_name):
    """Offline (no network): the exact serve lambdas the Hub smoke uses score the bundled `default`
    variant of each class to non-empty finite values (B r2 C1: a wrong signature fails CI, not the DGX)."""
    import silly_kicks.tracking as T

    scores = H._score_fn(cls_name)(getattr(T, cls_name).from_variant("default"))
    vals = np.asarray(scores, dtype=float).tolist()
    assert vals and all(math.isfinite(v) for v in vals)


def test_smoke_reports_non_finite_scores_as_failure():
    class FakeModel:
        training_commit = "abc"

    def fake_from_hub(repo_id):
        return FakeModel()

    out = H.smoke_repo("silly-kicks/x", "Fake", from_hub=fake_from_hub, score=lambda m: [float("nan")])
    assert out["loaded"] is True and out["finite"] is False


def _fake_world(tmp_path, *, stale_card=None, stale_mirror=None, refuse=None):
    """An offline Hub + repo root: every README equals its card and every mirror equals the wheel, except
    the one repo named stale (B r4 CCC-PLAN-32: the post-push gates must be tested both ways)."""
    import json

    from scripts._hub_publish import CARD_SOURCE

    hub, root = tmp_path / "hub", tmp_path / "root"
    for repo, rel in CARD_SOURCE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"# card " + repo.encode() + b"\n")
        d = hub / repo.replace("/", "__")
        d.mkdir(parents=True)
        (d / "README.md").write_bytes(b"# card " + repo.encode() + (b" STALE" if repo == stale_card else b"") + b"\n")
        (d / "metadata.json").write_text(json.dumps({"training_commit": ("f" if repo == stale_mirror else "c") * 40}))
    for rel in H.MIRROR_BUNDLE.values():
        m = root / "silly_kicks" / "tracking" / rel
        m.mkdir(parents=True)
        (m / "metadata.json").write_text(json.dumps({"training_commit": "c" * 40}))
    return {
        "listed": set(H.HUB_REGISTRY),
        "revision": lambda r: "rev",
        "download": lambda r, f: str(hub / r.replace("/", "__") / f),
        "load_model": lambda cls_name, repo_id: _refuse(repo_id) if repo_id == refuse else object(),
        "load_refusals": (_RefusalError,),
        "score_for": lambda cls_name: lambda m: [0.5],
        "prov": {"commit": "0" * 40, "dirty": False, "tree_state": "clean"},
        "root": root,
        "out": tmp_path / "out",
    }


def test_post_push_gates_pass_when_every_readme_and_mirror_matches(tmp_path):
    doc = H.run(**_fake_world(tmp_path), require_cards_match=True, require_mirrors_match_wheel=True)
    assert doc["cards_mismatched"] == [] and doc["mirrors_mismatched"] == [] and doc["all_finite"] is True
    assert (tmp_path / "out" / "hub_smoke.json").is_file()


def test_require_cards_match_fails_on_one_stale_readme(tmp_path):
    kw = _fake_world(tmp_path, stale_card="silly-kicks/xshot-occurrence-v1")
    assert H.run(**kw)["cards_mismatched"] == ["silly-kicks/xshot-occurrence-v1"]  # recorded, not gated, at C1
    with pytest.raises(SystemExit, match="differs from its in-repo card"):
        H.run(**kw, require_cards_match=True)


def test_require_mirrors_match_wheel_fails_on_a_stale_mirror(tmp_path):
    kw = _fake_world(tmp_path, stale_mirror="silly-kicks/ghost-outfield-v1")
    doc = H.run(**kw)
    assert doc["mirrors_mismatched"] == ["silly-kicks/ghost-outfield-v1"]
    assert doc["repos"]["silly-kicks/ghost-outfield-v1"]["wheel_training_commit"] == "c" * 40
    with pytest.raises(SystemExit, match="training_commit differs from the wheel"):
        H.run(**kw, require_mirrors_match_wheel=True)


def test_mirror_bundles_name_exactly_the_mirror_repos():
    assert set(H.MIRROR_BUNDLE) == {r for r, (_c, role) in H.HUB_REGISTRY.items() if role == "mirror"}


def test_main_threads_the_post_push_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(H, "_live_run", lambda **kw: seen.update(kw) or {})
    H.main(["--out", str(tmp_path), "--allow-dirty", "--require-cards-match", "--require-mirrors-match-wheel"])
    assert seen["require_cards_match"] is True and seen["require_mirrors_match_wheel"] is True
    assert seen["out"] == tmp_path and seen["prov"]["commit"]


class _RefusalError(Exception):
    """Stands in for the library's fail-closed load refusals (the per-model IntegrityError classes)."""


def _refuse(repo_id):
    raise _RefusalError(f"chirality mismatch for {repo_id}")


def test_smoke_records_a_fail_closed_load_refusal_instead_of_crashing():
    """A Hub artifact the library REFUSES to load (e.g. a pre-ADR-089 chirality fingerprint) is a recorded
    fact, never a crash that loses every other repo's result."""
    out = H.smoke_repo("silly-kicks/x", "Fake", from_hub=_refuse, score=lambda m: [0.5], refusals=(_RefusalError,))
    assert out["loaded"] is False and out["finite"] is False and out["n_scores"] == 0
    assert out["load_error"].startswith("_RefusalError: chirality mismatch")


def test_smoke_does_not_swallow_an_unexpected_error():
    def boom(repo_id):
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        H.smoke_repo("silly-kicks/x", "Fake", from_hub=boom, score=lambda m: [0.5], refusals=(_RefusalError,))


def test_a_refused_mirror_is_recorded_at_c1_and_gated_after_the_push(tmp_path):
    kw = _fake_world(tmp_path, refuse="silly-kicks/ghost-gk-sweeper-v1")
    doc = H.run(**kw)  # C1: recorded, not gated
    assert doc["load_refused"] == ["silly-kicks/ghost-gk-sweeper-v1"] and doc["all_finite"] is True
    with pytest.raises(SystemExit, match="refused to load"):
        H.run(**kw, require_mirrors_match_wheel=True)


def test_a_refused_hub_only_repo_always_fails(tmp_path):
    """The Hub-only repos are the published artifacts this cycle does not touch: a refusal there is a defect."""
    kw = _fake_world(tmp_path, refuse="silly-kicks/xshot-occurrence-v1")
    with pytest.raises(SystemExit, match="Hub-only"):
        H.run(**kw)
