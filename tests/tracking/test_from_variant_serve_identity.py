"""from_variant('public') must NOT serve the restricted sc_extended artifact (spec 2026-07-20 §8).

ADR-038's public-label gate operates at TRAINING time; this pins the SERVE path, which the trainer
gate structurally cannot observe.
"""

import json

import pytest

from silly_kicks.tracking._xcross_attempt import (
    _HUB_VARIANTS as XC_HUB,
)
from silly_kicks.tracking._xcross_attempt import (
    _VARIANT_ALIASES as XC_ALIASES,
)
from silly_kicks.tracking._xcross_attempt import (
    _XCROSS_WEIGHTS_ROOT,
)
from silly_kicks.tracking._xshot_occurrence import (
    _HUB_VARIANTS as XS_HUB,
)
from silly_kicks.tracking._xshot_occurrence import (
    _VARIANT_ALIASES as XS_ALIASES,
)
from silly_kicks.tracking._xshot_occurrence import (
    _XSHOT_WEIGHTS_ROOT,
)


@pytest.mark.parametrize("aliases", [XS_ALIASES, XC_ALIASES])
def test_public_resolves_to_bundled_default(aliases):
    assert aliases["public"] == "default"


@pytest.mark.parametrize("hub", [XS_HUB, XC_HUB])
def test_hub_variants_do_not_include_public_or_default(hub):
    """A name presented as reproducible must resolve inside the wheel, not the Hub."""
    assert hub.isdisjoint({"public", "default"})


@pytest.mark.parametrize("root", [_XSHOT_WEIGHTS_ROOT, _XCROSS_WEIGHTS_ROOT])
def test_bundled_default_declares_public_shipped_variant(root):
    """The alias is the literal truth, not a shim: default's metadata says shipped_variant=public."""
    meta = json.loads((root / "default" / "metadata.json").read_text())
    assert meta["shipped_variant"] == "public"


def test_xshot_public_and_default_return_the_same_bundled_object():
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    a = XShotOccurrenceModel.from_variant("public")
    b = XShotOccurrenceModel.from_variant("default")
    assert a is b  # same cached bundled instance; NOT a Hub download


def test_xcross_public_and_default_return_the_same_bundled_object():
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    a = XCrossAttemptModel.from_variant("public")
    b = XCrossAttemptModel.from_variant("default")
    assert a is b


# --- ADR-070: sc_extended_position_only is a SEPARATE Hub variant + repo (never overwrites faithful) ---


def test_hub_variants_include_faithful_and_position_only_keys():
    assert XS_HUB == {"sc_extended", "sc_extended_position_only"}
    assert XC_HUB == {"sc_extended", "sc_extended_position_only"}
    assert XS_HUB.isdisjoint({"public", "default"})  # reproducible-in-wheel rule still holds


def test_hub_repos_map_each_key_to_a_distinct_repo():
    from silly_kicks.tracking._xcross_attempt import _HF_REPO_ID as XC_REPO
    from silly_kicks.tracking._xcross_attempt import _HF_REPO_ID_POSITION_ONLY as XC_REPO_PO
    from silly_kicks.tracking._xcross_attempt import _HUB_REPOS as XC_REPOS
    from silly_kicks.tracking._xshot_occurrence import _HF_REPO_ID as XS_REPO
    from silly_kicks.tracking._xshot_occurrence import _HF_REPO_ID_POSITION_ONLY as XS_REPO_PO
    from silly_kicks.tracking._xshot_occurrence import _HUB_REPOS as XS_REPOS

    assert XS_REPOS == {"sc_extended": XS_REPO, "sc_extended_position_only": XS_REPO_PO}
    assert XC_REPOS == {"sc_extended": XC_REPO, "sc_extended_position_only": XC_REPO_PO}
    # distinct repos -> from_variant("sc_extended") can NEVER return the position-only artifact
    assert XS_REPO != XS_REPO_PO and XC_REPO != XC_REPO_PO
    assert "position-only" in XS_REPO_PO and "position-only" in XC_REPO_PO


@pytest.mark.parametrize(
    "mod_name, cls_name",
    [("_xshot_occurrence", "XShotOccurrenceModel"), ("_xcross_attempt", "XCrossAttemptModel")],
)
def test_from_variant_routes_each_hub_key_to_its_own_repo(monkeypatch, mod_name, cls_name):
    """The whole safety story (ADR-070): a caller asking for `sc_extended` hits the FAITHFUL repo,
    never the position-only one, so it can never silently receive a velocity-less model."""
    import importlib

    mod = importlib.import_module(f"silly_kicks.tracking.{mod_name}")
    cls = getattr(mod, cls_name)
    captured: list = []
    monkeypatch.setattr(cls, "from_hub", classmethod(lambda c, repo_id=None: (captured.append(repo_id), object())[1]))
    mod._VARIANT_CACHE.clear()
    cls.from_variant("sc_extended")
    mod._VARIANT_CACHE.clear()
    cls.from_variant("sc_extended_position_only")
    assert captured == [mod._HUB_REPOS["sc_extended"], mod._HUB_REPOS["sc_extended_position_only"]]


def test_publish_verify_only_is_feature_set_aware_for_a_position_only_artifact():
    """The publish verify-sample must match the artifact's feature_set (ADR-070): a hard-coded faithful
    (27-col) sample would raise an xgboost feature-count mismatch on the bundled position-only fit."""
    import subprocess
    import sys

    art = _XSHOT_WEIGHTS_ROOT / "position_only"
    r = subprocess.run(  # noqa: S603 -- controlled: sys.executable + a fixed in-repo script, no untrusted input
        [sys.executable, "scripts/publish_xshot_occurrence.py", "--artifact-dir", str(art), "--verify-only"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"verify-only crashed on a position-only artifact:\n{r.stdout}\n{r.stderr}"
    assert "feature_set=position_only" in r.stdout
