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
