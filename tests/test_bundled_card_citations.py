"""Every file that a bundled model card cites exists beside the card (HFU fit-units spec C6).

The defect this guards against: the xsuccess and expected_passing cards cited ``metrics.json`` while their
bundles held none (fixed in 4.140.0, when the launch-7 adoption shipped the files). No gate looked at a cited
file. The population is every ``MODEL_CARD.md`` under ``silly_kicks/``, derived from the tree and asserted
exactly (ADR-056): a card is the thing that cites, so a weights directory without a card has nothing to check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: The bundled cards of 4.140.0. A new bundle with a card must be added here (the population test fails first).
_EXPECTED_CARDS: tuple[str, ...] = (
    "silly_kicks/corners/_routine_weights/default/MODEL_CARD.md",
    "silly_kicks/corners/_run_zone_weights/default/MODEL_CARD.md",
    "silly_kicks/expected_passing/weights/MODEL_CARD.md",
    "silly_kicks/tracking/_receiver_weights/default/MODEL_CARD.md",
    "silly_kicks/win_probability/weights/MODEL_CARD.md",
    "silly_kicks/xsuccess/weights/MODEL_CARD.md",
    "silly_kicks/xtgk/_retention_weights/default/MODEL_CARD.md",
    "silly_kicks/xtgk/_retention_weights/skillcorner/MODEL_CARD.md",
)

#: A back-ticked token that is a bare bundle file name: ``SHA256SUMS`` or a name with a bundle extension.
_CITED = re.compile(r"`((?:SHA256SUMS)|(?:[A-Za-z0-9_.-]+\.(?:json|npz|md)))`")

#: ``{card posix path: {file name: reason}}``: a name that a card cites from ANOTHER place. Empty today.
_CITED_ELSEWHERE: dict[str, dict[str, str]] = {}


def _cards() -> list[Path]:
    return sorted((_ROOT / "silly_kicks").rglob("MODEL_CARD.md"))


def _rel(card: Path) -> str:
    return card.relative_to(_ROOT).as_posix()


def missing_citations(card: Path) -> set[str]:
    """The bundle file names that ``card`` cites and that do not exist beside it."""
    cited = set(_CITED.findall(card.read_text(encoding="utf-8")))
    return {name for name in cited if not (card.parent / name).is_file()}


def test_the_card_population_is_exact():
    assert [_rel(c) for c in _cards()] == sorted(_EXPECTED_CARDS)


@pytest.mark.parametrize("card", _EXPECTED_CARDS)
def test_every_cited_file_exists(card):
    assert missing_citations(_ROOT / card) <= set(_CITED_ELSEWHERE.get(card, {}))


def test_no_stale_exemption():
    for card, names in _CITED_ELSEWHERE.items():
        assert set(names) <= missing_citations(_ROOT / card), f"{card}: an exemption that the card no longer needs"


def test_the_gate_reads_real_citations():
    """Non-vacuity: the scan finds cited files in the real cards (six of the eight cite at least one)."""
    cited = {c: set(_CITED.findall((_ROOT / c).read_text(encoding="utf-8"))) for c in _EXPECTED_CARDS}
    assert "metrics.json" in cited["silly_kicks/xsuccess/weights/MODEL_CARD.md"]
    assert "metrics.json" in cited["silly_kicks/expected_passing/weights/MODEL_CARD.md"]
    assert sum(bool(v) for v in cited.values()) >= 6


def test_a_card_that_cites_a_missing_file_fails(tmp_path):
    """Detection: the 4.139 defect shape, a card that cites `metrics.json` in a bundle without one."""
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("Per-type reliability is in `metrics.json`; the weights are `model.json`.\n", encoding="utf-8")
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    assert missing_citations(card) == {"metrics.json"}
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    assert missing_citations(card) == set()
