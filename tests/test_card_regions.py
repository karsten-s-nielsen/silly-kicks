"""Generated card regions equal their re-render (ADR-128 spec 5.7); markers never reach the Hub (P7)."""

from pathlib import Path

import pytest

from scripts._hub_publish import card_bytes
from scripts.hf_jobs import manifest as M

MAN = M.load()
CARDS = sorted(
    {a.card for a in MAN.artifacts.values() if a.card}
    | {"silly_kicks/tracking/_receiver_weights/default/MODEL_CARD.md"}
)


@pytest.mark.parametrize("card", CARDS)
def test_regions_are_fresh(card):
    assert not M.stale_regions(Path(card), MAN)


@pytest.mark.parametrize("card", CARDS)
def test_every_card_has_its_required_regions(card):
    names = {m.group(1) for m in M.REGION.finditer(Path(card).read_text(encoding="utf-8"))}
    assert names >= M.required_regions(card, MAN), card


def test_card_bytes_strip_markers(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("A <!-- generated:hub_note -->B<!-- /generated:hub_note --> C\n", encoding="utf-8")
    assert card_bytes(p) == b"A B C\n"


def test_a_stale_region_is_detected(tmp_path):
    card = Path(CARDS[0])
    text = card.read_text(encoding="utf-8")
    m = next(M.REGION.finditer(text))
    poisoned = text[: m.start(2)] + "STALE" + text[m.end(2) :]
    p = tmp_path / card.name
    p.write_text(poisoned, encoding="utf-8")
    assert M.stale_regions(p, MAN, card_id=str(card).replace("\\", "/"))


def test_a_stale_table_number_is_detected(tmp_path):
    """A mirror card's row region is stale when ONE number differs from its bundle, wherever else that
    number may appear in the card (the CCC-PLAN-33 anti-vacuity, now on the regions)."""
    card = next(c for c in CARDS if c.endswith("ghost-outfield-v1-model-card.md"))
    text = Path(card).read_text(encoding="utf-8")
    m = next(r for r in M.REGION.finditer(text) if r.group(1).startswith("row:"))
    first = M._NUM.search(m.group(2))
    assert first is not None
    cell = m.group(2)[: first.start()] + "9999" + m.group(2)[first.end() :]
    p = tmp_path / Path(card).name
    p.write_text(text[: m.start(2)] + cell + text[m.end(2) :], encoding="utf-8")
    assert m.group(1) in M.stale_regions(p, MAN, card_id=card)


def test_every_card_belongs_to_exactly_one_artifact():
    """A card names ONE manifest artifact, so its regions render from one artifact's facts."""
    for card in CARDS:
        assert sum(a.card == card for a in MAN.artifacts.values()) == 1, card
