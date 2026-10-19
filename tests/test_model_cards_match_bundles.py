"""Model cards state the bundled artifacts they describe, each value in its own place
(combined-cycle spec 9, D9; B r4 CCC-PLAN-33: a value matched "anywhere" lets a stale table cell pass
whenever the right number survives in a history paragraph).

ADR-128 (spec 5.7): the re-fit EVENT each card states (version, ADR, basis) and every card's artifact
link (mirror, Hub-only wheel sibling, wheel card) come from the artifact manifest
(`docs/artifacts/manifest.toml`), not from constants here; the sentences are rendered by
`scripts.hf_jobs.manifest`. A card's provenance / re-fit line tracks the re-fit event per bundle, not
the live wheel version -- a no-retrain release (e.g. 4.129.0 / TF-58) does not re-stamp (that would
fabricate a re-fit that never happened). Region freshness is `tests/test_card_regions.py`; this file
keeps the per-card, per-row checks.
"""

import re
from pathlib import Path

import pytest

from scripts.hf_jobs import manifest as M

MAN = M.load()
_CARDS = Path("docs/huggingface/model-cards")
_NUM = re.compile(r"\d+(?:\.\d+)?")
# The four Hub mirrors of wheel-bundled variants (spec 0.10) and the Hub-only cards that describe a
# wheel-bundled sibling, by card name -> manifest artifact.
_HUB = {
    Path(a.card).name.removesuffix("-model-card.md"): a for a in MAN.artifacts.values() if a.kind == "hub" and a.card
}
_MIRRORS = {name: a for name, a in _HUB.items() if a.mirror_of}
_HF_ONLY = {name: a for name, a in _HUB.items() if a.wheel_sibling}
_WHEEL_CARDS = sorted(a.card for a in MAN.artifacts.values() if a.kind == "wheel" and a.card)


def _card_path(a: M.Artifact) -> str:
    assert a.card, a.id
    return a.card


def _card(name: str) -> str:
    return M.strip_markers((_CARDS / f"{name}-model-card.md").read_text(encoding="utf-8").replace("\r\n", "\n"))


def _row_value(text: str, label: str) -> str:
    """The value cell of the ONE table row whose label cell (bold stripped) is ``label``."""
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.lstrip().startswith("|")
    ]
    hits = [r[1] for r in rows if len(r) >= 2 and r[0].replace("*", "").strip() == label]
    assert len(hits) == 1, f"expected exactly one table row labelled {label!r}, found {len(hits)}"
    return hits[0]


def test_card_population_is_exact():
    assert {p.name.removesuffix("-model-card.md") for p in _CARDS.glob("*-model-card.md")} == set(_MIRRORS) | set(
        _HF_ONLY
    )


@pytest.mark.parametrize("card", sorted(_MIRRORS))
def test_mirror_card_states_the_refit_paragraph(card):
    assert _card(card).count(M.render_region("refit_head", _card_path(_MIRRORS[card]), MAN)) == 1


@pytest.mark.parametrize(
    ("card", "label"),
    [(c, lab) for c, a in sorted(_MIRRORS.items()) for lab in M.table_rows(MAN.artifacts[a.mirror_of or ""])],
)
def test_mirror_card_table_row_equals_the_bundle(card, label):
    want = M.row_values(MAN.artifacts[_MIRRORS[card].mirror_of or ""], label)
    assert _NUM.findall(_row_value(_card(card), label)) == want, (card, label)


@pytest.mark.parametrize("card", sorted(_HF_ONLY))
def test_hub_only_card_states_its_bundled_sibling_provenance(card):
    """The card carries the verifiable wheel-sibling provenance sentence exactly once."""
    assert _card(card).count(M.render_region("hub_note", _card_path(_HF_ONLY[card]), MAN)) == 1


@pytest.mark.parametrize("card", sorted(_HF_ONLY))
def test_hub_only_card_states_its_own_refit_event(card):
    """ADR-128: the Hub artifact's own re-fit event is a declared manifest expectation (its commit read
    from the published file), so the card states it once, rendered from the manifest."""
    assert _card(card).count(M.render_region("hub_refit", _card_path(_HF_ONLY[card]), MAN)) == 1


@pytest.mark.parametrize("card", _WHEEL_CARDS)
def test_wheel_model_card_states_its_provenance_line(card):
    text = M.strip_markers(Path(card).read_text(encoding="utf-8").replace("\r\n", "\n"))
    assert text.count("**Provenance (silly-kicks") == 1
    assert M.render_region("provenance", card, MAN) in text


def test_a_stale_table_cell_fails_even_when_the_right_number_is_elsewhere():
    """Anti-vacuity for CCC-PLAN-33: the row check must not be satisfied by a history paragraph."""
    card = (
        "| Metric | Value |\n|---|---|\n| Held-out CV euclidean MAE | **6.10 m** (per-provider: 6.14 / 5.92 / 6.33) |\n"
    )
    card += "\nHistory: the aggregate was 6.00 m.\n"
    assert _NUM.findall(_row_value(card, "Held-out CV euclidean MAE")) == ["6.10", "6.14", "5.92", "6.33"]
    assert _NUM.findall(_row_value(card, "Held-out CV euclidean MAE"))[0] != "6.00"
    with pytest.raises(AssertionError, match="exactly one table row"):
        _row_value(card + card, "Held-out CV euclidean MAE")  # a duplicated (stale) table is refused
