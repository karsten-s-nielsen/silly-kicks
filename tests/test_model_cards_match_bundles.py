"""Model cards state the bundled artifacts they describe, each value in its own place
(combined-cycle spec 9, D9; B r4 CCC-PLAN-33: a value matched "anywhere" lets a stale table cell pass
whenever the right number survives in a history paragraph)."""

import json
import re
from pathlib import Path

import pytest

from silly_kicks import __version__

_CARDS = Path("docs/huggingface/model-cards")
_W = Path("silly_kicks/tracking")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_GK_ROWS = {
    "Held-out CV euclidean MAE": [
        ("cv_mae_euclidean_mean", ".3f"),
        *[(("per_provider_mae_euclidean", p), ".3f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "> 30 m high-sweeper stratum MAE": [("high_sweeper_stratum_mae_mean", ".2f")],
    "Training corpus": [("n_games", "d"), ("n_samples", "d")],
}
_OUTFIELD_ROWS = {
    "Held-out CV euclidean MAE": [
        ("cv_mae", ".2f"),
        *[(("cv_mae_by_provider", p), ".2f") for p in ("gradientsports", "skillcorner", "sportec")],
    ],
    "Per-possession CV MAE": [(("cv_mae_by_possession", p), ".2f") for p in ("in_possession", "out_of_possession")],
    "Per-slot CV MAE (slots 1&ndash;4)": [(("cv_mae_by_slot", s), ".2f") for s in ("1", "2", "3", "4")],
    "Training corpus": [("n_games", "d"), ("n_rows", "d")],
}
# The four Hub mirrors of wheel-bundled variants (spec 0.10): card -> (bundled dir, metrics-table rows).
_MIRRORS = {
    "ghost-gk-sweeper-v1": ("_ghost_gk_weights/sweeper", _GK_ROWS),
    "ghost-gk-sweeper-position-only-v1": ("_ghost_gk_weights/sweeper_position_only", _GK_ROWS),
    "ghost-outfield-v1": ("_ghost_outfield_weights/default", _OUTFIELD_ROWS),
    "ghost-outfield-position-only-v1": ("_ghost_outfield_weights/position_only", _OUTFIELD_ROWS),
}
# Hub-only cards that describe a wheel-bundled sibling: card -> the bundled dir it names.
_HF_ONLY = {
    "ghost-gk-v1": "_ghost_gk_weights/default",
    "xshot-occurrence-v1": "_xshot_weights/default",
    "xshot-occurrence-position-only-v1": "_xshot_weights/position_only",
    "xcross-attempt-v1": "_xcross_weights/default",
    "xcross-attempt-position-only-v1": "_xcross_weights/position_only",
}
_WHEEL_CARDS = ["_receiver_weights/default", "_gk_completion_weights/default", "_gk_completion_weights/skillcorner"]


def _card(name: str) -> str:
    return (_CARDS / f"{name}-model-card.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def _json(rel: str, name: str) -> dict:
    return json.loads((_W / rel / name).read_text(encoding="utf-8"))


def _get(doc: dict, key):
    return doc[key] if isinstance(key, str) else doc[key[0]][key[1]]


def _short(rel: str) -> str:
    return _json(rel, "metadata.json")["training_commit"][:7]


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


def _provenance_line(metrics: dict) -> str:
    """The ONE provenance line a wheel MODEL_CARD.md carries, built from its metrics.json."""
    fields = [f"`run_commit={metrics['run_commit'][:7]}`"]
    fields += [f"`{k}: {metrics[k]}`" for k in ("corpus_visibility", "artifact_label") if k in metrics]
    fields += [f"{metrics[k]} {unit}" for k, unit in (("n_matches", "matches"), ("n_rows", "rows")) if k in metrics]
    return f"**Provenance (silly-kicks {__version__}).** " + " · ".join(fields)


def test_card_population_is_exact():
    assert {p.name.removesuffix("-model-card.md") for p in _CARDS.glob("*-model-card.md")} == set(_MIRRORS) | set(
        _HF_ONLY
    )


@pytest.mark.parametrize("card", sorted(_MIRRORS))
def test_mirror_card_states_the_f1b_refit_paragraph(card):
    rel = _MIRRORS[card][0]
    head = f"**F1b float32-frame re-fit (silly-kicks {__version__} / ADR-106; `training_commit={_short(rel)}`).**"
    assert _card(card).count(head) == 1


@pytest.mark.parametrize(("card", "label"), [(c, lab) for c, (_rel, rows) in sorted(_MIRRORS.items()) for lab in rows])
def test_mirror_card_table_row_equals_the_bundle(card, label):
    rel, rows = _MIRRORS[card]
    metrics = _json(rel, "metrics.json")
    want = [format(_get(metrics, key), fmt) for key, fmt in rows[label]]
    assert _NUM.findall(_row_value(_card(card), label)) == want, (card, label)


@pytest.mark.parametrize("card", sorted(_HF_ONLY))
def test_hub_only_card_states_the_unchanged_note_for_its_bundled_sibling(card):
    rel = _HF_ONLY[card]
    note = (
        f"In silly-kicks {__version__} the wheel's bundled `{Path(rel).name}` was re-fit on float32-stored frames "
        f"(`training_commit={_short(rel)}`). This Hub artifact is unchanged: trained on float64 frames"
    )
    assert _card(card).count(note) == 1


@pytest.mark.parametrize("rel", _WHEEL_CARDS)
def test_wheel_model_card_states_its_provenance_line(rel):
    text = (_W / rel / "MODEL_CARD.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("**Provenance (silly-kicks") == 1
    assert _provenance_line(_json(rel, "metrics.json")) in text


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
