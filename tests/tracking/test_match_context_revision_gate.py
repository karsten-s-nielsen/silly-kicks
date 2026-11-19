"""A gate for the hand-bumped tag ``match_context_revision``.

The fixed probe of ``tests/tracking/_match_context_probe.py`` runs the score lookup and the phase lookup (the two
builders of every score and phase input of the ghost-GK, ghost-outfield and xCross models) on fixed tables and fixed
queries. Their outputs are pinned for each revision. A change of those outputs with no bump of the tag fails here.
"""

from __future__ import annotations

from tests.tracking._match_context_probe import PHASE_QUERIES, RUNNING_QUERIES, SCORE_QUERIES, probe_outputs

#: The pinned probe outputs per ``match_context_revision`` (readable values: one per query, in query order).
#: score_* = home score minus away score at (period, time); phase = 0 open play, 1 set piece, 2 goal kick.
PINS: dict[float, dict[str, list]] = {
    1.0: {
        # SCORE_QUERIES: (1,50) (1,100) (1,899) (1,950) (2,10) (2,60) (2,650) (3,20) (3,40) (4,5) (5,100)
        "score_goal_rows": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        # RUNNING_QUERIES: (1,15) (1,50) (1,90) (2,50) (2,100) (5,40): a period-5 frame takes the end of period 4;
        # (4,5): before period 5 the effect rule holds (a change on the last row takes no effect)
        "score_running": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        "score_fallback": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # PHASE_QUERIES: (1,99) (1,105) (1,110) (1,111) (1,203) (1,207) (1,212) (1,302) (1,405) (2,55) (2,105) (1,55)
        #                (1,100) (1,200): at the set piece's own time (the kick frame)
        "phase": [0, 1, 1, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 2],
    },
}

_QUERIES = {
    "score_goal_rows": SCORE_QUERIES,
    "score_running": RUNNING_QUERIES,
    "score_fallback": RUNNING_QUERIES,
    "phase": PHASE_QUERIES,
}


def test_the_probe_outputs_equal_the_pin_of_the_current_revision():
    from silly_kicks.tracking._feature_contract import REVISION_CONTRACT_KEYS

    revision = REVISION_CONTRACT_KEYS["match_context_revision"]
    assert revision in PINS, f"no probe pin for match_context_revision {revision}: add one (and refit)"
    got, want = probe_outputs(), PINS[revision]
    changed = {
        name: [(q, w, g) for q, w, g in zip(_QUERIES[name], want[name], got[name], strict=True) if w != g]
        for name in want
    }
    changed = {k: v for k, v in changed.items() if v}
    assert not changed, (
        "the score and phase inputs changed (query, pinned, now): "
        f"{changed}. Raise match_context_revision in REVISION_CONTRACT_KEYS, add the pin of the new revision here, "
        "and refit the ghost-GK, ghost-outfield and xCross models."
    )


def test_every_pin_covers_every_query():
    for revision, pin in PINS.items():
        assert {k: len(v) for k, v in pin.items()} == {k: len(q) for k, q in _QUERIES.items()}, revision
