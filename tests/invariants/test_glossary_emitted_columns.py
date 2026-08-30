"""Non-vacuity + correctness gate for the run-and-diff emitted-column harness (glossary Task 10).

This module is one of the auto-enumerating gates that sweep every aggregator on defaults, so it opts
out of the SyntheticEPVWarning/IgnoredSurfaceInputsWarning error-filter at module level (ADR-041) --
the OBSO family's synthetic-EPV notice fires while the tracking leg runs add_obso/add_pausa/
add_space_creation on default (no injected xt) config and is expected + irrelevant here.
"""

import pytest

from tests.invariants import glossary_emitted_columns as E

pytestmark = [
    pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning"),
    pytest.mark.filterwarnings("ignore::silly_kicks.tracking.IgnoredSurfaceInputsWarning"),
]


def test_union_has_known_tracking_columns():
    cols = E.emitted_columns()  # union across all legs, base-normalised
    assert "packing_made" in cols and "defensive_credit_net" in cols
    assert all(isinstance(c, str) for c in cols)


def test_each_leg_is_non_vacuous():
    # THE anti-lie guard: a stubbed leg (return set()) would silently under-cover with green CI. Each
    # leg must be non-empty with a known-column anchor. LIMITATION (honest): an anchor proves
    # non-empty + contains-that-column; a leg returning {anchor} + only HALF its real columns STILL
    # passes -- partial-leg holes are uncatchable without a second independent enumeration. Read
    # "non-vacuous", NOT "complete".
    # The per-leg functions return RAW slotted names (base-normalisation happens at the
    # emitted_columns() union), so normalise here before matching base-name anchors.
    from tests.invariants.glossary_emitted_columns import _base

    assert "packing_made" in {_base(c) for c in E._tracking_add_star_columns()}
    assert "pitch_control_at_target__spearman" in {_base(c) for c in E._xfns_columns()}
    assert "start_coord_source" in {_base(c) for c in E._spadl_enricher_columns()}  # add_restart_coordinates
    assert E._vaep_columns(), "vaep leg empty (stubbed?) -- anchor with a real xfns_default column"
    assert "rd_num_superiority" in {_base(c) for c in E._restdefense_columns()}  # TF-60 compute_rest_defense
