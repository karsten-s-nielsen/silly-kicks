"""The SB360 audit's state vocabulary -- declared ONCE, namespaced by kind.

Namespacing is not tidiness. An earlier revision of the design had an observation named
``raises`` and an adjudication named ``raises`` denoting DIFFERENT things, while ``raises_b``
appeared in two vocabularies denoting the SAME thing. A flat name set conflates the first and
cannot express that the second is deliberate.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

# --- call_outcome.* -------------------------------------------------------------------
CALL_OUTCOMES: frozenset[str] = frozenset({"raises_a", "raises_b", "both_succeeded"})

#: The one call outcome that is NOT a precedence rule: it is the gate into row classification.
ROW_CLASSIFICATION_PRECONDITION: str = "both_succeeded"

# --- row_class.* ----------------------------------------------------------------------
ROW_CLASSES: frozenset[str] = frozenset({"row_identical", "row_differs", "row_nan_a", "row_nan_b", "row_nan_both"})

# --- observation.* --------------------------------------------------------------------
OBSERVATIONS: frozenset[str] = frozenset(
    {
        "raises_a",
        "raises_b",
        "leg_b_declined",
        "no_signal",
        "all_nan",
        "partial_nan",
        "differs",
        "identical",
    }
)

#: Names shared between call_outcome and observation because the state passes straight
#: through unchanged. DECLARED, not incidental -- see test_shared_names_are_declared.
SHARED_NAMES: frozenset[str] = frozenset({"raises_a", "raises_b"})

# --- kind.* ---------------------------------------------------------------------------
KINDS: frozenset[str] = frozenset({"terminal_fixture_failure", "budgeted", "adjudicated"})

OBSERVATION_KIND: dict[str, str] = {
    "raises_a": "adjudicated",
    "raises_b": "terminal_fixture_failure",
    "leg_b_declined": "terminal_fixture_failure",
    "no_signal": "budgeted",
    "all_nan": "adjudicated",
    "partial_nan": "adjudicated",
    "differs": "adjudicated",
    "identical": "adjudicated",
}

# --- precedence -----------------------------------------------------------------------
#: (rank, observation). First match wins. Ranks 1-2 are mutually exclusive by their Level 1
#: definitions (raises_a is "Leg A raised, Leg B irrelevant"), so their relative order is
#: immaterial; 2 follows 1 so this cannot read as disagreeing with Level 1.
PRECEDENCE: tuple[tuple[int, str], ...] = (
    (1, "raises_a"),
    (2, "raises_b"),
    (3, "leg_b_declined"),
    (4, "no_signal"),
    (5, "all_nan"),
    (6, "partial_nan"),
    (7, "differs"),
    (8, "identical"),
)

#: Which precedence rules read each row class. ``row_nan_both`` is consumed by ``no_signal``
#: via the informative-row denominator: it is the class whose EXCLUSION defines that rule.
ROW_CLASS_CONSUMERS: dict[str, frozenset[str]] = {
    "row_identical": frozenset({"identical"}),
    "row_differs": frozenset({"differs"}),
    "row_nan_a": frozenset({"all_nan", "partial_nan"}),
    "row_nan_b": frozenset({"leg_b_declined"}),
    "row_nan_both": frozenset({"no_signal"}),
}

# --- adjudication.* -------------------------------------------------------------------
ADJUDICATIONS: frozenset[str] = frozenset(
    {"works", "silent_degrade", "differs_by_design", "honest_nan", "not_exercised", "raises"}
)

ADMISSIBLE_FROM: dict[str, frozenset[str]] = {
    "works": frozenset({"identical"}),
    "silent_degrade": frozenset({"differs", "partial_nan"}),
    "differs_by_design": frozenset({"differs", "partial_nan"}),
    "honest_nan": frozenset({"all_nan", "partial_nan"}),
    "not_exercised": frozenset({"no_signal"}),
    "raises": frozenset({"raises_a"}),
}

#: Adjudications that ALWAYS require a written rationale.
RATIONALE_ALWAYS: frozenset[str] = frozenset({"silent_degrade", "differs_by_design", "not_exercised"})

#: Adjudications that require a rationale only under a stated condition. Disjoint from
#: RATIONALE_ALWAYS, or the condition would be dead.
RATIONALE_CONDITIONAL: dict[str, str] = {
    # Loosening a tolerance converts `differs` into `identical`, which would otherwise
    # manufacture a rationale-free `works`. This is the half of that mitigation that lives in
    # the vocabulary; the other half surfaces the tolerance next to the adjudication.
    "works": "tolerance is non-default",
    "honest_nan": "observation is partial_nan",
}

# --- verdict_provenance.* -------------------------------------------------------------
#: Whether a BOUNDARY entry's verdict is SUBSTANTIVE (a velocity-consuming function whose own
#: handling moved the value) or STRUCTURAL (a function the axes cannot substantively reach:
#: frame-blind -> `identical`; downstream-of-a-refusing-seam -> `honest_nan`). Scoped to
#: BOUNDARY_ENTRY_POINTS so an empty UNAUDITABLE_BOUNDARY is not misread as end-to-end coverage
#: (ADR-053 amendment / spec Part 4).
VERDICT_PROVENANCE: frozenset[str] = frozenset({"substantive", "structural"})

# --- applicability.* ------------------------------------------------------------------
APPLICABILITY: frozenset[str] = frozenset({"region_support", "no_support", "support_data_defined"})
