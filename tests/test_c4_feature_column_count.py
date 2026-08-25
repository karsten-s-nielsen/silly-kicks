"""The C4 DSL states a count of derived feature columns; nothing pinned it to the code.

The sibling gate ``test_c4_aggregator_count.py`` pins the "N action-coupled aggregators" sentence to
``tracking.__all__``. The "N derived feature columns" sentence in the SAME DSL box had no such pin,
and it drifted: the DSL read 341 while ``feature_glossary.FEATURE_GLOSSARY`` already held 349 (a prior
release added columns without updating the box). Prose beside a computed value goes stale; the
computation does not -- so this gate derives the number and asserts the DSL matches.

Decision: ADR-068 final-review.
"""

from __future__ import annotations

import pathlib
import re

from silly_kicks.feature_glossary import FEATURE_GLOSSARY

_DSL = pathlib.Path(__file__).resolve().parents[1] / "docs" / "c4" / "architecture.dsl"


def test_the_dsl_feature_column_count_matches_the_glossary():
    expected = len(FEATURE_GLOSSARY)
    # Meta-assertion: a broken import/registry would make this gate pass vacuously.
    assert expected >= 100, f"glossary discovery looks broken, found {expected} entries"

    text = _DSL.read_text(encoding="utf-8")
    found = re.search(r"(\d+) derived feature columns", text)
    assert found is not None, "docs/c4/architecture.dsl no longer states a feature-column count"
    assert int(found.group(1)) == expected, (
        f"architecture.dsl says {found.group(1)} derived feature columns; "
        f"feature_glossary.FEATURE_GLOSSARY holds {expected}. Update the glossary container "
        f"description AND regenerate docs/c4/architecture.html via the c4 skill."
    )
