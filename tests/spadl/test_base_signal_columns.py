"""Every converter emits the three TF-46 Arm 1 base signal columns (ADR-046 / decision D).

The columns are on base ``SPADL_COLUMNS``; ``_finalize_output`` strict projection raises ``KeyError`` if a
converter omits one. This is the producer-side guarantee behind the tolerant training-exclusion helper: the
column is always present on converter output, so a marked row can never leak unnoticed (decision E).

Parametrized over every ``silly_kicks/spadl/*.py`` ``convert_to_actions`` via the shared
``_converter_cases.CONVERTER_CASES`` registry (statsbomb/opta/wyscout/sportec/metrica/skillcorner/
gradientsports/kloppy). The anti-rot meta-test asserts that registry covers the full converter surface, so a
new provider cannot silently escape the gate.
"""

import pytest

from tests.spadl._converter_cases import CONVERTER_CASES, discover_converter_modules

_BASE_SIGNAL_COLUMNS = ("result_source", "restart_evidence", "type_source")


@pytest.mark.parametrize("provider", sorted(CONVERTER_CASES))
def test_converter_emits_base_signal_columns(provider):
    case = CONVERTER_CASES[provider]
    actions = case.run(case.build_input())
    for col in _BASE_SIGNAL_COLUMNS:
        assert col in actions.columns, f"{provider} output is missing base signal column {col!r}"


def test_converter_cases_cover_every_converter():
    """Anti-rot: the parametrized registry must cover every ``convert_to_actions`` module.

    A new provider converter that does not appear in ``CONVERTER_CASES`` fails here rather than
    silently escaping the per-converter emit gate above.
    """
    assert set(CONVERTER_CASES) == discover_converter_modules()
