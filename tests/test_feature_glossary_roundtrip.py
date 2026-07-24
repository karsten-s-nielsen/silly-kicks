import json

from silly_kicks.feature_glossary import FEATURE_GLOSSARY, GLOSSARY_SCHEMA_VERSION, dump_glossary
from silly_kicks.reporting import describe_level


def test_dump_reload_and_direction_flip(tmp_path):
    """dump -> reload JSON -> every entry survives -> describe_level over BOTH directions via real entries.

    Exercises the glossary-direction -> describe_level flip through real authored entries (one
    higher_is_better=True and one False), not just the True path -- proving the foundation is consumable.
    """
    p = tmp_path / "g.json"
    dump_glossary(p)
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == GLOSSARY_SCHEMA_VERSION
    assert set(payload["columns"]) == set(FEATURE_GLOSSARY)  # every entry survives the roundtrip

    higher = next(c for c, v in payload["columns"].items() if v["higher_is_better"] is True)
    lower = next(c for c, v in payload["columns"].items() if v["higher_is_better"] is False)
    # A strongly-positive z reads top-band for a higher-is-better column, bottom-band for lower-is-better.
    assert describe_level(2.0, higher_is_better=payload["columns"][higher]["higher_is_better"]) == "outstanding"
    assert describe_level(2.0, higher_is_better=payload["columns"][lower]["higher_is_better"]) == "poor"
