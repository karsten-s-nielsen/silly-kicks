import json

from silly_kicks.feature_glossary import (
    FEATURE_GLOSSARY,
    GLOSSARY_SCHEMA_VERSION,
    FeatureColumn,
    dump_glossary,
    emitting_module_is_importable,
    glossary_to_json,
    undocumented_columns,
)


def test_entry_shape_defaults():
    fc = FeatureColumn(name="x", definition="d", unit="metres", emitting_module="silly_kicks.tracking._packing")
    assert fc.attribution is None and fc.higher_is_better is None
    assert isinstance(FEATURE_GLOSSARY, dict)


def test_undocumented_columns():
    assert undocumented_columns(["definitely_not_a_real_column"]) == {"definitely_not_a_real_column"}


def test_json_is_pure_and_versioned():
    payload = json.loads(glossary_to_json())
    assert payload["schema_version"] == GLOSSARY_SCHEMA_VERSION
    assert "columns" in payload


def test_dump_glossary_writes(tmp_path):
    p = tmp_path / "g.json"
    dump_glossary(p)
    assert json.loads(p.read_text())["schema_version"] == GLOSSARY_SCHEMA_VERSION


def test_emitting_module_is_importable():
    assert emitting_module_is_importable("silly_kicks.tracking._packing")
    assert not emitting_module_is_importable("silly_kicks.tracking._does_not_exist_xyz")
