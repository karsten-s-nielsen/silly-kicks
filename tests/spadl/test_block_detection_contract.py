"""Cross-provider contract for the block-detection columns (TF-51 prereq).

Every converter emits `shot_blocked` and `cross_blocked` as dtype "boolean". Value-invariance of
the *other* columns is guaranteed by the additive-only edits (no existing column's code path is
touched) plus the existing golden/parity suites; the additivity test here only asserts *presence*,
which is the honest scope of a structural check.
"""


def _assert_block_columns(actions):
    for col in ("shot_blocked", "cross_blocked"):
        assert col in actions.columns, f"missing {col}"
        assert str(actions[col].dtype) == "boolean", f"{col} dtype {actions[col].dtype}"


def test_all_real_fixture_providers_emit_boolean_block_columns():
    # the committed-real-fixture providers, via the shared invariant loaders
    from tests.invariants._loaders import (
        load_metrica_native_per_period,
        load_sportec_native_per_period,
        load_statsbomb,
    )

    for loaded in (
        load_statsbomb(7298),
        load_sportec_native_per_period(),
        load_metrica_native_per_period(),
    ):
        actions = loaded[0] if isinstance(loaded, tuple) else loaded
        _assert_block_columns(actions)


def test_block_columns_present_and_additive(metrica_dataset):
    from silly_kicks.spadl import kloppy

    actions, _ = kloppy.convert_to_actions(metrica_dataset)
    assert {"shot_blocked", "cross_blocked"} <= set(actions.columns)
    # every pre-existing canonical column still present (additive: nothing dropped or renamed)
    for col in kloppy.KLOPPY_SPADL_COLUMNS:
        if col not in ("shot_blocked", "cross_blocked"):
            assert col in actions.columns
