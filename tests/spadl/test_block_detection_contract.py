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


def test_cross_blocked_is_subset_of_open_play_cross_type():
    """CS-1 (TF-51): ``cross_blocked`` non-NA => SPADL type == 'cross' (open-play). It must be
    ``pd.NA`` on every ``corner_crossed`` / ``freekick_crossed``, so the bravery open-play denominator
    is well-defined. TEETH: the GS synthetic fixture carries non-NA ``cross_blocked`` on open-play
    crosses (verified non-vacuous), so this locks the invariant against future converter drift."""
    from silly_kicks.spadl import config as spadlconfig
    from tests.invariants._loaders import (
        load_gradientsports_synthetic,
        load_metrica_native_per_period,
        load_sportec_native_per_period,
        load_statsbomb,
        load_wyscout_2team_synthetic,
    )

    open_cross = spadlconfig.actiontype_id["cross"]
    saw_non_na = False
    for loader in (
        lambda: load_statsbomb(7298),
        load_sportec_native_per_period,
        load_metrica_native_per_period,
        load_wyscout_2team_synthetic,
        load_gradientsports_synthetic,
    ):
        loaded = loader()
        actions = loaded[0] if isinstance(loaded, tuple) else loaded
        if "cross_blocked" not in actions.columns:
            continue
        non_na = actions["cross_blocked"].notna()
        saw_non_na = saw_non_na or bool(non_na.any())
        offending = actions[non_na & (actions["type_id"] != open_cross)]
        assert offending.empty, (
            f"cross_blocked is non-NA on {len(offending)} non-open-play-cross rows "
            f"(types {sorted(offending['type_id'].unique())}); it must be pd.NA on set-piece crosses "
            f"(corner_crossed/freekick_crossed). Bravery's open-play denominator depends on this."
        )
    assert saw_non_na, "invariant is vacuous -- no fixture emitted a non-NA cross_blocked (add one with teeth)"
