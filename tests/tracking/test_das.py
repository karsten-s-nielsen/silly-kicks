"""TF-28 DAS facade + features integration (NATIVE engine, ADR-107/108).

Rewritten for the native engine: the pre-native suite tested the ``accessible-space`` seam
(`_to_das_coords` / `_prepare_frames` / `_call_simulation` / `_check_das_output_alignment` /
`_pin_attacking_direction` / `_has_simulatable_frame` / `_import_accessible_space`), all of which
are gone. The engine numerics live in ``test_das_engine_parity.py`` / ``test_das_pack.py`` /
``test_das_quadrature.py`` / ``test_xc_native.py``; the reference divergences in
``test_das_divergences.py``. This file covers the PUBLIC facade contract (``get_das`` /
``get_individual_das`` / ``get_xc`` / ``estimate_das_cost``) and the ``features`` integration
(``add_das`` / ``das_at_action`` / ``das_xfns``, the ``das_source`` provenance, the linked-frame
restriction).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das import DasUnscoreableError
from silly_kicks.tracking._das_taxonomy import (
    DAS_SOURCE_UNSCOREABLE_FRAME,
    DAS_SOURCE_VALUES,
)

# ---------------------------------------------------------------------------
# Fixtures: native DAS resolves direction from the GoalMap (keeper geometry, ADR-055),
# so scoring frames MUST carry both teams' keepers (Home near x=0, Away near x=105).
# ---------------------------------------------------------------------------


def _keeper_frames(frame_ids: tuple[int, ...] = (1,), *, poss: object = "Home") -> pd.DataFrame:
    """Ball + 5v5 per frame, each team with a keeper, so direction resolves.

    Home keeper near x=0 (defends 0, attacks 105); Away keeper near x=105. Home is forward
    (x 30-45), Away is deep (x 60-72), so the in-possession team carries the larger DAS.
    """
    rows: list[dict] = []
    for fid in frame_ids:
        t = float(fid)
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id="ball",
                team_id=None,
                is_ball=True,
                is_goalkeeper=False,
                x=45.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession=poss,
                ball_carrier_player_id="H0",
            )
        )
        for i in range(5):
            gk = i == 0
            hx, ax, yy = (3.0 if gk else 30.0 + i * 3), (102.0 if gk else 60.0 + i * 3), 20.0 + i * 3
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    time_seconds=t,
                    player_id=f"H{i}",
                    team_id="Home",
                    is_ball=False,
                    is_goalkeeper=gk,
                    x=hx,
                    y=yy,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession=poss,
                    ball_carrier_player_id="H0",
                )
            )
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    time_seconds=t,
                    player_id=f"A{i}",
                    team_id="Away",
                    is_ball=False,
                    is_goalkeeper=gk,
                    x=ax,
                    y=yy,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession=poss,
                    ball_carrier_player_id="H0",
                )
            )
    return pd.DataFrame(rows)


def _das_actions(rows: list[tuple[int, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [a for a, _ in rows],
            "game_id": [1] * len(rows),
            "period_id": [1] * len(rows),
            "time_seconds": [1.0] * len(rows),
            "team_id": [t for _, t in rows],
        }
    )


def _links(pairs: list[tuple[int, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [a for a, _ in pairs],
            "frame_id": [f for _, f in pairs],
            "time_offset_seconds": [0.0] * len(pairs),
            "n_candidate_frames": [1] * len(pairs),
            "link_quality_score": [1.0] * len(pairs),
        }
    )


# ---------------------------------------------------------------------------
# Degradation taxonomy (ADR-043): the closed vocabulary + the one degradable error.
# ---------------------------------------------------------------------------


class TestDasTaxonomy:
    def test_dasunscoreableerror_subclasses_valueerror(self) -> None:
        assert issubclass(DasUnscoreableError, ValueError)

    def test_das_source_token_is_validated(self) -> None:
        with pytest.raises(ValueError, match="das_source must be one of"):
            DasUnscoreableError("boom", das_source="not_a_token")

    def test_vocabulary_has_exactly_five_tokens(self) -> None:
        assert len(DAS_SOURCE_VALUES) == 5


# ---------------------------------------------------------------------------
# Facade input validation (the contract lives in _das_pack; here at the facade edge).
# ---------------------------------------------------------------------------


class TestFacadeValidation:
    def test_missing_team_in_possession_raises(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        frames = _keeper_frames((1,)).drop(columns=["team_in_possession"])
        with pytest.raises(ValueError, match="team_in_possession"):
            get_individual_das(frames)

    def test_missing_velocity_without_marker_raises(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        frames = _keeper_frames((1,)).drop(columns=["vx", "vy"])
        with pytest.raises(ValueError, match="velocity columns"):
            get_individual_das(frames)

    def test_velocity_unavailable_marker_degrades(self) -> None:
        from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE
        from silly_kicks.tracking._das import get_individual_das

        frames = _keeper_frames((1,)).drop(columns=["vx", "vy"])
        frames["speed_source"] = SPEED_SOURCE_UNAVAILABLE
        with pytest.raises(DasUnscoreableError) as exc:
            get_individual_das(frames)
        assert exc.value.das_source == DAS_SOURCE_UNSCOREABLE_FRAME

    def test_reference_quadrature_is_refused_on_the_public_surface(self) -> None:
        import dataclasses

        from silly_kicks.tracking._das import get_das
        from silly_kicks.tracking._das_params import DAS_PARAMS

        ref = dataclasses.replace(DAS_PARAMS, quadrature="reference")
        with pytest.raises(ValueError, match="parity-only"):
            get_das(_keeper_frames((1,)), params=ref)

    def test_goal_map_and_direction_col_are_mutually_exclusive(self) -> None:
        from silly_kicks.tracking._das import get_das

        with pytest.raises(ValueError, match="not both"):
            get_das(_keeper_frames((1,)), goal_map=object(), attacking_direction_col="x")

    def test_unknown_kwarg_raises_typeerror(self) -> None:
        """No **kwargs, no use_progress_bar (the accessible-space relic)."""
        from silly_kicks.tracking._das import get_das

        with pytest.raises(TypeError):
            get_das(_keeper_frames((1,)), use_progress_bar=False)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# get_das / get_individual_das output contract.
# ---------------------------------------------------------------------------


class TestGetDasContract:
    def test_get_das_broadcasts_team_value_to_every_row_incl_ball(self) -> None:
        from silly_kicks.tracking._das import get_das

        frames = _keeper_frames((1,))
        out = get_das(frames)
        assert {"AS", "DAS"} <= set(out.columns)
        assert len(out) == len(frames)
        # One team value broadcast to every row of the frame, including the ball row.
        vals = out["DAS"].dropna().unique()
        assert len(vals) == 1 and np.isfinite(vals[0])
        assert np.isfinite(out.loc[out["is_ball"], "DAS"]).all(), "the ball row carries the team value too"

    def test_get_individual_das_ball_row_is_nan(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        out = get_individual_das(_keeper_frames((1,)))
        assert out.loc[out["is_ball"], "DAS"].isna().all()
        assert out.loc[~out["is_ball"], "DAS"].notna().any()

    def test_input_is_not_mutated_and_player_id_dtype_preserved(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        frames = _keeper_frames((1,))
        before = frames.copy(deep=True)
        get_individual_das(frames)
        pd.testing.assert_frame_equal(frames, before)

    def test_team_das_differs_between_asymmetric_teams(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        out = get_individual_das(_keeper_frames((1,)))
        players = out[~out["is_ball"]]
        home = players.loc[players["team_id"] == "Home", "DAS"].sum(min_count=1)
        away = players.loc[players["team_id"] == "Away", "DAS"].sum(min_count=1)
        # Non-vacuity: both must be finite, else np.isclose(nan, nan) is False and the guard is hollow.
        assert np.isfinite(home) and np.isfinite(away), f"vacuous: Home={home} Away={away} not both finite"
        assert not np.isclose(home, away), f"Home={home} Away={away}: asymmetric teams must differ"

    def test_non_scoreable_frame_is_nan(self) -> None:
        from silly_kicks.tracking._das import get_das

        frames = _keeper_frames((1,))
        frames["team_in_possession"] = np.nan  # dead ball -> unscoreable
        with pytest.raises(DasUnscoreableError):
            get_das(frames)


class TestDasCostGuardrail:
    def test_estimate_das_cost_is_positive_and_scales(self) -> None:
        from silly_kicks.tracking._das import estimate_das_cost

        small = estimate_das_cost(_keeper_frames((1,)))
        big = estimate_das_cost(_keeper_frames(tuple(range(1, 6))))
        assert small > 0 and big > small

    def test_estimate_das_cost_is_thread_aware(self) -> None:
        """n_threads>1 selects the prange kernel -> a lower estimate (§6.14)."""
        from silly_kicks.tracking._das import estimate_das_cost

        frames = _keeper_frames(tuple(range(1, 6)))
        serial = estimate_das_cost(frames)
        parallel = estimate_das_cost(frames, n_threads=4)
        assert parallel < serial

    def test_warn_cost_fires_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod
        from silly_kicks.tracking._warnings import DasCostWarning

        monkeypatch.setattr(das_mod, "_DAS_COST_WARN_SECONDS", 0.01)  # 3 frames x 0.02 s = 0.06 s > 0.01
        with pytest.warns(DasCostWarning):
            das_mod.get_das(_keeper_frames((1, 2, 3)))

    def test_warn_cost_false_silences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        monkeypatch.setattr(das_mod, "_DAS_COST_WARN_SECONDS", 0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            das_mod.get_das(_keeper_frames((1, 2, 3)), warn_cost=False)


# ---------------------------------------------------------------------------
# features integration: add_das / das_at_action / das_xfns.
# ---------------------------------------------------------------------------


class TestDasXfns:
    def test_das_xfns_are_frame_aware(self) -> None:
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        for xfn in das_xfns:
            assert is_frame_aware(xfn), f"{xfn.__name__} is not frame_aware"

    def test_das_xfns_feature_column_names(self) -> None:
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.features import feature_column_names

        cols = feature_column_names(das_xfns, nb_prev_actions=3)  # type: ignore[arg-type]
        expected = {f"das_{m}_a{i}" for m in ("team", "opponent", "diff") for i in range(3)}
        assert expected == set(cols)

    def test_das_xfns_length(self) -> None:
        from silly_kicks.tracking.features import das_xfns

        assert len(das_xfns) == 1

    def test_das_source_is_not_a_vaep_feature_column(self) -> None:
        """VAEP feature matrices stay numeric: das_source is an aggregator column only."""
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.features import feature_column_names

        cols = feature_column_names(das_xfns, nb_prev_actions=3)  # type: ignore[arg-type]
        assert not any("das_source" in c for c in cols)

    def test_das_at_action_introspection(self) -> None:
        from silly_kicks.tracking.features import das_at_action

        dummy = pd.DataFrame({"action_id": [1, 2], "team_id": [1, 1]})
        result = das_at_action(dummy, None)
        assert result.name == "das_team"
        assert result.isna().all() and len(result) == 2


class TestChunkSizePassthrough:
    """chunk_size threads add_das/das_at_action -> get_individual_das -> compute_das."""

    def test_add_das_threads_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        captured: dict = {}
        real = das_mod.compute_das

        def spy(packed, params, **kwargs):
            captured["chunk_size"] = kwargs.get("chunk_size")
            return real(packed, params, **kwargs)

        monkeypatch.setattr(das_mod, "compute_das", spy)
        from silly_kicks.tracking.features import add_das

        frames = _keeper_frames((1,))
        add_das(_das_actions([(1, "Home")]), frames, links=_links([(1, 1)]), chunk_size=250)
        assert captured["chunk_size"] == 250


class TestDasLinkedFrameRestriction:
    """add_das/das_xfns simulate ONLY the linked frames (perf), direction pinned on full frames."""

    def test_add_das_restricts_simulation_to_linked_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        seen: dict = {}
        real = das_mod.get_individual_das

        def spy(frames, **kwargs):
            seen["frame_ids"] = sorted(frames["frame_id"].unique().tolist())
            return real(frames, **kwargs)

        monkeypatch.setattr("silly_kicks.tracking.features.get_individual_das", spy, raising=False)
        # features imports get_individual_das lazily inside _precompute_das_lookup, so patch the source.
        monkeypatch.setattr(das_mod, "get_individual_das", spy)
        from silly_kicks.tracking.features import add_das

        frames = _keeper_frames((1, 2, 3))
        add_das(_das_actions([(1, "Home")]), frames, links=_links([(1, 2)]))
        assert seen["frame_ids"] == [2], "only the linked frame is simulated"


# ---------------------------------------------------------------------------
# attacking_direction_col passthrough (caller-supplied per-frame numeric direction).
# ---------------------------------------------------------------------------


def _numeric_dir_frames(dir_by_frame: dict[int, float], *, rows_per_frame: int = 2) -> pd.DataFrame:
    rows = []
    for fid, dval in dir_by_frame.items():
        for p in range(rows_per_frame):
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    player_id=f"P{p}",
                    team_id="A" if p % 2 == 0 else "B",
                    is_ball=False,
                    attacking_direction=dval,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="A",
                )
            )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                player_id="ball",
                team_id=None,
                is_ball=True,
                attacking_direction=np.nan,
                x=50.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="A",
            )
        )
    return pd.DataFrame(rows)


class TestAttackingDirectionColValidation:
    """Fail-loud validation of a caller-supplied per-frame direction column (errors PROPAGATE)."""

    def _actions(self) -> pd.DataFrame:
        return _das_actions([(1, "A")])

    def test_missing_column_raises_valueerror(self) -> None:
        from silly_kicks.tracking.features import add_das

        with pytest.raises(ValueError, match="not found"):
            add_das(self._actions(), _numeric_dir_frames({0: 1.0}), attacking_direction_col="nope")

    def test_non_numeric_column_raises_typeerror(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({0: 1.0})
        frames["attacking_direction"] = "ltr"
        with pytest.raises(TypeError, match="numeric"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_all_nan_group_raises_valueerror_naming_group(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({0: np.nan, 1: np.nan})
        with pytest.raises(ValueError, match="period_id=1"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_partial_coverage_group_raises_valueerror_naming_frames(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({10: 1.0, 11: np.nan, 12: 1.0})
        with pytest.raises(ValueError, match="11"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_valid_numeric_column_threads_to_precompute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking.features as feat_mod

        captured: dict = {}

        def spy_precompute(
            frames,
            *,
            chunk_size=None,
            link_frame_ids=None,
            attacking_direction_col=None,
            goal_map=None,
            params=None,
            n_threads=None,
        ):
            captured["adc"] = attacking_direction_col
            return {}

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)
        frames = _numeric_dir_frames({0: 1.0, 1: -1.0})
        feat_mod.add_das(self._actions(), frames, links=_links([(1, 0)]), attacking_direction_col="attacking_direction")
        assert captured["adc"] == "attacking_direction"

    def test_goal_map_threads_to_precompute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``add_das(goal_map=…)`` hands the caller's map straight to the lookup (plan Task 10)."""
        import silly_kicks.tracking.features as feat_mod

        captured: dict = {}
        sentinel = object()

        def spy_precompute(
            frames,
            *,
            chunk_size=None,
            link_frame_ids=None,
            attacking_direction_col=None,
            goal_map=None,
            params=None,
            n_threads=None,
        ):
            captured["goal_map"] = goal_map
            return {}

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)
        frames = _numeric_dir_frames({0: 1.0, 1: -1.0})
        feat_mod.add_das(self._actions(), frames, links=_links([(1, 0)]), goal_map=sentinel)
        assert captured["goal_map"] is sentinel

    def test_goal_map_and_direction_col_are_mutually_exclusive(self) -> None:
        """A caller cannot pin direction two ways at once (spec §6.2)."""
        from silly_kicks.tracking.features import add_das

        with pytest.raises(ValueError, match="not both"):
            add_das(self._actions(), _keeper_frames((1,)), goal_map=object(), attacking_direction_col="x")


# ---------------------------------------------------------------------------
# das_source provenance (ADR-043): distinguishes "could not compute" from "genuinely NaN".
# ---------------------------------------------------------------------------


class TestDasSourceProvenance:
    def test_vocabulary_is_closed_and_every_branch_is_reachable(self) -> None:
        from silly_kicks.tracking import DAS_SOURCE_VALUES, add_das

        frames = _keeper_frames((1,))
        # action 1: links to the live frame, real team  -> computed
        # action 2: no pointer at all                   -> unlinked
        # action 3: pointer to a frame that has no DAS  -> unscoreable_frame
        # action 4: links to the live frame, alien team -> team_unresolved
        actions = _das_actions([(1, "Home"), (2, "Home"), (3, "Home"), (4, "Nowhere")])
        links = _links([(1, 1), (3, 99), (4, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)

        assert set(out["das_source"]) <= set(DAS_SOURCE_VALUES)
        by_action = dict(zip(out["action_id"], out["das_source"], strict=True))
        assert by_action == {1: "computed", 2: "unlinked", 3: "unscoreable_frame", 4: "team_unresolved"}
        vals = dict(zip(out["action_id"], out["das_team"], strict=True))
        assert np.isfinite(vals[1]), "the 'computed' row must actually carry a finite DAS"
        assert all(pd.isna(vals[a]) for a in (2, 3, 4))

    def test_unscoreable_call_on_a_genuine_dead_ball_window(self) -> None:
        from silly_kicks.tracking import add_das

        frames = _keeper_frames((1,))
        frames["team_in_possession"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(_das_actions([(1, "Home")]), frames, links=_links([(1, 1)]))
        assert out["das_team"].isna().all()
        assert (out["das_source"] == "unscoreable_call").all()

    def test_unmarked_missing_velocity_propagates_not_degrades(self) -> None:
        """A caller-contract violation (forgot derive_velocities) must fail loud."""
        from silly_kicks.tracking import add_das

        frames = _keeper_frames((1,)).drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity columns"):
            add_das(_das_actions([(1, "Home")]), frames, links=_links([(1, 1)]))

    def test_unexpected_valueerror_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain ValueError is NOT the degrade -- it must reach the caller."""
        import silly_kicks.tracking.features as feat_mod

        def boom(
            frames,
            *,
            chunk_size=None,
            link_frame_ids=None,
            attacking_direction_col=None,
            goal_map=None,
            params=None,
            n_threads=None,
        ):
            raise ValueError("a real defect, not an unscoreable window")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", boom)
        with pytest.raises(ValueError, match="a real defect"):
            feat_mod.add_das(_das_actions([(1, "Home")]), _keeper_frames((1,)))


# ---------------------------------------------------------------------------
# Degenerate-frame degradation: a subset with no simulatable frame -> NaN, never crash.
# ---------------------------------------------------------------------------


class TestZeroFrameSubsetDegradesToNaN:
    """A link-restricted subset whose frames have no ball, or no players, degrades to NaN."""

    def _disjoint_frames(self) -> pd.DataFrame:
        """Frame 1 = ball only; frame 2 = players only (no single frame has both)."""
        return pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    is_goalkeeper=False,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="H0",
                    team_id="Home",
                    is_ball=False,
                    is_goalkeeper=True,
                    x=5.0,
                    y=34.0,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="A0",
                    team_id="Away",
                    is_ball=False,
                    is_goalkeeper=True,
                    x=100.0,
                    y=34.0,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
        )

    def test_get_individual_das_no_crash_returns_nan(self) -> None:
        from silly_kicks.tracking._das import get_individual_das

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_individual_das(self._disjoint_frames())
        assert {"AS", "DAS"} <= set(out.columns)
        assert out["DAS"].isna().all() and out["AS"].isna().all()
        assert len(out) == 3  # shape preserved

    def test_add_das_via_links_degrades_to_nan(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = self._disjoint_frames()
        actions = _das_actions([(1, "Home"), (2, "Home")])
        links = _links([(1, 1), (2, 2)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)
        assert out["das_team"].isna().all()


class TestXcZeroFrameSubsetDegradesToNaN:
    """get_xc degrades to NaN (one aggregated warning) for a pass whose frame is unscoreable."""

    def _frames(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    is_goalkeeper=False,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="A",
                    team_id="Home",
                    is_ball=False,
                    is_goalkeeper=True,
                    x=5.0,
                    y=30.0,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="B",
                    team_id="Away",
                    is_ball=False,
                    is_goalkeeper=True,
                    x=100.0,
                    y=30.0,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    is_goalkeeper=False,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
        )

    def _pass_at(self, frame_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    action_id=1,
                    game_id=1,
                    period_id=1,
                    frame_id=frame_id,
                    player_id="A",
                    team_id="Home",
                    start_x=40.0,
                    start_y=30.0,
                    end_x=60.0,
                    end_y=30.0,
                )
            ]
        )

    def test_get_xc_unscoreable_frame_returns_nan(self) -> None:
        from silly_kicks.tracking._das import get_xc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_xc(self._pass_at(1), self._frames())  # frame 1 = ball only
        assert "xC" in out.columns
        assert out["xC"].isna().all() and len(out) == 1
