"""Offline gates for the TF-54b owner-run drivers (the full corpus RUNS are owner-gated / e2e).

Mirrors ``tests/scripts/test_provenance_wiring.py`` for the two new drivers -- provenance wiring
(imports the shared helper, offers ``--allow-dirty``, no bare ``git rev-parse`` CALL, calls the guard
from ``main()``) -- plus the two pure pieces those drivers rest on: the synthetic-interception
perturbation (SPEC-02/09: it corrupts BOTH the intended distance AND the intended direction) and the
LOCKED pre-registered constants + ``decide_promotion`` that reads them (never inline float literals).
The full corpus pass is deliberately NOT run here.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
_DRIVERS = ("train_pass_completion", "validate_territory_counterfactual")


def _source(name: str) -> str:
    return (_SCRIPTS / f"{name}.py").read_text(encoding="utf-8")


def _function(src: str, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def _calls_in(fn: ast.AST, name: str) -> list[int]:
    return [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == name]


def _shells_out_to_rev_parse(src: str) -> bool:
    """True only for an actual CALL passing "rev-parse", never for prose mentioning it (the
    established idiom from test_provenance_wiring._shells_out_to_rev_parse)."""
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "rev-parse" in arg.value:
                return True
    return False


# ----------------------------------------------------------------------------- provenance wiring


@pytest.mark.parametrize("name", _DRIVERS)
def test_driver_imports_the_shared_provenance_helper(name):
    assert "_provenance" in _source(name), f"{name}.py must import scripts._provenance (rev-parse is not provenance)"


@pytest.mark.parametrize("name", _DRIVERS)
def test_driver_offers_an_allow_dirty_escape_hatch(name):
    assert "--allow-dirty" in _source(name), f"{name}.py has no --allow-dirty flag"


@pytest.mark.parametrize("name", _DRIVERS)
def test_driver_never_shells_out_to_rev_parse_directly(name):
    assert not _shells_out_to_rev_parse(_source(name)), (
        f"{name}.py calls `git rev-parse` directly -- route it through scripts._provenance."
    )


@pytest.mark.parametrize("name", _DRIVERS)
def test_the_entry_point_enforces_the_clean_tree(name):
    """require_clean_tree must be called from main() -- the CLI entry point."""
    main_fn = _function(_source(name), "main")
    assert main_fn is not None, f"{name}.py has no main()"
    assert _calls_in(main_fn, "require_clean_tree"), (
        f"{name}.py never calls require_clean_tree from main() -- a CLI run could write from a dirty tree."
    )


@pytest.mark.parametrize("name", _DRIVERS)
def test_the_guard_precedes_the_corpus_walk_within_main(name):
    main_fn = _function(_source(name), "main")
    assert main_fn is not None
    guard = _calls_in(main_fn, "require_clean_tree")
    walk = _calls_in(main_fn, "load_matches") + _calls_in(main_fn, "for_each")
    assert walk, f"{name}.py main() does not drive a corpus walk"
    assert min(guard) < min(walk), (
        f"{name}.py starts the corpus walk at line {min(walk)} before checking the tree at {min(guard)}."
    )


@pytest.mark.parametrize("name", _DRIVERS)
def test_driver_is_enrolled_in_the_artifact_provenance_registry(name):
    """Both drivers write cited artifacts, so the standing provenance-wiring gate must cover them."""
    from tests.scripts.test_provenance_wiring import ARTIFACT_DRIVERS

    assert name in ARTIFACT_DRIVERS, f"{name} must be enrolled in ARTIFACT_DRIVERS"


def test_the_rev_parse_detector_distinguishes_a_CALL_from_PROSE():
    """Non-vacuity: the detector fires on the real thing and stays silent on a mention."""
    assert _shells_out_to_rev_parse('subprocess.run(["git", "rev-parse", "HEAD"])')
    assert not _shells_out_to_rev_parse('"""We must never call git rev-parse HEAD directly."""')


# --------------------------------------------------------------- synthetic interception (SPEC-02/09)


def _perp_distance(origin, end, death) -> float:
    """Perpendicular distance from ``death`` to the infinite line through ``origin`` and ``end``."""
    ox, oy = origin
    ex, ey = end
    vx, vy = ex - ox, ey - oy
    length = np.hypot(vx, vy)
    return abs((death[0] - ox) * vy - (death[1] - oy) * vx) / length


def test_perturb_zero_angle_lands_on_the_segment_at_fraction():
    """angle_offset_rad=0 -> the death is exactly on the origin->end segment at the given fraction."""
    from scripts._synthetic_interception import perturb_interception

    origin, end, f = (20.0, 30.0), (80.0, 50.0), 0.4
    dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=0.0)
    assert np.isclose(float(dx), origin[0] + f * (end[0] - origin[0]))
    assert np.isclose(float(dy), origin[1] + f * (end[1] - origin[1]))
    # On the ray => zero perpendicular distance (collinear).
    assert _perp_distance(origin, end, (float(dx), float(dy))) < 1e-9


def test_perturb_nonzero_angle_moves_the_death_off_the_ray():
    """angle_offset_rad != 0 -> the death leaves the ray by exactly f*|v|*|sin(delta)| (> 0)."""
    from scripts._synthetic_interception import perturb_interception

    origin, end, f, delta = (20.0, 30.0), (80.0, 50.0), 0.4, 0.3
    dx0, dy0 = perturb_interception(origin, end, fraction=f, angle_offset_rad=0.0)
    dx, dy = perturb_interception(origin, end, fraction=f, angle_offset_rad=delta)
    perp = _perp_distance(origin, end, (float(dx), float(dy)))
    # Off the ray, by the exact ground-truth perpendicular distance (validates the direction corruption).
    assert perp > 1e-3
    length = np.hypot(end[0] - origin[0], end[1] - origin[1])
    assert np.isclose(perp, f * length * abs(np.sin(delta)))
    # And it genuinely moved relative to the zero-angle death.
    assert not (np.isclose(float(dx), float(dx0)) and np.isclose(float(dy), float(dy0)))


def test_perturb_is_vectorized():
    from scripts._synthetic_interception import perturb_interception

    ox, oy = np.array([20.0, 0.0]), np.array([30.0, 0.0])
    ex, ey = np.array([80.0, 10.0]), np.array([50.0, 10.0])
    f = np.array([0.4, 0.5])
    delta = np.array([0.0, 0.2])
    dx, dy = perturb_interception((ox, oy), (ex, ey), fraction=f, angle_offset_rad=delta)
    assert dx.shape == (2,) and dy.shape == (2,)
    assert np.isclose(dx[0], 20.0 + 0.4 * 60.0)  # first row has zero angle -> on the segment


# --------------------------------------------------------------- locked constants + decide_promotion


def test_elite_defender_prior_is_a_nonempty_frozenset():
    import scripts.validate_territory_counterfactual as v

    assert isinstance(v.ELITE_DEFENDER_PRIOR, frozenset)
    assert len(v.ELITE_DEFENDER_PRIOR) > 0


def test_the_four_numeric_floors_are_module_level_float_constants():
    import scripts.validate_territory_counterfactual as v

    for name in (
        "COMPLETION_AUC_FLOOR",
        "COMPLETION_ECE_CEILING",
        "COMPLETION_BRIER_SKILL_FLOOR",
        "ELITE_DEFENDER_TOP_QUANTILE",
    ):
        assert isinstance(getattr(v, name), float), f"{name} must be a module-level float constant"


def test_decide_promotion_reads_the_constants_never_inline_literals():
    """AST: decide_promotion references the locked module constants and carries NO bare float literal
    for any threshold (the TF-19 / ADR-056 pre-registration idiom: the gate cannot be moved by editing
    a number in the function body)."""
    fn = _function(_source("validate_territory_counterfactual"), "decide_promotion")
    assert fn is not None
    floats = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert not floats, f"decide_promotion carries bare float literals {floats}; reference the locked constants"
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert {
        "COMPLETION_AUC_FLOOR",
        "COMPLETION_ECE_CEILING",
        "COMPLETION_BRIER_SKILL_FLOOR",
        "ELITE_DEFENDER_TOP_QUANTILE",
    } <= names, "decide_promotion must reference every locked threshold constant by name"


def test_decide_promotion_gates_on_every_prong_both_ways():
    from scripts.validate_territory_counterfactual import decide_promotion

    good = {
        # BSS = 1 - 0.10 / (0.75*0.25 = 0.1875) = 0.467 >= 0.10
        "completion": {"auc": 0.80, "ece": 0.05, "brier": 0.10, "base_rate": 0.75},
        "mechanism": {"counterfactual_error": 1.0, "baseline_death_error": 2.0, "baseline_centroid_error": 3.0},
        "elite_prior": {"elite_quantile": 0.90},
    }
    out = decide_promotion(good)
    assert out["promote"] is True
    assert np.isclose(out["brier_skill_score"], 1 - 0.10 / (0.75 * 0.25))

    def _flip(**patch):
        m = {k: dict(v) for k, v in good.items()}
        for section, kv in patch.items():
            m[section].update(kv)
        return decide_promotion(m)

    assert _flip(completion={"auc": 0.50})["promote"] is False  # AUC below floor
    assert _flip(completion={"ece": 0.20})["promote"] is False  # ECE above ceiling
    assert _flip(completion={"brier": 0.19})["promote"] is False  # BSS ~0 below floor
    assert _flip(mechanism={"counterfactual_error": 5.0})["promote"] is False  # loses to baselines
    assert _flip(elite_prior={"elite_quantile": 0.10})["promote"] is False  # elites rank low


def test_decide_promotion_surfaces_the_uncomputed_real_data_leg():
    """Primary-1's real-data mechanism leg is infeasible event-only (spec section 7.2); ``promote: True``
    must not be over-readable as covering it. The promote/no-promote LOGIC is unchanged -- it already
    gates only on the computable legs -- this pins the added TRANSPARENCY fields that make the omission
    visible on the decision itself."""
    from scripts.validate_territory_counterfactual import decide_promotion

    good = {
        "completion": {"auc": 0.80, "ece": 0.05, "brier": 0.10, "base_rate": 0.75},
        "mechanism": {
            "counterfactual_error": 1.0,
            "baseline_death_error": 2.0,
            "baseline_centroid_error": 3.0,
            "real_data_leg": {"status": "not_computed_requires_owner_decision"},
        },
        "elite_prior": {"elite_quantile": 0.90},
    }
    out = decide_promotion(good)
    assert out["promote"] is True  # gate logic untouched -- still gates on the computable legs only
    assert out["real_data_leg_uncomputed"] is True
    assert isinstance(out["promote_scope"], str) and "real-data" in out["promote_scope"].lower()

    computed = {k: dict(v) for k, v in good.items()}
    computed["mechanism"] = {**good["mechanism"], "real_data_leg": {"status": "computed"}}
    assert decide_promotion(computed)["real_data_leg_uncomputed"] is False

    # A caller that omits `real_data_leg` entirely (e.g. the existing prong-gating fixture above) must
    # not crash -- absence reads as "not flagged uncomputed", never a KeyError.
    no_leg = {k: dict(v) for k, v in good.items()}
    no_leg["mechanism"] = {
        "counterfactual_error": 1.0,
        "baseline_death_error": 2.0,
        "baseline_centroid_error": 3.0,
    }
    assert decide_promotion(no_leg)["real_data_leg_uncomputed"] is False


def test_brier_skill_score_matches_the_definition():
    from scripts.validate_territory_counterfactual import brier_skill_score

    assert np.isclose(brier_skill_score(0.10, 0.75), 1 - 0.10 / (0.75 * 0.25))
    assert np.isnan(brier_skill_score(0.10, 1.0))  # degenerate base rate -> NaN, not a division blowup


# ------------------------------------------------- run_battery on a TINY synthetic corpus (offline)
# This is the network-free regression that catches C1 (_aggregate_defenders KeyError on the v1 schema)
# and C2 (dead elite prior: player_name never emitted by territory) at test speed.


def _synthetic_match(game_id: int):
    """One tiny SPADL match: an ELITE defender (named) + an ordinary defender, both with own-half
    defensive-action clouds (hulls), opponent passes aimed into those hulls (completed AND failed),
    a couple of home passes, and shots. Every match carries both pass outcomes so a single-match
    training fold still fits the completion model's two classes."""
    from silly_kicks.spadl import config as spc

    tackle, pas, shot = spc.actiontype_id["tackle"], spc.actiontype_id["pass"], spc.actiontype_id["shot"]
    ok, bad = spc.result_id["success"], spc.result_id["fail"]
    rows: list[dict] = []

    def add(team, player, name, type_id, result_id, sx, sy, ex, ey):
        rows.append(
            {
                "game_id": game_id,
                "period_id": 1,
                "time_seconds": float(len(rows)),
                "team_id": team,
                "player_id": player,
                "player_name": name,
                "type_id": type_id,
                "result_id": result_id,
                "bodypart_id": 0,
                "start_x": float(sx),
                "start_y": float(sy),
                "end_x": float(ex),
                "end_y": float(ey),
            }
        )

    # Elite defender (player 1, team 10): a TIGHT own-half blob, centroid ~ (10, 34). Trimming barely
    # moves a tight blob, and a pass reflected ONTO the centroid is inside ANY convex hull.
    for x, y in [(8, 32), (12, 32), (12, 36), (8, 36), (10, 34), (9, 33), (11, 35), (10, 31), (10, 37)]:
        add(10, 1, "Virgil van Dijk", tackle, ok, x, y, x, y)
    # Ordinary defender (player 2, team 10): a tight blob, centroid ~ (35, 20).
    for x, y in [(33, 18), (37, 18), (37, 22), (33, 22), (35, 20), (34, 19), (36, 21), (35, 17), (35, 23)]:
        add(10, 2, "Ordinary Defender", tackle, ok, x, y, x, y)
    # Opponent (team 20) passes; reflected end (105-ex, 68-ey) lands ON the elite centroid (10, 34).
    add(20, 99, "Opp A", pas, ok, 80, 34, 95, 34)  # -> (10, 34) conceded
    add(20, 98, "Opp B", pas, ok, 78, 33, 95, 35)  # -> (10, 33)
    add(20, 99, "Opp A", pas, ok, 82, 35, 94, 33)  # -> (11, 35)
    add(20, 99, "Opp A", pas, bad, 80, 36, 96, 36)  # -> (9, 32) prevented
    add(20, 98, "Opp B", pas, bad, 76, 32, 95, 34)  # -> (10, 34)
    # Opponent passes reflecting ONTO the ordinary centroid (35, 20): end ~ (70, 48).
    add(20, 99, "Opp A", pas, ok, 80, 40, 70, 48)  # -> (35, 20)
    add(20, 99, "Opp A", pas, bad, 82, 42, 71, 47)  # -> (34, 21)
    add(20, 98, "Opp B", pas, ok, 78, 45, 69, 49)  # -> (36, 19)
    # A couple of home passes (xt variety) + shots (team_conceded proxy). At least some shots are
    # GOALS (result=success) so the singh scoring_prob is non-zero and the fitted xT grid is non-zero
    # (require_fitted_xt checks np.any(xT); an all-miss corpus stays all-zero and reads as unfitted).
    add(10, 3, "Mid X", pas, ok, 30, 34, 60, 34)
    add(10, 3, "Mid X", pas, bad, 40, 20, 70, 20)
    add(20, 99, "Opp A", shot, ok, 95, 34, 105, 34)  # goal
    add(20, 98, "Opp B", shot, ok, 92, 30, 105, 30)  # goal
    add(10, 3, "Mid X", shot, bad, 90, 34, 105, 34)  # miss

    df = pd.DataFrame(rows)
    df["action_id"] = range(len(df))
    return df


def _toy_xt(value: float = 0.1):
    """A uniform fitted xT (mirrors tests/territory/test_compute.py) -- values_at_points is constant."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat()
    xt.xT = np.full(np.asarray(xt.xT).shape, value, dtype=float)
    return xt


def test_run_battery_on_a_tiny_synthetic_corpus_produces_the_full_battery():
    from scripts.validate_territory_counterfactual import run_battery

    shards = [_synthetic_match(g) for g in (1, 2, 3)]
    metrics, defender_table = run_battery(shards, n_folds=3, cone_deg=45.0, min_support=1e-6, seed=7)

    # Top-level battery shape.
    assert {"completion", "mechanism", "elite_prior", "secondary", "n_folds", "n_matches"} <= set(metrics)
    assert {"auc", "ece", "brier", "base_rate"} <= set(metrics["completion"])
    # Mechanism: synthetic keys decide_promotion reads + the SURFACED (not stubbed) real-data leg.
    assert {"counterfactual_error", "baseline_death_error", "baseline_centroid_error", "real_data_leg"} <= set(
        metrics["mechanism"]
    )
    assert metrics["mechanism"]["real_data_leg"]["status"] == "not_computed_requires_owner_decision"
    # Secondary: the previously-MISSING legs are present.
    sec = metrics["secondary"]
    assert {"outcome_lens", "reliability", "discriminant"} <= set(sec)
    assert "split_half_spearman" in sec["reliability"]
    assert {"vs_team_strength_spearman", "beats_shuffled_placebo"} <= set(sec["discriminant"])

    # C2 fixed: the elite name resolves to exactly one player, it is counted, and the prior is finite.
    ep = metrics["elite_prior"]
    assert ep["name_match_counts"]["van Dijk"]["n_players"] == 1  # IMPORTANT-4 census is populated
    assert ep["n_elite_matched"] >= 1
    assert not np.isnan(ep["elite_quantile"])
    # The defender table carries the JOINED name + is_elite + the counterfactual-only volume column.
    assert "player_name" in defender_table.columns and bool(defender_table["is_elite"].any())
    assert "territory_passes_aimed_into_hull" in defender_table.columns


def test_aggregate_defenders_is_columns_aware_on_the_v1_schema():
    """C1 regression: _aggregate_defenders must NOT KeyError on the v1 completed_failed table, which
    lacks the counterfactual-only columns (territory_expected_threat_faced / _passes_aimed_into_hull)."""
    from scripts.validate_territory_counterfactual import _aggregate_defenders
    from silly_kicks.territory import compute_territorial_dominance

    v1, _report = compute_territorial_dominance(_synthetic_match(1), xt=_toy_xt())
    assert "territory_passes_aimed_into_hull" not in v1.columns  # precondition: v1 lacks the cf column
    table = _aggregate_defenders(v1, name_map={})  # must NOT raise
    assert "territory_xt_prevented" in table.columns
    assert "territory_passes_aimed_into_hull" not in table.columns  # not fabricated from thin air


def test_render_model_card_generates_a_card_with_metrics_and_adr088():
    """IMPL-04: the bundled PassCompletionModel ships a MODEL_CARD.md, generated by the trainer from the
    out-of-fold metrics so it cannot drift; every bundled model carries a card (ADR-088)."""
    from scripts.train_pass_completion import render_model_card

    metrics = {
        "auc": 0.734,
        "ece": 0.041,
        "brier": 0.182,
        "base_rate": 0.78,
        "n_pass_rows": 61944,
        "n_matches": 64,
        "providers": "statsbomb",
        "training_commit": "abc1234",
    }
    card = render_model_card(metrics)
    assert card.isascii()  # scripts/bundle ASCII discipline
    assert "Pass-completion model" in card and "PassCompletionModel.bundled()" in card
    assert "ADR-088" in card  # the card-required rule is satisfied for the in-repo path
    # metrics interpolated (not a static template): auc/brier/matches/commit all present
    assert "0.734" in card and "0.182" in card and "64" in card and "abc1234" in card
    assert "Brier skill score" in card  # computed vs the base-rate baseline base*(1-base)
