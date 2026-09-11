"""TF-54b TerritorialDefenseReport conservation (ADR-042 dropped-and-counted)."""

# DEMOTED (ADR-090): the metric + its types are retained on the private path for the redesign.
from silly_kicks.territorial_defense._config import TerritorialDefenseParams
from silly_kicks.territorial_defense._report import TerritorialDefenseReport


def test_conservation_holds():
    r = TerritorialDefenseReport(
        TerritorialDefenseParams(),
        10,
        6,
        {"fov_cropped_local": 3, "removal_undersupported": 1},
    )
    assert r.n_frames_scored + sum(r.drop_reasons.values()) == r.n_frames_in


def test_default_drop_reasons_empty():
    r = TerritorialDefenseReport(TerritorialDefenseParams(), 5, 5)
    assert r.drop_reasons == {}
    assert r.n_frames_scored == r.n_frames_in
    # Arm-B census defaults are conserving-and-empty (a Layer-1-only caller).
    assert r.arm_b_n_in == 0 and r.arm_b_n_scored == 0 and r.arm_b_drop_reasons == {}


def test_arm_b_conservation_holds():
    # Arm B conserves over qualifying (defender, in-hull opponent-pass) pairs (ADR-042): a missing
    # frame / depleted removal / unresolvable geometry / non-finite delta is dropped-AND-counted.
    r = TerritorialDefenseReport(
        TerritorialDefenseParams(),
        3,
        2,
        {"no_actor": 1},
        arm_b_n_in=7,
        arm_b_n_scored=4,
        arm_b_drop_reasons={"removal_undersupported": 2, "missing_frame": 1},
    )
    assert r.arm_b_n_scored + sum(r.arm_b_drop_reasons.values()) == r.arm_b_n_in
