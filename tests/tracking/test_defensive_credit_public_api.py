import silly_kicks.tracking as T


def test_public_names_exported():
    for name in (
        "compute_defensive_credits",
        "add_defensive_credit",
        "compute_bravery",
        "DefensiveCreditParams",
        "DEFENSIVE_CREDIT_RULES",
    ):
        assert name in T.__all__, f"{name} missing from tracking.__all__"
        assert hasattr(T, name)
