from scripts.hf_jobs.dgx_e2e import check_shipped_variant


def test_matching_shipped_variant_is_ok():
    assert check_shipped_variant(fresh="public", committed="public") is None


def test_wrong_shipped_variant_is_flagged():
    # the GATE-1 gap: an exit-0 run that shipped sc_extended instead of the committed public must BLOCK.
    err = check_shipped_variant(fresh="sc_extended", committed="public")
    assert err is not None
    assert "sc_extended" in err and "public" in err
