def test_tf23_public_surface():
    import silly_kicks.tracking as t

    assert hasattr(t, "skillcorner") and hasattr(t.skillcorner, "convert_to_frames")
    assert hasattr(t, "metrica") and hasattr(t.metrica, "convert_to_frames")
    assert "orient_frames_to_ltr_by_geometry" in t.__all__
    assert "SKILLCORNER_TRACKING_FRAMES_COLUMNS" in t.__all__
    assert "METRICA_TRACKING_FRAMES_COLUMNS" in t.__all__
