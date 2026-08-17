"""`n_flipped` must be ATTRIBUTABLE, not merely reported.

Spec 1.1: the migration has exactly two contributors -- the 1 cm band (20.15 -> 20.16) and the depth
boundary (`<` -> `<=`). A flip count that cannot be decomposed is a number that cannot be reasoned
about next cycle.

A consistency assertion (`n_flipped == sum of parts`) is NOT enough on its own: any partition of
`flipped` satisfies it, including one that labels every flip "boundary". Demonstrated. So the real
guard is PER-CASE attribution with hand-derived answers, plus the negative case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.measure_box_constant_delta import classify_flips, frame_parquets

_ZERO = {
    "n_flipped": 0,
    "n_flipped_band_only": 0,
    "n_flipped_boundary_only": 0,
    "n_flipped_both": 0,
}


def test_band_only():
    """y inside the 1 cm strip, x comfortably inside: in under 20.16, out under 20.15."""
    out = classify_flips(np.array([5.0]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_band_only": 1}


def test_boundary_only():
    """Exactly on the depth line: `<` excludes, `<=` includes."""
    out = classify_flips(np.array([16.5]), np.array([34.0]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_boundary_only": 1}


def test_both_causes_is_its_own_bucket():
    """Both changes individually NECESSARY -- neither pure bucket may claim it, or the other
    becomes a systematic undercount."""
    out = classify_flips(np.array([16.5]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_both": 1}


def test_unaffected_point_flips_nothing():
    assert classify_flips(np.array([5.0]), np.array([34.0])) == _ZERO


def test_y_in_strip_but_x_outside_does_not_flip():
    """Negative case: the band change is irrelevant when depth already excludes the point."""
    assert classify_flips(np.array([40.0]), np.array([34.0 + 20.155])) == _ZERO


def test_the_shipped_legacy_BAND_form_is_modelled_not_the_abs_form():
    """THE regression that matters.

    `y = 13.85` is the ONLY value separating the two legacy forms (spec 1.1 item 3): the shipped
    min/max band says OUTSIDE (13.85 sits fractionally below `(68-40.3)/2`), the abs form says
    INSIDE (`|13.85-34.0|` is exactly 20.15), and canonical says inside. So it IS a flip -- and a
    driver modelling legacy with the abs form reports 0, an UNDERCOUNT at the exact boundary this
    driver exists to measure.
    """
    out = classify_flips(np.array([5.0]), np.array([13.85]))
    assert out["n_flipped"] == 1, "the shipped band form was replaced by the abs form"
    assert out["n_flipped_band_only"] == 1


def test_the_buckets_partition_flipped_over_a_large_random_sample():
    """Consistency is necessary but NOT sufficient -- see the module docstring. Kept as a companion
    to the per-case tests above, never as a substitute."""
    rng = np.random.default_rng(0)
    gr_x = rng.uniform(-5.0, 25.0, 200_000)
    y = rng.uniform(0.0, 68.0, 200_000)
    out = classify_flips(gr_x, y)
    parts = out["n_flipped_band_only"] + out["n_flipped_boundary_only"] + out["n_flipped_both"]
    assert out["n_flipped"] == parts
    assert out["n_flipped"] > 0, "sample produced no flips; the attribution is untested here"


def _write(path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(path)


def test_sidecar_directories_are_excluded_from_the_frame_glob(tmp_path):
    """REGRESSION, measured on the real 179-match pass. The bare `**/*.parquet` fallback was safe
    only while frames were the only parquet under `--out`. Once `materialize_tc3_frames` began
    emitting `_actions/<key>.parquet` for the trainer, this driver swept them up and died on
    `ArrowInvalid: No match for FieldRef.Name(x)` -- SPADL actions carry `start_x`, never `x`."""
    _write(tmp_path / "shards" / "abc123" / "skillcorner__1.parquet", {"x": [1.0], "y": [2.0]})
    _write(tmp_path / "_actions" / "skillcorner__1.parquet", {"start_x": [1.0], "action_id": [7]})
    found = frame_parquets(tmp_path)
    assert [p.name for p in found] == ["skillcorner__1.parquet"]
    assert all("_actions" not in p.parts for p in found)


def test_the_exclusion_does_not_swallow_real_shards(tmp_path):
    """Non-vacuity partner: the same layout WITHOUT a sidecar must still find every shard, so the
    test above is passing because `_actions` was excluded and not because the glob found nothing."""
    _write(tmp_path / "shards" / "abc123" / "skillcorner__1.parquet", {"x": [1.0], "y": [2.0]})
    _write(tmp_path / "shards" / "abc123" / "gradientsports__2.parquet", {"x": [3.0], "y": [4.0]})
    assert len(frame_parquets(tmp_path)) == 2


def test_the_tc3_tree_layout_still_wins_when_present(tmp_path):
    """`{provider}/{id}/frames.parquet` is the established layout and takes precedence, so a corpus
    written by `_loader_pining_to_cache.py` is read by its NAME rather than by the fallback."""
    _write(tmp_path / "skillcorner" / "1" / "frames.parquet", {"x": [1.0], "y": [2.0]})
    _write(tmp_path / "_actions" / "1.parquet", {"start_x": [1.0]})
    assert [p.name for p in frame_parquets(tmp_path)] == ["frames.parquet"]


def test_main_emits_the_training_flip_decision_inputs(tmp_path, monkeypatch):
    """End-to-end `main()`: builds a tc3-shaped cache and asserts the Phase-B fields land in
    metrics.json. Nothing else exercises `main()`, which is exactly how the last defect there
    (`prov.commit` on a dict) reached a corpus pass before it was caught."""
    import json
    import sys

    from scripts.measure_box_constant_delta import main
    from silly_kicks.spadl import config as spc
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    cache = tmp_path / "cache"
    (cache / "shards" / "tok").mkdir(parents=True)
    (cache / "_actions").mkdir()
    (cache / "_home").mkdir()
    frames = pd.concat(
        [_make_ghost_gk_frames(frame_id=1, timestamp=1.0), _make_ghost_gk_frames(frame_id=2, timestamp=2.0)],
        ignore_index=True,
    )
    frames.loc[frames["player_id"] == "a10", ["x", "y"]] = [-1.0, 34.0]  # a behind-line attacker
    frames.to_parquet(cache / "shards" / "tok" / "skillcorner__100.parquet")
    pd.DataFrame(
        {
            "game_id": ["100"],
            "period_id": [1],
            "team_id": [2],
            "time_seconds": [1.0],
            "type_id": [spc.actiontype_id["cross"]],
            "result_id": [spc.result_id["success"]],
        }
    ).to_parquet(cache / "_actions" / "skillcorner__100.parquet")
    (cache / "_home" / "skillcorner__100.json").write_text(json.dumps({"home_team_id": 1}))

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["measure_box_constant_delta.py", "--data-dir", str(cache), "--out", str(out), "--allow-dirty"]
    )
    main()

    metrics = json.loads((out / "metrics.json").read_text())
    assert "training_flip" in metrics
    ghost = metrics["training_flip"]["ghost"]
    assert ghost["n_examples"] > 0
    assert ghost["n_behind_line"] == 2  # a10 behind-line in the 2 home-GK examples
    assert ghost["changed_fraction"] > 0.0
    assert metrics["off_pitch_margin_m"] == 2.0
    # main() stamps the REAL git state, not a fixed value: `--allow-dirty` PERMITS a dirty tree, it
    # does not fabricate dirtiness. The value tracks the live checkout (clean on CI -> False, dirty in
    # dev -> True), so assert the field is stamped as a bool rather than a tree-state-dependent value.
    assert isinstance(metrics["run_tree_dirty"], bool)
    assert metrics["run_commit"]  # provenance stamped (real SHA on a clean tree, "unknown" only if git absent)
