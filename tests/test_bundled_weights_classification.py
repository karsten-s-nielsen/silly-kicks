"""F1b (ADR-106) — every bundled weights dir is classified frame-geometry vs event-only vs out.

The float32 frame migration retrains every model that trains on tracking-frame geometry (its features
shift by the storage-rounding). The `test_frame_coord_upcast_gate` covers compute SITES; this gate
covers the MODEL SET: a FUTURE frame-geometry model that ships un-retrained on float32 frames would be
a silent train/serve skew, so every bundled weights dir must be classified here. Complete by
enumeration (ADR-056): the mechanical dir listing must equal the classification map exactly, and
`_UNDERIVABLE` is asserted empty.

Classifications:
- ``frame_geometry`` -- trains on tracking-frame coordinates -> RETRAINED on float32 (F1b commit-2).
- ``event_only``     -- trains on SPADL action coords (float64, unchanged) -> NOT retrained.
- ``out``            -- deferred this cycle for a separate reason (recorded).
"""

from pathlib import Path

_ROOT = Path("silly_kicks")

# dir (posix, relative to repo root) -> (classification, reason)
WEIGHTS_CLASSIFICATION: dict[str, tuple[str, str]] = {
    # frame-geometry -> retrained on float32 frames (commit-2)
    "silly_kicks/tracking/_xshot_weights": ("frame_geometry", "xShot features from frame positions"),
    "silly_kicks/tracking/_xcross_weights": ("frame_geometry", "xCross features from frame positions"),
    "silly_kicks/tracking/_ghost_gk_weights": ("frame_geometry", "ghost-GK regresses on frame positions"),
    "silly_kicks/tracking/_ghost_outfield_weights": ("frame_geometry", "ghost-outfield regresses on frame positions"),
    "silly_kicks/tracking/_receiver_weights": ("frame_geometry", "receiver model reads pre-pass frame positions"),
    "silly_kicks/tracking/_gk_completion_weights": (
        "frame_geometry",
        "consumes frame geometry via the optional defender_density (receiver_zone_density); "
        "the T10 feature-delta measurement confirms/reclassifies, retrain-conservative default",
    ),
    # event-only -> SPADL action coords (float64), NOT retrained
    "silly_kicks/expected_passing/weights": ("event_only", "PassCompletion: event-only action geometry"),
    "silly_kicks/match_outcome/weights": ("event_only", "DependenceModel rho: event-only, no frames"),
    "silly_kicks/win_probability/weights": ("event_only", "win-probability: event-only state, no frames"),
    "silly_kicks/xsuccess/weights": ("event_only", "xSuccess: END-BLIND event-only action geometry"),
    # out of this cycle
    "silly_kicks/xtgk/_retention_weights": ("out", "xt-gk-v2 rho (item #4) -- deferred, v2 quarantined"),
}

_VALID = {"frame_geometry", "event_only", "out"}

# Asserted EMPTY (ADR-056): a dir that is neither discoverable NOR classified would slip both sets.
_UNDERIVABLE: tuple[str, ...] = ()


def _discover_weights_dirs() -> set[str]:
    dirs = {p.as_posix() for p in _ROOT.glob("**/weights") if p.is_dir()}
    dirs |= {p.as_posix() for p in _ROOT.glob("**/_*_weights") if p.is_dir()}
    return dirs


def test_every_bundled_weights_dir_is_classified():
    discovered = _discover_weights_dirs()
    classified = set(WEIGHTS_CLASSIFICATION)
    assert discovered == classified, (
        f"weights-dir classification drift -- discovered-but-unclassified: {discovered - classified}; "
        f"classified-but-missing: {classified - discovered}"
    )


def test_classifications_are_valid():
    bad = {d: c for d, (c, _r) in WEIGHTS_CLASSIFICATION.items() if c not in _VALID}
    assert not bad, f"invalid classification(s): {bad}"


def test_underivable_is_empty():
    assert not _UNDERIVABLE, f"a weights dir escaped both discovery and classification: {_UNDERIVABLE}"
