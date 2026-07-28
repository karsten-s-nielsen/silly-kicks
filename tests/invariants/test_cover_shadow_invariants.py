"""Physical invariant tests for TF-30 cover shadow features."""

from __future__ import annotations

import pytest

from tests.tracking import _cover_shadow_inputs as _csi


@pytest.fixture
def cover_shadow_result():
    """Per-test COPY of the session-memoized build. The copy is the whole point."""
    return _csi.cover_shadow_result().copy()


@pytest.fixture
def cover_shadow_raw():
    """Per-action unclamped (threat_original, threat_unblocked). Read-only; do not mutate."""
    return _csi.cover_shadow_raw()


def test_per_test_fixture_is_a_copy(cover_shadow_result):
    """Memoizing must not hand the SAME object to two tests.

    A shared, non-copied DataFrame is a classic order-dependence flake: a ``.loc`` assignment, an
    ``inplace=True`` sort, or an added column in one test silently changes every later one. This
    risk is INTRODUCED by memoizing, so it is guarded rather than assumed away.

    ONE test, no ordering assumption. A mutator/observer pair would itself be order-dependent --
    vacuous under ``pytest -k``, under running the file's tests individually, under ``-x``, or
    after a rename.
    """
    assert cover_shadow_result is not _csi.cover_shadow_result()


# The plant must clear the tolerance by a real margin, not merely dip below zero -- otherwise a
# shrinking fixture could leave it "passing" on float noise.
_PLANT_MARGIN = 1e-6


def test_blocking_score_monotone_on_RAW_fields(cover_shadow_raw):
    """Removing defenders cannot decrease threat -- asserted on the UNCLAMPED difference.

    The shipped ``blocking_score`` column is clamped with ``max(..., 0.0)``, so the previous
    version of this test (``assert blocking_score >= -1e-9``) was green by construction and had
    never checked the property it named.

    COVERAGE, honestly: the measured minimum raw difference is ~+3.8 against a 1e-9 tolerance --
    nine orders of headroom over 9 actions. This catches GROSS breakage (a sign flip, the wrong
    team, a degenerate grid). It will NOT catch a subtle monotonicity violation that only appears
    in geometry this fixture does not contain.
    """
    from silly_kicks.tracking._cover_shadows import TOL_INVARIANT

    rows = cover_shadow_raw["rows"]
    assert rows, "fixture produced no scoreable actions -- FIX THE FIXTURE, do not skip"
    for aid, _frame, _tid, res in rows:
        assert res.threat_unblocked - res.threat_original >= -TOL_INVARIANT, aid


def test_a_negative_difference_is_reachable(cover_shadow_raw):
    """A permanent canary -- NOT a substitute for observing the guard itself fail.

    This calls ``compute_blocking_score`` with DIFFERENT arguments
    (``defenders_to_remove=[attacker_id]``) and a DIFFERENT assertion than the guard above, so it
    demonstrates a negative difference is REACHABLE. It does not demonstrate that
    ``test_blocking_score_monotone_on_RAW_fields`` fails when the production path breaks -- and it
    exercises the explicit ``defenders_to_remove`` branch while the guard's rows come from
    auto-identification, so a regression in ``_classify_man_markers`` is invisible to it. That
    claim is established by the one-off RED observation recorded in this commit's message.

    Direction AND target both matter. Dropping a NON-dangerous attacker can leave the counted cell
    set nearly unchanged -- its cells reassign to neighbours, some of them dangerous, so cells can
    even be ADDED -- and the plant silently becomes a no-op. Dropping a DANGEROUS attacker that
    owns non-zero per_receiver threat is guaranteed negative: no cell is ever added (the vacated
    region redistributes only among existing generators) so the counted set weakly shrinks, AND
    attacking pitch control weakly falls.
    """
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.tracking._cover_shadows import compute_blocking_score

    xt, home_team_id = cover_shadow_raw["xt"], cover_shadow_raw["home_team_id"]
    planted = 0
    for _aid, frame_data, tid, _res in cover_shadow_raw["rows"]:
        # `attacking_team_id` is POSITIONAL -- there is no `team=` keyword.
        surface = cs.compute_pitch_control(frame_data, tid, method="spearman")
        _total, per_receiver = cs._voronoi_threat(
            surface, xt, frame_data, attacking_team_id=tid, home_team_id=home_team_id
        )
        targets = [pid for pid, thr in per_receiver.items() if thr > 0.0]
        if not targets:
            continue
        res = compute_blocking_score(frame_data, tid, xt, home_team_id=home_team_id, defenders_to_remove=[targets[0]])
        assert res.threat_unblocked - res.threat_original < -_PLANT_MARGIN, (
            "plant did not go negative -- it has degenerated into a no-op and proves nothing"
        )
        planted += 1

    assert planted > 0, (
        "no fixture action offered a dangerous attacker with non-zero per_receiver threat. "
        "That is a FIXTURE INADEQUACY TO FIX, not a plant to weaken or skip."
    )


class TestCoverShadowInvariants:
    """Physical invariant properties of cover shadow features."""

    # NOTE: `test_blocking_score_non_negative` was DELETED here. It asserted
    # `blocking_score >= -1e-9` on a column already clamped with `max(..., 0.0)` -- green by
    # construction, and it had never checked the monotonicity it named. Superseded by the
    # module-level `test_blocking_score_monotone_on_RAW_fields` above, which asserts on the
    # unclamped `threat_unblocked - threat_original` and has been observed to fail on a plant.

    def test_blocked_threat_fraction_bounded(self, cover_shadow_result):
        """blocked_threat_fraction in [0, 1]."""
        valid = cover_shadow_result["blocked_threat_fraction"].dropna()
        assert (valid >= -1e-9).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_n_blocked_le_n_potential(self, cover_shadow_result):
        """Cannot block more lanes than exist."""
        df = cover_shadow_result
        both_valid = df[df["n_blocked_receivers"].notna() & df["n_potential_receivers"].notna()]
        if len(both_valid) == 0:
            pytest.skip("No valid rows")
        assert (both_valid["n_blocked_receivers"] <= both_valid["n_potential_receivers"]).all()

    def test_n_blocked_non_negative(self, cover_shadow_result):
        """n_blocked_receivers >= 0."""
        valid = cover_shadow_result["n_blocked_receivers"].dropna()
        assert (valid >= 0).all()

    def test_blocking_score_tracks_n_blocked_receivers(self, cover_shadow_result):
        """`blocking_score` rises with `n_blocked_receivers` -- two INDEPENDENTLY computed things.

        Renamed from `test_zero_blocked_implies_low_score`, which asserted only non-negativity and
        admitted so in its own comment. Repairing it in place exposed a second problem the spec had
        not anticipated: the zero-blocked population **does not exist** on provider frames. With ~10
        lane blockers per action some lane is always blocked -- measured 0/9 actions under EVERY
        decision rule (`any`, `majority`, `all`). The old body `pytest.skip`ped on the empty
        population, so it had never executed once since it was written.

        So the property is asked in the form the data CAN answer. This is not a weaker question: the
        real claim behind the original name is "does `blocking_score` track blocking at all?", and
        the monotone form tests it across the range that occurs rather than at a boundary that does
        not. If a zero-blocked row ever appears it simply becomes the lowest rung -- no conditional,
        no skip.

        What makes this non-vacuous: `n_blocked_receivers` comes from the per-receiver lane
        classifier (`p_blocked > p_received` on >=2 of 3 lanes) while `blocking_score` is the
        Voronoi/pitch-control threat integral. Different code paths, so agreement is real evidence.

        MEASURED: rho = 0.935, p = 0.00021 on 9 actions --
        n_blocked=1 (n=4) mean 11.34, =2 (n=2) mean 20.59, =3 (n=3) mean 24.95.

        The 0.5 floor is deliberately far below that observation. It is a GROSS-BREAKAGE catcher (a
        sign flip, the wrong team, a degenerate grid), not a calibrated threshold -- and it is
        honestly not pre-registered, since rho was measured before this assertion was written.
        Tuning the floor to 0.935 would have made it a fixture fingerprint rather than a guard. At
        n=9 this is DIRECTIONAL EVIDENCE; passing says little, failing is a real finding.
        """
        from scipy.stats import spearmanr

        df = cover_shadow_result
        both = df[["n_blocked_receivers", "blocking_score"]].dropna()
        counts = set(both["n_blocked_receivers"].tolist())
        # Non-vacuity: spearmanr on a constant column returns NaN, and `NaN >= 0.5` is False, so a
        # degenerate fixture would fail below -- but with an unreadable message. Say why instead.
        assert len(counts) >= 2, (
            f"fixture has a single n_blocked_receivers value {counts} -- nothing to correlate. "
            "FIX THE FIXTURE, do not skip."
        )
        # scipy types the result element as `object`; the repo idiom for an unavoidable
        # stub-vs-runtime gap is the targeted ignore (features.py does the same for `.at[]`).
        rho = float(spearmanr(both["n_blocked_receivers"].astype(float), both["blocking_score"])[0])  # type: ignore[arg-type]
        assert rho >= 0.5, f"blocking_score does not track n_blocked_receivers (rho={rho:.3f})"
