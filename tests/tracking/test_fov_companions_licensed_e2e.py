"""FOV-observability companions on the REAL licensed SB360 corpus (owner-run, e2e; ADR-077, Task 9).

The committed 6-frame golden slice already exercises the wiring in CI (``test_fov_*.py``,
``test_validate_sb360_licensed_corpus.py``). This is the LICENSED backstop: on the real 30-match
StatsBomb-360 corpus it asserts the opt-in ``visible_area`` companions are present and SANELY
distributed on ``fov_cropped`` matches -- a fraction in ``[0, 1]`` exactly where the source is
``observed``, NaN everywhere else, and at least one cropped action with a fraction strictly ``< 1`` (so
the signal is genuinely live on real FOV crops, not a full-coverage rubber stamp). It also re-asserts
the additive guarantee (primary columns byte-identical with and without ``visible_area``).

Licensed data can never be in CI, so this is marked ``e2e`` (deselected in the normal suite) and
self-skips when the pining corpus is unreachable (no owner token / network / empty corpus). Mirrors
``scripts/validate_sb360_licensed_corpus.py``'s ``load_statsbomb_matches`` load path.
"""

from __future__ import annotations

import inspect
import os

import pytest

pytestmark = pytest.mark.e2e

#: Geometry/positional companioned aggregators that score on velocity-less freeze-frames with only
#: (actions, frames, links, visible_area) -- no fitted model / xt / xg input. Deliberately excludes
#: add_player_influence / add_xt_gk / add_defensive_credit (model/grid inputs) to keep the e2e robust;
#: their companions are covered by the committed FOV unit tests + the licensed-corpus driver.
_MODEL_FREE_COMPANIONED = (
    "add_action_context",
    "add_pressure_on_actor",
    "add_defensive_line",
    "add_team_shape",
    "add_packing",
)


def _load_licensed_matches(max_matches):
    """Yield ``(match_id, actions, frames, visible_area, home)`` from the licensed SB360 corpus.

    Self-skips (via ``pytest.skip``) when the owner pining corpus is unreachable for ANY reason -- no
    token, no network, or an empty manifest -- because this is an owner-run backstop, never a CI gate.
    """
    # Fast, network-free skip: the licensed SB360 corpus is owner-only, and the loader falls back to a
    # PUBLIC token that would hit the network and yield nothing for the licensed statsbomb corpus. Skip
    # up front when the owner token env var is absent so a non-owner run never blocks on the network.
    if not os.environ.get("PINING_FOR_THE_DATA_TOKEN"):
        pytest.skip("PINING_FOR_THE_DATA_TOKEN not set -- licensed SB360 corpus is owner-run only")
    try:
        from scripts._loader_pining import load_statsbomb_matches

        loaded = list(load_statsbomb_matches(max_matches=max_matches))
    except Exception as exc:  # no token / network / manifest -- owner-run only
        pytest.skip(f"licensed SB360 corpus unavailable: {exc}")
    if not loaded:
        pytest.skip("licensed SB360 corpus resolved to zero matches (no owner token?)")
    return [(mid, actions, frames, visible_area, home) for _prov, mid, actions, frames, home, visible_area in loaded]


def _call_with_supported_kwargs(fn, actions, *, frames, links, visible_area, home):
    """Invoke a companioned aggregator, passing only the kwargs its signature actually accepts."""
    params = inspect.signature(fn).parameters
    kwargs: dict = {}
    if "frames" in params:
        kwargs["frames"] = frames
    if "links" in params:
        kwargs["links"] = links
    if "visible_area" in params:
        kwargs["visible_area"] = visible_area
    if "home_team_id" in params:
        kwargs["home_team_id"] = home
    return fn(actions, **kwargs)


def _companion_pairs(out) -> list[tuple[str, str]]:
    """``(fraction_col, source_col)`` pairs for every FOV companion present on ``out``."""
    pairs = []
    for col in out.columns:
        if col.endswith("_observed_fraction"):
            src = col[: -len("_observed_fraction")] + "_observed_source"
            if src in out.columns:
                pairs.append((col, src))
    return pairs


def test_fov_companions_are_live_and_honest_on_the_licensed_corpus():
    import pandas as pd

    import silly_kicks.tracking as T
    from silly_kicks.tracking import link_actions_to_frames, validate_fov
    from silly_kicks.tracking.features import add_action_context

    matches = _load_licensed_matches(max_matches=30)

    n_cropped = 0
    any_partial_across_corpus = False
    any_companion_scored = False
    # A model-free aggregator may legitimately REFUSE a match (no in-domain actions, an unresolved
    # defended end -> GoalEndUnresolvedError, both ValueError); that is skipped robustly. Any OTHER
    # exception is an API break / real regression -- recorded here and asserted-none at the end, so
    # the broad-swallow that hid it is gone WITHOUT losing the owner-run robustness.
    unexpected_raises: list[str] = []

    for match_id, actions, frames, visible_area, home in matches:
        # Narrow the loader's ``object`` visible_area to a DataFrame (statsbomb always carries one);
        # ``isinstance`` is False for None, so it covers the absent case too.
        if not isinstance(visible_area, pd.DataFrame) or not len(visible_area):
            continue

        # FOV regime for the whole match: this backstop is about the cropped matches (the SB360 case).
        diag = validate_fov(visible_area, on_mismatch="warn")
        if diag.regime != "fov_cropped":
            continue
        n_cropped += 1

        # Pre-link ONCE; freeze-frame link rate is legitimately partial (ignore, not an error).
        links = link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]

        # Additive guarantee: primary columns byte-identical with and without visible_area.
        base = add_action_context(actions, frames, links=links)
        withva = add_action_context(actions, frames, links=links, visible_area=visible_area)
        for c in ("nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"):
            assert base[c].equals(withva[c]), f"{match_id}: {c} changed when visible_area was supplied"

        for name in _MODEL_FREE_COMPANIONED:
            fn = getattr(T, name)
            try:
                out = _call_with_supported_kwargs(
                    fn, actions, frames=frames, links=links, visible_area=visible_area, home=home
                )
            except ValueError:
                # Legitimate domain refusal (no in-domain actions, an unresolved defended end);
                # robustly skipped. The corpus-level assertions below still bind on what DID score,
                # and add_action_context above is the loud anchor.
                continue
            except Exception as exc:  # NOT a ValueError -> an API break / real regression: record it
                # instead of swallowing, and continue so the corpus-level assertions still run.
                unexpected_raises.append(f"{match_id}:{name}: {type(exc).__name__}: {exc}")
                continue

            for frac_col, src_col in _companion_pairs(out):
                any_companion_scored = True
                frac = out[frac_col]
                src = out[src_col]

                # A fraction exists IFF the source is 'observed'; NaN everywhere else -- never fabricated.
                observed = src == "observed"
                obs_frac = frac[observed]
                assert obs_frac.notna().all(), f"{match_id}:{frac_col} NaN where source=='observed'"
                assert ((obs_frac >= 0.0) & (obs_frac <= 1.0)).all(), f"{match_id}:{frac_col} out of [0,1]"
                assert frac[~observed].isna().all(), f"{match_id}:{frac_col} non-NaN where source!='observed'"

                # Live signal: at least one partially-observed action somewhere in the corpus.
                if ((obs_frac > 0.0) & (obs_frac < 1.0)).any():
                    any_partial_across_corpus = True

    if n_cropped == 0:
        pytest.skip("no fov_cropped match in the loaded licensed corpus -- nothing to assert")
    # The corpus-level floor for the looped aggregators (any one may skip on a given match): these two
    # asserts are the ONLY guarantee here that a companion was scored + is live. Per-aggregator presence
    # for the model-input families deliberately left out of this loop (add_player_influence / add_xt_gk /
    # add_defensive_credit) is covered by the committed FOV unit tests, not this owner-run backstop.
    assert any_companion_scored, "no FOV companion column was scored on any cropped licensed match"
    assert any_partial_across_corpus, (
        "every observed companion fraction was exactly 0 or 1 across the licensed corpus -- "
        "the FOV signal is a rubber stamp, not live on real crops"
    )
    # A non-ValueError from any model-free aggregator is a real regression, not a legitimate refusal.
    assert not unexpected_raises, "model-free FOV aggregator(s) raised unexpectedly on real data: " + "; ".join(
        unexpected_raises
    )
