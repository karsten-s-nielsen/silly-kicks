"""The registered corpus taxonomy (spec 3.2).

Public-vs-owner is keyed on the manifest's `visibility` field, NEVER on the provider name. The
98 owner-tier SkillCorner matches added in 2026-07 carry provider `skillcorner`; the old rule
(the deleted provider-name allowlist `{"skillcorner", "idsse"}`) would absorb them into the PUBLIC
arm and ship a model trained on non-redistributable data under a `public` label. That rule is gone.
"""

from __future__ import annotations

import numpy as np

# The 17 matches we may redistribute. Drift here fails the run loudly (spec 3.2).
PUBLIC_CORPUS: dict[str, frozenset[str]] = {
    "skillcorner": frozenset(
        {"1886347", "1899585", "1925299", "1953632", "1996435", "2006229", "2011166", "2013725", "2015213", "2017461"}
    ),
    "idsse": frozenset(
        {
            "DFL-MAT-J03WMX",
            "DFL-MAT-J03WN1",
            "DFL-MAT-J03WOH",
            "DFL-MAT-J03WOY",
            "DFL-MAT-J03WPY",
            "DFL-MAT-J03WQQ",
            "DFL-MAT-J03WR9",
        }
    ),
}


def is_public_row(
    *, providers: np.ndarray, match_ids: np.ndarray, visibility: dict[tuple[str, str], str]
) -> np.ndarray:
    """Per-row public mask. FAIL-CLOSED: an absent (provider, match) is RESTRICTED."""
    return np.array(
        [visibility.get((str(p), str(m)), "private") == "public" for p, m in zip(providers, match_ids, strict=True)],
        dtype=bool,
    )


def artifact_label(*, providers: set[str], all_public: bool) -> str:
    """The shipped artifact's label, derived from the SHIP MASK's composition -- not from names."""
    if all_public:
        return "public"
    if "gradientsports" in providers:
        return "full"
    return "sc_extended"


def assert_public_corpus(visibility: dict[tuple[str, str], str], *, expect_full_public_arm: bool = False) -> None:
    """No match may claim `public` unless it is one of the registered 17 (spec 3.2, reviewer m4).

    SUBSET by default (nothing unregistered may call itself public -- a LICENSING failure). Equality
    only when expect_full_public_arm=True, the maintainer run that loads every public provider (the
    registered set must all be present -- a DRIFT failure). An unconditional equality check would
    SystemExit on every legitimate partial run (a two-match test corpus, a GS-only run, a smoke).
    """
    seen = {(p, m) for (p, m), v in visibility.items() if v == "public"}
    registered = {(prov, mid) for prov, ids in PUBLIC_CORPUS.items() for mid in ids}
    unregistered = seen - registered
    if unregistered:
        raise SystemExit(
            f"UNREGISTERED public match(es): {sorted(unregistered)}. A match claiming `public` that "
            "is not in PUBLIC_CORPUS would enter the redistributable training arm. Refusing to run."
        )
    if expect_full_public_arm and seen != registered:
        raise SystemExit(
            f"PUBLIC_CORPUS drift: missing {sorted(registered - seen)}. The registered public set "
            "must be fully present in a maintainer run -- a change here alters what 'public' means."
        )
