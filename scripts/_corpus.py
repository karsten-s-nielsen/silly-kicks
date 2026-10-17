"""The registered corpus taxonomy (spec 3.2).

Public-vs-owner is keyed on the manifest's `visibility` field, NEVER on the provider name. The
98 owner-tier SkillCorner matches added in 2026-07 carry provider `skillcorner`; the old rule
(the deleted provider-name allowlist `{"skillcorner", "idsse"}`) would absorb them into the PUBLIC
arm and ship a model trained on non-redistributable data under a `public` label. That rule is gone.
"""

from __future__ import annotations

import numpy as np

from scripts._corpus_identity import match_id_pairs

# The 27 matches we may redistribute. Drift here fails the run loudly (spec 3.2). SkillCorner grew
# 10 -> 20 on 2026-09-09 (a second public drop, uploaded to pining-for-the-data); the pre-existing
# bundled xshot/xcross/ghost models were trained on the ORIGINAL public 17 (10 SkillCorner + 7 IDSSE)
# and are NOT retrained by this change -- their model cards' "17 matches" is a correct historical fact.
PUBLIC_CORPUS: dict[str, frozenset[str]] = {
    "skillcorner": frozenset(
        {
            "1874553",
            "1886347",
            "1899585",
            "1925299",
            "1927964",
            "1953632",
            "1959846",
            "1986691",
            "1996435",
            "1996436",
            "2006229",
            "2006363",
            "2007448",
            "2007721",
            "2010085",
            "2011166",
            "2013725",
            "2015213",
            "2016236",
            "2017461",
        }
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
    """No match may claim `public` unless it is one of the registered 27 (spec 3.2, reviewer m4).

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


# The 17 matches the wheel-bundled PUBLIC-arm models were trained on: xshot / xcross `default` and
# `position_only` (all 17) and gk_completion `skillcorner` (the 10 SkillCorner ids). This is
# PUBLIC_CORPUS as it stood before the 2026-09-10 SkillCorner growth (ce0401a^). Re-fits pin THIS set;
# widening to the current PUBLIC_CORPUS is a separate, gated decision (combined-cycle-completion D4).
BUNDLED_PUBLIC_ARM: dict[str, tuple[str, ...]] = {
    "skillcorner": (
        "1886347",
        "1899585",
        "1925299",
        "1953632",
        "1996435",
        "2006229",
        "2011166",
        "2013725",
        "2015213",
        "2017461",
    ),
    "idsse": (
        "DFL-MAT-J03WMX",
        "DFL-MAT-J03WN1",
        "DFL-MAT-J03WOH",
        "DFL-MAT-J03WOY",
        "DFL-MAT-J03WPY",
        "DFL-MAT-J03WQQ",
        "DFL-MAT-J03WR9",
    ),
}


def bundled_public_arm_pairs(providers: tuple[str, ...] = ("idsse", "skillcorner")) -> list[list[str]]:
    """``BUNDLED_PUBLIC_ARM`` restricted to ``providers``, in the :func:`match_id_pairs` shape."""
    provs = [p for p in providers for _ in BUNDLED_PUBLIC_ARM[p]]
    mids = [m for p in providers for m in BUNDLED_PUBLIC_ARM[p]]
    return match_id_pairs(provs, mids)


def requested_is_all_public(pairs, visibility: dict[tuple[str, str], str]) -> bool:
    """True iff the requested corpus is non-empty and every requested match is public (fail-closed)."""
    if not pairs:
        return False
    return bool(
        is_public_row(
            providers=np.asarray([p for p, _ in pairs]),
            match_ids=np.asarray([m for _, m in pairs]),
            visibility=visibility,
        ).all()
    )
