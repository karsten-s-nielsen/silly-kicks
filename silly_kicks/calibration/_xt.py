"""Frozen exogenous xT artifact for the TF-24 calibration harness (C2, Option 1).

xT is a fixed upstream feature *extractor* — at deployment the calibrated tracking defaults run
against ONE fixed league-level xT grid, never a per-match refit. So we fit ``ExpectedThreat`` ONCE
on a corpus DISJOINT from the calibration matches, freeze it as a checksummed artifact, and use
that single grid for every match/fold/trial. This removes the held-out leak (the grid never sees a
calibration action) and gives a cleaner TPE signal (xT injects zero fold-structure variance).

See NOTICE for the xT (Decroos / Van Roy) citation. Spec §4a.

Examples
--------
Fit, freeze, and reload a calibration xT grid::

    from silly_kicks.calibration._xt import fit_frozen_xt, save_xt, load_xt

    frozen = fit_frozen_xt(corpus_actions, exclude_match_ids={"game_42"},
                           match_id_col="game_id", source="bronze.spadl_actions")
    save_xt(frozen, "calibration_xt.npz")
    frozen = load_xt("calibration_xt.npz")  # sha256-verified
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat


def _grid_sha256(grid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class FrozenXt:
    """A frozen, checksummed xT grid + its provenance (for the calibration manifest).

    Examples
    --------
    Fit a frozen xT artifact from a corpus, excluding the calibration matches::

        from silly_kicks.calibration._xt import fit_frozen_xt

        frozen = fit_frozen_xt(corpus, exclude_match_ids={"g1"}, source="x")
        frozen.sha256, frozen.n_excluded, frozen.manifest()
    """

    xt: ExpectedThreat
    source: str
    corpus_match_ids: tuple[str, ...]
    n_excluded: int  # calibration matches actually removed from the corpus (H2 audit)
    fit_date: str  # ISO date; supplied by the caller (pure core does not read the clock)
    grid_shape: tuple[int, int]
    sha256: str

    def manifest(self) -> dict:
        """Provenance dict for the report (§6 R3) — JSON-serialisable, no grid payload.

        The grid itself is deliberately absent: the manifest goes into the calibration
        report, where what matters is WHICH grid was used (``sha256``) and what it was fit
        on, not the values. ``n_excluded`` is the H2 audit number — the count of calibration
        matches actually removed from the corpus, which is what makes the no-leak claim
        checkable after the fact rather than merely asserted.

        Examples
        --------
        >>> from silly_kicks.calibration._xt import FrozenXt
        >>> from silly_kicks.xthreat import ExpectedThreat
        >>> frozen = FrozenXt(
        ...     xt=ExpectedThreat(),
        ...     source="bronze.spadl_actions",
        ...     corpus_match_ids=("g1", "g2"),
        ...     n_excluded=1,
        ...     fit_date="2026-07-18",
        ...     grid_shape=(16, 12),
        ...     sha256="ab12",
        ... )
        >>> frozen.manifest()["n_corpus_matches"]
        2
        >>> "xt" in frozen.manifest()  # the grid never enters the report
        False
        """
        return {
            "source": self.source,
            "corpus_match_ids": list(self.corpus_match_ids),
            "n_corpus_matches": len(self.corpus_match_ids),
            "n_excluded": self.n_excluded,
            "fit_date": self.fit_date,
            "grid_shape": list(self.grid_shape),
            "sha256": self.sha256,
        }


def fit_frozen_xt(
    corpus_actions: pd.DataFrame,
    *,
    exclude_match_ids: Iterable,
    match_id_col: str = "game_id",
    source: str,
    fit_date: str = "",
) -> FrozenXt:
    """Fit ``ExpectedThreat`` on ``corpus_actions`` MINUS ``exclude_match_ids`` and freeze it.

    The exclusion is the whole point — it guarantees the calibration matches never enter the xT
    grid (zero leak). **Fails CLOSED (H2):** if any excluded id is NOT present in the corpus, the
    exclusion silently did nothing (an id-space mismatch — pining match_id vs bronze game_id),
    which would reintroduce the leak. So we require EVERY excluded id to be found and removed. Also
    raises if the disjoint corpus is empty.

    Examples
    --------
    Fit once on a corpus that excludes every calibration match, then reuse the frozen grid
    for every match / fold / trial in the study::

        frozen = fit_frozen_xt(
            corpus_actions,
            exclude_match_ids={m.match_id for m in calibration_matches},
            match_id_col="game_id",
            source="bronze.spadl_actions",
            fit_date="2026-07-18",
        )
        assert frozen.n_excluded == len(calibration_matches)  # H2 audit

    The exclusion FAILS CLOSED rather than no-opping, because an exclusion that silently
    matched nothing is indistinguishable from no exclusion at all — and would put the
    held-out matches straight back into the grid:

    >>> import pandas as pd
    >>> from silly_kicks.calibration._xt import fit_frozen_xt
    >>> corpus = pd.DataFrame({"game_id": ["g1", "g2"]})
    >>> fit_frozen_xt(corpus, exclude_match_ids={"g3"}, source="demo")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: xT-corpus exclusion is unsafe: 1/1 calibration match ids were NOT found...
    """
    excluded = {str(m) for m in exclude_match_ids}
    corpus_ids = {str(m) for m in corpus_actions[match_id_col].unique()}
    found = excluded & corpus_ids
    if excluded and len(found) < len(excluded):
        missing = sorted(excluded - corpus_ids)
        raise ValueError(
            f"xT-corpus exclusion is unsafe: {len(missing)}/{len(excluded)} calibration match ids "
            f"were NOT found in corpus[{match_id_col!r}] (e.g. {missing[:5]}). The id spaces differ "
            "(pining match_id vs bronze game_id?) — the exclusion would no-op and LEAK held-out "
            "matches into the xT fit. Map the ids to a common space before fitting."
        )
    keep = corpus_actions[~corpus_actions[match_id_col].astype(str).isin(excluded)]
    remaining = tuple(sorted(str(m) for m in keep[match_id_col].unique()))
    if len(keep) == 0 or not remaining:
        raise ValueError(
            "disjoint corpus is empty after excluding calibration matches — "
            "supply a larger corpus or a smaller exclusion set"
        )
    xt = ExpectedThreat().fit(keep)
    grid = np.asarray(xt.xT, dtype=np.float64)
    return FrozenXt(
        xt=xt,
        source=source,
        corpus_match_ids=remaining,
        n_excluded=len(found),
        fit_date=fit_date,
        grid_shape=(int(grid.shape[0]), int(grid.shape[1])),
        sha256=_grid_sha256(grid),
    )


def save_xt(frozen: FrozenXt, path: str | Path) -> None:
    """Serialise a ``FrozenXt`` to ``path`` (npz grid + JSON-sidecar provenance in one file).

    ONE file, two arrays: the grid and the manifest as embedded JSON. Keeping the provenance
    inside the artifact is what lets :func:`load_xt` re-check the checksum against the grid
    it actually loaded — a sidecar the caller could lose or swap would make that check
    meaningless.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import numpy as np
    >>> from silly_kicks.calibration._xt import FrozenXt, save_xt
    >>> from silly_kicks.xthreat import ExpectedThreat
    >>> xt = ExpectedThreat()
    >>> xt.xT = np.zeros((16, 12))
    >>> frozen = FrozenXt(
    ...     xt=xt,
    ...     source="bronze.spadl_actions",
    ...     corpus_match_ids=("g1",),
    ...     n_excluded=0,
    ...     fit_date="2026-07-18",
    ...     grid_shape=(16, 12),
    ...     sha256="ab12",
    ... )
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "calibration_xt.npz"
    >>> save_xt(frozen, path)
    >>> sorted(np.load(path, allow_pickle=True).files)
    ['meta_json', 'xT']
    """
    meta = frozen.manifest()
    np.savez(
        path,
        xT=np.asarray(frozen.xt.xT, dtype=np.float64),
        meta_json=np.array(json.dumps(meta)),
    )


def load_xt(path: str | Path) -> FrozenXt:
    """Load a ``FrozenXt`` from ``path``, re-checking the grid sha256 against the stored value.

    The reloaded ``FrozenXt`` carries a grid-only ``ExpectedThreat``: ``.xT`` is set
    directly and nothing is re-fit, which is the whole point of freezing. The checksum is
    verified against the grid as loaded, so a post-fit edit is a REFUSAL, not a quietly
    different calibration run:

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import numpy as np
    >>> from silly_kicks.calibration._xt import FrozenXt, load_xt, save_xt
    >>> from silly_kicks.xthreat import ExpectedThreat
    >>> xt = ExpectedThreat()
    >>> xt.xT = np.zeros((16, 12))
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "tampered.npz"
    >>> save_xt(  # a digest that does not match the grid stands in for a post-fit edit
    ...     FrozenXt(
    ...         xt=xt,
    ...         source="demo",
    ...         corpus_match_ids=("g1",),
    ...         n_excluded=0,
    ...         fit_date="2026-07-18",
    ...         grid_shape=(16, 12),
    ...         sha256="ab12",
    ...     ),
    ...     path,
    ... )
    >>> load_xt(path)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: xT artifact sha256 mismatch (stored ab12, recomputed ...
    """
    data = np.load(path, allow_pickle=True)
    grid = np.asarray(data["xT"], dtype=np.float64)
    meta = json.loads(str(data["meta_json"]))
    recomputed = _grid_sha256(grid)
    if recomputed != meta["sha256"]:
        raise ValueError(
            f"xT artifact sha256 mismatch (stored {meta['sha256']}, recomputed {recomputed}) — "
            "the grid was modified after fitting; refuse to load a tampered artifact"
        )
    xt = ExpectedThreat()
    xt.xT = grid  # inference uses .xT directly; gk_influence/cover_shadows read the grid
    return FrozenXt(
        xt=xt,
        source=meta["source"],
        corpus_match_ids=tuple(meta["corpus_match_ids"]),
        n_excluded=int(meta.get("n_excluded", 0)),
        fit_date=meta["fit_date"],
        grid_shape=tuple(meta["grid_shape"]),
        sha256=meta["sha256"],
    )
