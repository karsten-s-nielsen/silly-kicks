"""Sharded xT fit as a ``for_each`` count pass reduced by ``ExpectedThreat.fit_from_counts`` (ADR-052 D15).

An xT fit is event-only and its per-match zone counts are ADDITIVE (ADR-102), so it is a corpus pass
like any other: per-match ``XtZoneCounts`` become sparse shards, the reduce sums them and calls
``fit_from_counts``, and the fit inherits resume-before-load, recorded failures, exclusions and the CI
gate. Byte-identical to a pooled ``fit(actions)`` (integer sums), so no retrain (spec §4.4).

Imports are QUALIFIED (``from scripts._xxx import ...``): consumers import this as
``scripts._xt_corpus`` from outside tests/scripts/, where the scripts/-on-sys.path scope that
``tests/scripts/conftest.py`` provides is absent, so a bare ``from _driver import ...`` would not resolve.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat, XtZoneCounts

if TYPE_CHECKING:
    from scripts._driver import CorpusPassResult
    from scripts._loader_pining import MatchRef


class _AdmissionRecordLike(Protocol):
    """The two fields `fit_xt_from_count_pass` reads off an ``AdmissionRecord`` (§4.1).

    Duck-typed rather than imported, so this module (Task 5) does not depend on
    ``scripts/_events_admission.py`` (Task 6); the real ``AdmissionRecord`` satisfies it structurally.
    Declared READ-ONLY (properties, not bare attributes): ``AdmissionRecord`` is a frozen dataclass, so
    its fields are read-only, which a writable protocol attribute would reject (reportArgumentType).
    """

    @property
    def digest(self) -> str | None: ...

    @property
    def unmeasured_admitted(self) -> tuple[str, ...]: ...


#: Sparse long-form count-shard schema. The four ZONE aggregates carry ``to_zone == -1``; the
#: ``transition`` aggregate carries a real ``(from_zone, to_zone)``. Only non-zero entries are stored.
COUNT_SHARD_COLUMNS = ["aggregate", "from_zone", "to_zone", "n"]
COUNT_SHARD_SCHEMA_VERSION = "xt-counts-1"

_ZONE_AGGREGATES = ("shot", "goal", "move", "transition_start")
_AGG_TO_ATTR = {
    "shot": "shot_counts",
    "goal": "goal_counts",
    "move": "move_counts",
    "transition_start": "transition_start_counts",
}


def counts_to_frame(counts: XtZoneCounts) -> pd.DataFrame:
    """One match's :class:`XtZoneCounts` as a SPARSE ``(aggregate, from_zone, to_zone, n)`` shard.

    The four ``(w, l)`` zone aggregates are flattened (row-major, matching ``fit_from_counts``'s
    ``.ravel()``) and stored with ``to_zone == -1``; the ``(w*l, w*l)`` transition matrix stores its
    non-zero ``(from, to)`` cells. Round-trips exactly through :func:`counts_from_frames`.
    """
    parts: list[pd.DataFrame] = []
    for name in _ZONE_AGGREGATES:
        flat = np.asarray(getattr(counts, _AGG_TO_ATTR[name])).ravel()
        z = np.flatnonzero(flat)
        parts.append(pd.DataFrame({"aggregate": name, "from_zone": z, "to_zone": -1, "n": flat[z]}))
    tc = np.asarray(counts.transition_counts)
    fr, to = np.nonzero(tc)
    parts.append(pd.DataFrame({"aggregate": "transition", "from_zone": fr, "to_zone": to, "n": tc[fr, to]}))
    out = pd.concat(parts, ignore_index=True)
    return out.astype({"aggregate": "object", "from_zone": "int64", "to_zone": "int64", "n": "int64"})


def counts_from_frames(frames: Iterable[pd.DataFrame], *, l: int, w: int) -> XtZoneCounts:
    """Sum sparse count shards back into one :class:`XtZoneCounts` on the ``l`` x ``w`` grid.

    Empty input (no shards, or only empty ones) yields the all-zero counts -- the correct identity for
    a pass that produced nothing.
    """
    n = w * l
    non_empty = [f for f in frames if len(f)]
    shot = np.zeros(n, dtype=np.int64)
    goal = np.zeros(n, dtype=np.int64)
    move = np.zeros(n, dtype=np.int64)
    tstart = np.zeros(n, dtype=np.int64)
    tc = np.zeros((n, n), dtype=np.int64)
    if non_empty:
        combined = pd.concat(non_empty, ignore_index=True)
        agg = combined.groupby(["aggregate", "from_zone", "to_zone"], as_index=False)["n"].sum()
        zone_targets = {"shot": shot, "goal": goal, "move": move, "transition_start": tstart}
        for name, target in zone_targets.items():
            sub = agg[agg["aggregate"] == name]
            target[sub["from_zone"].to_numpy()] = sub["n"].to_numpy()
        tsub = agg[agg["aggregate"] == "transition"]
        tc[tsub["from_zone"].to_numpy(), tsub["to_zone"].to_numpy()] = tsub["n"].to_numpy()
    return XtZoneCounts(
        l=l,
        w=w,
        shot_counts=shot.reshape(w, l),
        goal_counts=goal.reshape(w, l),
        move_counts=move.reshape(w, l),
        transition_start_counts=tstart.reshape(w, l),
        transition_counts=tc,
    )


@dataclasses.dataclass(frozen=True)
class XtFitProvenance:
    """What an xT count-pass fit consumed, for a provenance-stamped artifact (spec §4.4)."""

    fit_keys: tuple[str, ...]  # shards that contributed counts (this pass's shard keys)
    excluded: Mapping[str, str]  # key -> reason (from the pass)
    failed: Mapping[str, str]  # key -> error (from the pass)
    l: int
    w: int
    counts_digest: str  # digest of the summed counts
    admission_digest: str | None  # §8 verdict artifact consulted (events-only SkillCorner), else None
    allowed_failed: bool  # True iff the caller passed allow_failed=True
    unmeasured_admitted: tuple[str, ...]  # keys admitted under --allow-unmeasured (§4.1), else ()


def xt_count_pass(
    refs: Sequence[MatchRef],
    *,
    key: Callable[[MatchRef], object],
    load_actions: Callable[[MatchRef], pd.DataFrame],
    shard_root,
    token_inputs: Mapping[str, object],
    tag: str = "all",
    l: int = 16,
    w: int = 12,
    label: str = "match",
) -> CorpusPassResult:
    """Walk ``refs``, writing one sparse zone-count shard per match (spec §4.4).

    ``load_actions(ref)`` returns that match's SPADL actions (event-only -- for pining, the admitted
    ``events_only_loader``; for open data, ``load_open_data_match(ref).actions``). The shard is
    ``counts_to_frame(ExpectedThreat(l, w).zone_counts(actions))``, so it inherits resume-before-load,
    recorded failures and exclusions from ``for_each``. ``events_only``, ``l``, ``w`` and the count-shard
    schema version join the token (the 4.77.1 schema/token pair).
    """
    from scripts._driver import for_each

    model = ExpectedThreat(l=l, w=w)
    token = {**dict(token_inputs), "events_only": True, "count_schema": COUNT_SHARD_SCHEMA_VERSION, "l": l, "w": w}
    return for_each(
        refs,
        key=key,
        load=load_actions,
        work=lambda actions: counts_to_frame(model.zone_counts(actions)),
        shard_root=shard_root,
        token_inputs=token,
        tag=tag,
        label=label,
    )


def _counts_digest(counts: XtZoneCounts) -> str:
    """A stable digest of the summed counts -- provenance for the fitted grid."""
    h = hashlib.sha256()
    h.update(f"{counts.l}x{counts.w}".encode())
    for name in ("shot_counts", "goal_counts", "move_counts", "transition_start_counts", "transition_counts"):
        h.update(np.ascontiguousarray(getattr(counts, name), dtype=np.int64).tobytes())
    return h.hexdigest()


def fit_xt_from_count_pass(
    res: CorpusPassResult,
    *,
    l: int = 16,
    w: int = 12,
    allow_failed: bool = False,
    admission: _AdmissionRecordLike | None = None,
) -> tuple[ExpectedThreat, XtFitProvenance]:
    """Reduce a :func:`xt_count_pass` result to a fitted grid + provenance.

    Sums the shards of ``res.shard_keys`` (an excluded match has a marker, not a parquet) and calls
    ``fit_from_counts`` -- byte-identical to a pooled ``fit``. REFUSES by default when the pass had
    failures (CDLS-SPEC-17): a failure is transient (a resume retries it) or a data defect (a
    deliberate decision), never a silent shrink. ``allow_failed=True`` fits without them and RECORDS
    the failed keys in the provenance -- the ``--allow-dirty`` idiom.
    """
    from scripts._driver import shard_path

    if res.failures and not allow_failed:
        raise RuntimeError(
            f"{len(res.failures)} match(es) failed the count pass: {sorted(res.failures)}. Re-run to "
            f"retry only them (a failure wrote no shard), or pass allow_failed=True to fit without "
            f"them (recorded in XtFitProvenance)."
        )
    frames = [pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys]
    total = counts_from_frames(frames, l=l, w=w)
    xt = ExpectedThreat(l=l, w=w).fit_from_counts(**total.as_fit_kwargs())
    prov = XtFitProvenance(
        fit_keys=tuple(res.shard_keys),
        excluded=dict(res.exclusions),
        failed=dict(res.failures),
        l=l,
        w=w,
        counts_digest=_counts_digest(total),
        admission_digest=(admission.digest if admission is not None else None),
        allowed_failed=allow_failed,
        unmeasured_admitted=(tuple(admission.unmeasured_admitted) if admission is not None else ()),
    )
    return xt, prov
