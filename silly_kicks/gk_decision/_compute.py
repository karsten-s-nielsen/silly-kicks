"""compute_gk_decision_value -- tier-agnostic engine over an OptionSet; conserving; + summarize_*.

Groups on the RAW option rows so EVERY input decision is counted (ADR-042 dropped-and-counted); NaN-EV
handling is EXPLICIT (a NaN chosen -> counted ``chosen_unvalued``; NaN alternatives -> finite-subset
scoring, or ``too_few_options`` if that drops below ``min_options``), never a silent pre-filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id_series

from ._columns import GK_DECISION_SAMPLE_COLUMNS
from ._config import GkDecisionParams
from ._optionset import OptionSet
from ._report import GkDecisionReport
from ._value import option_value

_DEFAULT_PARAMS = GkDecisionParams()


def compute_gk_decision_value(
    option_set: OptionSet,
    *,
    params: GkDecisionParams = _DEFAULT_PARAMS,
    extra_drops: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, GkDecisionReport]:
    """Score each GK build-up decision as chosen-vs-available. Returns (samples, report).

    The reconstruction adapter's ``no_frame`` / ``fov_cropped`` counts (PR2) are folded into the Report
    AND into ``n_decisions_in`` so the census stays the TRUE decision population -- a decision the adapter
    drops before it yields any option row would otherwise be uncounted. They are pulled from the
    option_set's ``drop_counts()`` automatically (AFTER ``option_rows()`` runs); pass ``extra_drops``
    only to override. The native path (no ``drop_counts``) is byte-identical to Phase 1.

    Examples
    --------
    Score a native SkillCorner option set (built from parsed GI ``passing_option`` rows)::

        from silly_kicks.gk_decision import SkillCornerGIOptionSet, compute_gk_decision_value
        option_set = SkillCornerGIOptionSet(parsed_options, keeper_ids=roster_gk_ids)
        samples, report = compute_gk_decision_value(option_set)
        report.n_scored  # decisions with >= 3 valued options and a unique chosen option
    """
    rows = option_set.option_rows().copy()
    rows["ev"] = option_value(rows, params=params).to_numpy()
    rows["_chosen"] = rows["is_chosen"].astype(bool)
    n_in = n_scored = n_few = n_nochosen = n_unvalued = 0
    out: list[dict] = []
    for (_g, _d), g in rows.groupby(["game_id", "decision_id"], sort=False):
        n_in += 1
        if int(g["_chosen"].sum()) != 1:  # 0 or >1 chosen in the input
            n_nochosen += 1
            continue
        chosen_ev = float(g.loc[g["_chosen"], "ev"].iloc[0])
        if not np.isfinite(chosen_ev):  # the unique chosen option is unvalued (NaN EV)
            n_unvalued += 1
            continue
        valid = g[np.isfinite(g["ev"].to_numpy())]  # chosen kept; NaN-EV ALTERNATIVES dropped (finite-subset)
        if len(valid) < params.min_options:
            n_few += 1
            continue
        ev = valid["ev"].to_numpy(dtype="float64")
        ev_alt = valid.loc[~valid["_chosen"], "ev"].to_numpy(dtype="float64")
        best = float(ev.max())
        dpct = float((np.sum(ev_alt < chosen_ev) + 0.5 * np.sum(ev_alt == chosen_ev)) / len(ev_alt))
        first = valid.iloc[0]
        out.append(
            dict(
                game_id=first["game_id"],
                period_id=first["period_id"],
                decision_id=_d,
                keeper=canonical_id_series(valid["keeper_id"]).iloc[0],
                keeper_raw=first["keeper_id"],
                team_id=first["team_id"],
                decision_value=chosen_ev - float(ev.mean()),
                chosen_ev=chosen_ev,
                best_ev=best,
                sel_efficiency=(chosen_ev / best) if best > 0 else np.nan,
                decision_pct=dpct,
                n_options=len(valid),
                option_set_source=first["option_set_source"],
            )
        )
        n_scored += 1
    # Adapter drops (no_frame / fov_cropped) are read AFTER option_rows() ran above -- an inline
    # `extra_drops=os.drop_counts()` at the call site would capture the pre-run zeros. An explicit
    # extra_drops still overrides. SkillCornerGIOptionSet has no drop_counts -> {} -> Phase-1 identical.
    _dc = getattr(option_set, "drop_counts", None)
    if extra_drops is None and _dc is not None:
        extra_drops = _dc()
    _extra = extra_drops or {}
    n_no_frame = int(_extra.get("no_frame", 0))
    n_fov_cropped = int(_extra.get("fov_cropped", 0))
    samples = pd.DataFrame(out, columns=list(GK_DECISION_SAMPLE_COLUMNS))
    report = GkDecisionReport(
        params,
        n_decisions_in=n_in + n_no_frame + n_fov_cropped,
        n_scored=n_scored,
        n_too_few_options=n_few,
        n_no_unique_chosen=n_nochosen,
        n_chosen_unvalued=n_unvalued,
        n_no_frame=n_no_frame,
        n_fov_cropped=n_fov_cropped,
    )
    return samples, report


def summarize_gk_decision(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-``(keeper, game_id)`` means + decision counts (the coach-facing aggregate).

    Examples
    --------
    Aggregate the per-decision samples to one row per keeper-match::

        from silly_kicks.gk_decision import summarize_gk_decision
        per_keeper_match = summarize_gk_decision(samples)
        per_keeper_match[["keeper", "game_id", "n_decisions", "decision_pct_mean"]]
    """
    g = samples.groupby(["keeper", "game_id"], dropna=False)
    return g.agg(
        n_decisions=("decision_value", "size"),
        decision_value_mean=("decision_value", "mean"),
        sel_efficiency_mean=("sel_efficiency", "mean"),
        decision_pct_mean=("decision_pct", "mean"),
    ).reset_index()
