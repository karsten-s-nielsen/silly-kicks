"""OptionSet port + the native SkillCorner GI adapter. Reconstruction adapters land in PR2."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from silly_kicks.id_compat import ids_isin

from ._columns import OPTION_ROW_COLUMNS


@runtime_checkable
class OptionSet(Protocol):
    """A provider-agnostic source of GK-decision option sets.

    ``option_rows`` yields the uniform table (:data:`OPTION_ROW_COLUMNS`); exactly one ``is_chosen``
    row per ``decision_id``. The engine (:func:`compute_gk_decision_value`) consumes only this, so it
    is fully decoupled from provider data.

    Examples
    --------
    Any object yielding the uniform option-rows table satisfies the port::

        class MyOptionSet:
            def option_rows(self):
                return df  # cols: game_id, period_id, decision_id, keeper_id, team_id,
                #                  is_chosen, completion, opponents_bypassed, option_set_source
    """

    def option_rows(self) -> pd.DataFrame:
        """Return the uniform option-rows table (:data:`OPTION_ROW_COLUMNS`).

        Examples
        --------
        Consume the port from an adapter instance::

            rows = option_set.option_rows()  # one row per option; exactly one is_chosen per decision
        """
        ...


class SkillCornerGIOptionSet:
    """Native tier: SkillCorner GI ``passing_option`` rows -> the uniform option-rows schema.

    ``parsed_options`` is :func:`silly_kicks.providers.skillcorner.parse_passing_options` output;
    ``keeper_ids`` (roster GK ids) restrict to GK build-up possessions. The native option set is already
    curated, so no reachability filter is applied (``option_set_source == "native"``).

    Examples
    --------
    Wrap parsed SkillCorner GI rows as an option set for the engine::

        from silly_kicks.gk_decision import SkillCornerGIOptionSet, compute_gk_decision_value
        option_set = SkillCornerGIOptionSet(parsed_options, keeper_ids=roster_gk_ids)
        samples, report = compute_gk_decision_value(option_set)
    """

    def __init__(self, parsed_options: pd.DataFrame, *, keeper_ids) -> None:
        self._parsed = parsed_options
        self._keeper_ids = list(keeper_ids)

    def option_rows(self) -> pd.DataFrame:
        """Uniform option-rows for the GK-possession options in the parsed GI table.

        Examples
        --------
        Restrict parsed GI rows to keeper possessions and map to the uniform schema::

            rows = SkillCornerGIOptionSet(parsed_options, keeper_ids=[99]).option_rows()
        """
        p = self._parsed
        gk = p[ids_isin(p["possessor_id"], self._keeper_ids).to_numpy()].copy()
        out = pd.DataFrame(
            {
                "game_id": gk["game_id"],
                "period_id": gk["period_id"],
                "decision_id": gk["decision_id"],
                "keeper_id": gk["possessor_id"],
                "team_id": gk["team_id"],
                "is_chosen": gk["is_chosen"].astype(bool),
                "completion": gk["completion"],
                "opponents_bypassed": gk["opponents_bypassed"],
                "option_set_source": "native",
            }
        )
        return out.reset_index(drop=True)[list(OPTION_ROW_COLUMNS)]
