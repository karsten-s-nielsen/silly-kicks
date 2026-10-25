"""Shared support for the SkillCorner GI period-label repair tests (TF-46 Part 2 spec section 26; DL ruling
2026-10-25, Q8).

- :func:`load_opendata` reads one committed opendata slim fixture (events + ``match.json``).
- :func:`canonical_digests` hashes a frame column by column in a pandas-version-neutral form (values, not reprs), so
  a golden pins the ``1007ef6`` conversion under pandas 2 and 3 alike.
- :func:`mislabel_p2_to_p1` / :func:`mislabel_p1_to_p2` plant the observed defect (and its reverse) on a clean match:
  the boundary possession block takes the other period's label and a clock on that label's basis, exactly as GI
  computes a label-derived clock (the inverse of the section 26.2 re-base).

Running this module writes ``_skillcorner_period_golden_1007ef6.json``: the per-column digests of the
``1007ef6`` SkillCorner conversion and ``parse_passing_options`` output of every opendata fixture. It was run ONCE,
on the ``1007ef6`` library, and must not be re-run on a later library (the golden pins the pre-repair output).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OPENDATA = Path(__file__).parents[1] / "datasets" / "skillcorner" / "opendata"
GOLDEN = Path(__file__).parent / "_skillcorner_period_golden_1007ef6.json"

#: The GI frame-clock event types the repair applies to (spec section 26.2.6).
FRAME_CLOCK_TYPES = ("player_possession", "passing_option", "off_ball_run")
_OFFSET_DS = {1: 0, 2: 27000, 3: 54000, 4: 63000, 5: 72000}  # period start offsets in deciseconds


def opendata_ids() -> list[str]:
    """The match ids of the committed opendata slim fixtures, sorted."""
    return sorted(p.name.split("_", 1)[0] for p in OPENDATA.glob("*_events_slim.csv"))


def load_opendata(match_id: str) -> tuple[pd.DataFrame, dict]:
    """``(events, match_metadata)`` of one committed opendata slim fixture."""
    events = pd.read_csv(OPENDATA / f"{match_id}_events_slim.csv", low_memory=False)
    meta = json.loads((OPENDATA / f"{match_id}_match.json").read_text(encoding="utf-8"))
    return events, meta


def _canon(v: object) -> object:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else "f:" + repr(float(v))
    return str(v)


def _dtype_kind(dtype: object) -> str:
    """The dtype name, with every text dtype (``object`` / pandas 3 ``str`` / ``string``) as one kind ``text``: a
    frame that is not dtype-finalized (``parse_passing_options``) carries ``object`` on pandas 2 and ``str`` on
    pandas 3 for the same values."""
    name = str(dtype)
    return "text" if name in ("object", "str") or name.startswith("string") else name


def canonical_digests(df: pd.DataFrame) -> dict[str, str]:
    """``{column: sha256}`` over the column's dtype kind and its values in a version-neutral form (NA / NaN ->
    null, floats by exact ``repr``). Dict equality ignores the column ORDER, so the golden stores it separately
    (``action_columns`` / ``option_columns``) and the no-op test compares that list too."""
    out: dict[str, str] = {}
    for col in df.columns:
        payload = json.dumps([_dtype_kind(df[col].dtype), [_canon(v) for v in df[col].tolist()]], allow_nan=False)
        out[str(col)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return out


def _clock_ds(time_start: pd.Series) -> np.ndarray:
    """The GI clock in integer deciseconds (``MM:SS.d``)."""
    parts = time_start.astype(str).str.split(":", expand=True)
    return np.round((parts[0].astype("float64") * 60 + parts[1].astype("float64")).to_numpy() * 10).astype("int64")


def _fmt_ds(ds: int) -> str:
    """Integer deciseconds -> the GI ``MM:SS.d`` string."""
    minutes, rem = divmod(int(ds), 600)
    return f"{minutes:02d}:{rem // 10:02d}.{rem % 10}"


def implied_starts(events: pd.DataFrame) -> dict[int, int]:
    """Each period's start frame from the GI clock identity (the mode over its ``player_possession`` events)."""
    pp = events[events["event_type"] == "player_possession"].dropna(subset=["frame_start", "period", "time_start"])
    out: dict[int, int] = {}
    for period in sorted({int(x) for x in pp["period"].tolist()}):
        grp = pp[pp["period"] == period]
        v = grp["frame_start"].astype("int64").to_numpy() - (_clock_ds(grp["time_start"]) - _OFFSET_DS[period])
        vals, counts = np.unique(v, return_counts=True)
        out[period] = int(vals[np.argmax(counts)])
    return out


def _block(events: pd.DataFrame, period: int, *, first: bool) -> tuple[pd.Index, object]:
    """The rows of the first (``first=True``) or last possession of ``period`` in frame order, plus its
    ``passing_option`` / ``off_ball_run`` rows (by ``associated_player_possession_event_id``)."""
    pp = events[(events["event_type"] == "player_possession") & (events["period"] == period)]
    pp = pp.sort_values(["frame_start", "frame_end"], kind="mergesort")
    head = pp.iloc[0] if first else pp.iloc[-1]
    eid = head["event_id"]
    assoc = events.get("associated_player_possession_event_id")
    linked = events["event_type"].isin(["passing_option", "off_ball_run"]) & (
        assoc.astype(str) == str(eid) if assoc is not None else False
    )
    rows = events.index[(events["event_id"] == eid) & (events["event_type"] == "player_possession")].union(
        events.index[linked]
    )
    return rows, eid


def _relabel(events: pd.DataFrame, rows: pd.Index, src: int, dst: int) -> pd.DataFrame:
    """Give ``rows`` the label ``dst`` and a clock on ``dst``'s basis (a label-derived clock, as GI computes)."""
    s = implied_starts(events)
    out = events.copy()
    ds = _clock_ds(out.loc[rows, "time_start"])
    shifted = ds + (_OFFSET_DS[dst] - _OFFSET_DS[src]) - (s[dst] - s[src])
    out.loc[rows, "period"] = dst
    out.loc[rows, "time_start"] = [_fmt_ds(v) for v in shifted]
    return out


def mislabel_p2_to_p1(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """The observed defect: the FIRST period-2 possession block labelled period 1 on the period-1 clock basis.
    Returns ``(mislabelled events, the planted rows)``."""
    rows, _eid = _block(events, 2, first=True)
    return _relabel(events, rows, 2, 1), rows


def mislabel_p1_to_p2(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """The reverse boundary: the LAST period-1 possession block labelled period 2 on the period-2 clock basis."""
    rows, _eid = _block(events, 1, first=False)
    return _relabel(events, rows, 1, 2), rows


def _write_golden() -> None:
    from silly_kicks.providers.skillcorner import parse_passing_options
    from silly_kicks.spadl import skillcorner

    golden: dict = {"generated_at_commit": "1007ef6", "matches": {}}
    for mid in opendata_ids():
        events, meta = load_opendata(mid)
        actions, _report = skillcorner.convert_to_actions(events, meta)
        options = parse_passing_options(events, game_id=mid)
        golden["matches"][mid] = {
            "actions": canonical_digests(actions),
            "action_columns": [str(c) for c in actions.columns],
            "n_actions": len(actions),
            "options": canonical_digests(options),
            "option_columns": [str(c) for c in options.columns],
            "n_options": len(options),
        }
    GOLDEN.write_bytes((json.dumps(golden, indent=1, sort_keys=True) + "\n").encode("utf-8"))


if __name__ == "__main__":
    _write_golden()
