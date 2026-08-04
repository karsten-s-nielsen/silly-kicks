#!/usr/bin/env python
"""Build the frozen directional feature-vector fixture for the xCross CI gates.

Companion to ``scripts/make_xshot_directional_fixture.py``. xCross has no single rankable proxy --
"a cross is imminent" is a MULTI-feature condition (wide AND advanced AND fast toward goal AND a
teammate arriving) and the model's dominant feature is ``ball_speed``, not a positional distance --
so the fixture is built from SYNTHETIC states with realistic, varying features.

What the 4.74.0 rebuild fixed, and why each part is load-bearing
---------------------------------------------------------------
The previous fixture was measured **inert in 9 of its 16 features** and the gate reading it reported
``AUC = 1.0000`` while the entire GK block was constant -- so a model ignoring keeper position
entirely scored identically to the real one (verified: overwriting the GK block with ``0``, ``99`` or
``NaN`` each left AUC at 1.0000).

* **The keeper is swept, not pinned.** It was hard-coded at ``(2.0, 34.0)`` on every frame, so
  ``gk_r``/``gk_theta``/``gk_lateral_offset``/``gk_carrier_side`` were constant BY CONSTRUCTION and
  no gate could ever detect a GK-blind model.
* **``score_differential`` is varied.** It was ``np.nan`` on every row -- an all-NaN column, and the
  model's confounder #1. Note a bare ``(max - min) <= tol`` range check does NOT catch this, because
  ``nan <= tol`` is False; the precondition gate uses an explicit ``notna().any()`` clause.
* **``ten_minute_warning`` is varied** via ``time_seconds``: it requires ``>= 35*60`` in period 1/2
  (``_xcross_attempt.py:258``) and the generator hard-set ``time_seconds=0.0``.
* **``box_off_def_ratio`` is varied** via in-box ring occupancy; the fixed rings gave 3/4 on every row.
* **Negatives are redesigned.** They were central and deep, so restricting to the model's trained
  ``wide_area_only`` domain removed EVERY negative (measured: 8 in-domain rows, all label 1) and left
  a single-class fixture that ``roc_auc_score`` cannot score. Negatives are now **wide + advanced but
  not cross-imminent**: slow ball, no arriving teammate.
* **Raw geometry is persisted** (``ball_x``, ``ball_y``, ``goal_x``, ``gk_x``, ``gk_y``,
  ``carrier_y``). Without them the precondition's in-domain clause is *unfalsifiable* --
  ``_in_wide_area`` takes four arguments and the frozen features carry none of them -- and the GK
  sweep would have to move six columns rather than a keeper.

Single-ended by design
----------------------
The fixture is built at ``goal_x=0.0`` only. Chirality coverage does NOT belong here: a committed
table provably cannot carry it (a reflection pair maps onto the SAME goal-relative configuration, so
the two ends hold identical values and a fabricated half is numerically indistinguishable from a real
extraction -- bit-identical on integer coordinates). ``tests/tracking/test_pr5_chirality_gates.py``
covers it properly instead: live extraction, both extractors, both axes, permanent non-vacuity plant.

Regenerate the committed parquet::

    python scripts/make_xcross_directional_fixture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCrossAttemptModel,
    extract_xcross_features,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "datasets" / "tracking" / "xcross_directional" / "frozen_rows.parquet"

_DEF_TEAM_ID = 1
_POSS_TEAM_ID = 2
_CARRIER_ID = 21
_GOAL_X = 0.0

#: Raw geometry persisted alongside the features. `goal_x` is included even though this fixture is
#: single-ended: it makes the precondition self-describing rather than resting on a constant the test
#: would have to hard-code and keep in sync.
_PROVENANCE_COLS = ("ball_x", "ball_y", "goal_x", "gk_x", "gk_y", "carrier_y")


def _build_frame(
    fid: int,
    *,
    ball_x: float,
    ball_y: float,
    ball_speed: float,
    gk_x: float,
    gk_y: float,
    time_seconds: float,
    n_def_box: int,
    n_att_box: int,
    mate_near: bool,
) -> pd.DataFrame:
    """One synthetic frame. Every varying knob is a parameter -- nothing is pinned in the body."""

    def _r(pid, tid, x, y, *, is_ball=False, is_gk=False, vx=0.0, vy=0.0):
        return dict(
            player_id=pid, team_id=tid, is_ball=is_ball, is_goalkeeper=is_gk,
            x=x, y=y, frame_id=fid, time_seconds=time_seconds, vx=vx, vy=vy,
        )  # fmt: skip

    rows = [
        # `ball_speed = hypot(vx, vy)` is a MAGNITUDE, so vx carries it entirely.
        _r(-1, -1, ball_x, ball_y, is_ball=True, vx=-ball_speed, vy=0.0),
        _r(10, _DEF_TEAM_ID, gk_x, gk_y, is_gk=True),  # defending GK -- SWEPT
        _r(20, _POSS_TEAM_ID, 103.0, 34.0, is_gk=True),  # attacking GK
        _r(_CARRIER_ID, _POSS_TEAM_ID, ball_x + 0.3, ball_y),  # carrier
    ]
    # Defenders: `n_def_box` inside the penalty area (x <= 16.5, |y-34| <= 20.16), rest outside.
    for k in range(4):
        inside = k < n_def_box
        rows.append(_r(11 + k, _DEF_TEAM_ID, 5.0 + 2 * k if inside else 30.0 + 3 * k, 26.0 + 3 * k))
    # Attackers: `n_att_box` inside the box, rest outside.
    for k in range(4):
        inside = k < n_att_box
        rows.append(_r(31 + k, _POSS_TEAM_ID, 9.0 + 2 * k if inside else 34.0 + 3 * k, 30.0 + 2 * k))
    mate_dx, mate_dy = (3.0, 2.0) if mate_near else (28.0, -20.0)
    rows.append(_r(30, _POSS_TEAM_ID, ball_x + mate_dx, ball_y + mate_dy))

    f = pd.DataFrame(rows)
    f["game_id"] = "g"
    f["period_id"] = 1
    f["z"] = 0.0
    f["frame_rate"] = 10.0
    f["ball_state"] = "alive"
    f["source_provider"] = "synthetic"
    return f


#: Keeper positions swept across scenes. All are plausible for a defended goal at x=0 -- on the line,
#: off the line, and displaced toward each post -- so the response gate probes realizable states.
_GK_SWEEP = (
    (1.0, 34.0), (2.0, 30.0), (2.0, 38.0), (4.0, 34.0),
    (5.0, 28.0), (5.0, 40.0), (8.0, 32.0), (8.0, 36.0),
    (11.0, 34.0), (3.0, 26.0), (3.0, 42.0), (6.0, 31.0),
)  # fmt: skip

#: Clock values: below and above the 35-minute `ten_minute_warning` threshold in period 1.
_CLOCK = (300.0, 900.0, 1500.0, 2160.0, 2400.0, 2640.0)

#: (n_def_box, n_att_box) -- drives `box_off_def_ratio` across distinct values.
_BOX = ((4, 3), (3, 3), (4, 2), (2, 3), (3, 4), (4, 4))

#: score_differential values (the model's confounder #1).
_SCORE_DIFF = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _specs() -> list[dict]:
    """48 scenes: 24 cross-imminent, 24 not. ALL wide and advanced, i.e. all in the trained domain.

    `_in_wide_area` requires `|ball_x - goal_x| <= 35` AND (`ball_y < 14` or `ball_y > 54`), so the
    negatives cannot be central-and-deep the way they used to be -- that is what emptied the negative
    class when the domain restriction landed.
    """
    out: list[dict] = []
    for i in range(24):
        flank_low = i % 2 == 0
        base_y = 6.0 + (i % 6) * 1.2 if flank_low else 62.0 - (i % 6) * 1.2
        out.append(
            dict(
                label=1,
                ball_x=4.0 + (i % 8) * 1.7,
                ball_y=base_y,
                ball_speed=7.0 + (i % 7) * 1.1,
                mate_near=True,
                gk=_GK_SWEEP[i % len(_GK_SWEEP)],
                time_seconds=_CLOCK[i % len(_CLOCK)],
                box=_BOX[i % len(_BOX)],
                score_differential=_SCORE_DIFF[i % len(_SCORE_DIFF)],
            )
        )
    for i in range(24):
        flank_low = i % 2 == 1
        base_y = 5.0 + (i % 6) * 1.3 if flank_low else 63.0 - (i % 6) * 1.3
        out.append(
            dict(
                label=0,
                ball_x=18.0 + (i % 8) * 1.9,  # still advanced (<= 35), but deeper
                ball_y=base_y,
                ball_speed=0.6 + (i % 7) * 0.25,  # slow: no cross imminent
                mate_near=False,  # nobody arriving
                gk=_GK_SWEEP[(i + 5) % len(_GK_SWEEP)],
                time_seconds=_CLOCK[(i + 3) % len(_CLOCK)],
                box=_BOX[(i + 2) % len(_BOX)],
                score_differential=_SCORE_DIFF[(i + 1) % len(_SCORE_DIFF)],
            )
        )
    return out


def build(out_path: Path = OUT) -> pd.DataFrame:
    """Generate and write the fixture. Importable so a test can regenerate in-process."""
    rows = []
    for fid, spec in enumerate(_specs()):
        gk_x, gk_y = spec["gk"]
        n_def_box, n_att_box = spec["box"]
        frame = _build_frame(
            fid,
            ball_x=spec["ball_x"],
            ball_y=spec["ball_y"],
            ball_speed=spec["ball_speed"],
            gk_x=gk_x,
            gk_y=gk_y,
            time_seconds=spec["time_seconds"],
            n_def_box=n_def_box,
            n_att_box=n_att_box,
            mate_near=spec["mate_near"],
        )
        feat = extract_xcross_features(
            frame,
            gk_team_id=_DEF_TEAM_ID,
            goal_x=_GOAL_X,
            carrier_player_id=_CARRIER_ID,
            score_differential=spec["score_differential"],
        )
        rec = feat.iloc[0].to_dict()
        rec["label"] = spec["label"]
        rec["ball_x"] = spec["ball_x"]
        rec["ball_y"] = spec["ball_y"]
        rec["goal_x"] = _GOAL_X
        rec["gk_x"] = gk_x
        rec["gk_y"] = gk_y
        rec["carrier_y"] = spec["ball_y"]  # carrier is placed at (ball_x + 0.3, ball_y)
        rows.append(rec)

    frozen = pd.DataFrame(rows)[[*XCROSS_FEATURE_NAMES_FAITHFUL, "label", *_PROVENANCE_COLS]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(out_path)
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="destination parquet (default: committed path)")
    a = ap.parse_args()
    frozen = build(a.out)

    from silly_kicks.tracking._xcross_attempt import _ADVANCE_M, _in_wide_area

    inert = [
        c
        for c in XCROSS_FEATURE_NAMES_FAITHFUL
        if not frozen[c].notna().any() or float(frozen[c].max() - frozen[c].min()) <= 1e-9
    ]
    bx = frozen["ball_x"].to_numpy(dtype=float)
    by = frozen["ball_y"].to_numpy(dtype=float)
    gx = frozen["goal_x"].to_numpy(dtype=float)
    dom = [_in_wide_area(bx[i], by[i], gx[i], _ADVANCE_M) for i in range(len(frozen))]
    distinct = len(frozen[list(XCROSS_FEATURE_NAMES_FAITHFUL)].round(9).drop_duplicates())

    model = XCrossAttemptModel.from_variant("default")
    p = model.predict_proba(frozen[XCROSS_FEATURE_NAMES_FAITHFUL])

    print(f"Wrote {len(frozen)} rows to {a.out}")
    print(f"  labels          : {frozen['label'].value_counts().to_dict()}")
    print(f"  distinct vectors: {distinct}")
    print(f"  in-domain       : {sum(dom)}/{len(frozen)}")
    print(f"  inert features  : {inert or 'none'}")
    print(f"  base rate mean p: {p.mean():.6f}   range {p.min():.6f}-{p.max():.6f}")


if __name__ == "__main__":
    main()
