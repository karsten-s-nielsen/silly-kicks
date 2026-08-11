"""Deterministic generator for the Gradient Sports synthetic match fixture.

Produces ``synthetic_match.json`` covering every dispatch row from
``silly_kicks/spadl/gradientsports.py`` with >=2x redundancy, plus exclusion / set-piece
/ result / body-part / tackle-winner / foul / card variations needed to
exercise the full converter contract, the three ADR-018 goal-capture events (see
:func:`_hand_authored`), and an input-convention block that lets the ADR-006 detector
CLASSIFY rather than defer (see the comment at that block).

Run:
    uv run python tests/datasets/gradientsports/_generate_synthetic_match.py

The resulting JSON is committed and is the canonical test artifact. This generator is a
maintainer-time tool and is not invoked BY pytest -- but
``test_gradientsports.py::test_the_generator_reproduces_the_committed_fixture`` imports
``build_match()`` and asserts its serialization equals the committed file. That gate exists
because its absence let the two drift for a full release cycle: the generator emitted 51
events against a committed 54, so regenerating would have silently deleted the own goal and
the cross-goal a live test asserts. **Never hand-edit the JSON -- change the generator and
re-run it.**
"""

from __future__ import annotations

import json
from pathlib import Path

# Two synthetic teams. Player IDs 1-11 home, 12-22 away.
HOME_TEAM_ID = 100
AWAY_TEAM_ID = 200
HOME_TEAM_NAME = "Synthetic FC"
AWAY_TEAM_NAME = "Test United"

OUTPUT_PATH = Path(__file__).parent / "synthetic_match.json"


def _ge_envelope(ge_type, *, period, time_s, team_id, player_id, set_piece="O", home_team=True):
    """gameEvents skeleton."""
    return {
        "gameEventType": ge_type,
        "initialNonEvent": False,
        "startGameClock": int(time_s),
        "startFormattedGameClock": f"{int(time_s) // 60:02d}:{int(time_s) % 60:02d}",
        "period": period,
        "videoMissing": False,
        "teamId": team_id,
        "teamName": HOME_TEAM_NAME if home_team else AWAY_TEAM_NAME,
        "homeTeam": home_team,
        "playerId": player_id,
        "playerName": f"Player {player_id}",
        "touches": 1,
        "touchesInBox": 0,
        "setpieceType": set_piece,
        "earlyDistribution": False,
        "videoUrl": None,
        "endType": None,
        "outType": None,
        "subType": None,
        "playerOffId": None,
        "playerOffName": None,
        "playerOffType": None,
        "playerOnId": None,
        "playerOnName": None,
    }


def _empty_pe():
    """possessionEvents skeleton with all qualifier fields null."""
    return {
        "possessionEventType": None,
        "nonEvent": False,
        "gameClock": 0,
        "formattedGameClock": "00:00",
        "eventVideoUrl": None,
        "ballHeightType": None,
        "bodyType": None,
        "highPointType": None,
        "passerPlayerId": None,
        "passerPlayerName": None,
        "passType": None,
        "passOutcomeType": None,
        "crosserPlayerId": None,
        "crosserPlayerName": None,
        "crossType": None,
        "crossZoneType": None,
        "crossOutcomeType": None,
        "targetPlayerId": None,
        "targetPlayerName": None,
        "targetFacingType": None,
        "receiverPlayerId": None,
        "receiverPlayerName": None,
        "receiverFacingType": None,
        "defenderPlayerId": None,
        "defenderPlayerName": None,
        "blockerPlayerId": None,
        "blockerPlayerName": None,
        "deflectorPlayerId": None,
        "deflectorPlayerName": None,
        "failedBlockerPlayerId": None,
        "failedBlockerPlayerName": None,
        "failedBlocker2PlayerId": None,
        "failedBlocker2PlayerName": None,
        "accuracyType": None,
        "incompletionReasonType": None,
        "secondIncompletionReasonType": None,
        "linesBrokenType": None,
        "shooterPlayerId": None,
        "shooterPlayerName": None,
        "bodyMovementType": None,
        "ballMoving": None,
        "shotType": None,
        "shotNatureType": None,
        "shotInitialHeightType": None,
        "shotOutcomeType": None,
        "keeperPlayerId": None,
        "keeperPlayerName": None,
        "saveHeightType": None,
        "saveReboundType": None,
        "keeperTouchType": None,
        "glClearerPlayerId": None,
        "glClearerPlayerName": None,
        "badParry": None,
        "saveable": None,
        "clearerPlayerId": None,
        "clearerPlayerName": None,
        "clearanceOutcomeType": None,
        "carrierPlayerId": None,
        "carrierPlayerName": None,
        "ballCarrierPlayerId": None,
        "ballCarrierPlayerName": None,
        "ballCarryOutcome": None,
        "carryDefenderPlayerId": None,
        "carryDefenderPlayerName": None,
        "carryIntent": None,
        "carrySuccessful": None,
        "carryType": None,
        "challengerPlayerId": None,
        "challengerPlayerName": None,
        "challengeType": None,
        "challengeOutcomeType": None,
        "challengeWinnerPlayerId": None,
        "challengeWinnerPlayerName": None,
        "challengeKeeperPlayerId": None,
        "challengeKeeperPlayerName": None,
        "tackleAttemptType": None,
        "rebounderPlayerId": None,
        "rebounderPlayerName": None,
        "reboundOutcomeType": None,
        "touchPlayerId": None,
        "touchPlayerName": None,
        "touchType": None,
        "touchOutcomeType": None,
        "missedTouchPlayerId": None,
        "missedTouchPlayerName": None,
        "missedTouchType": None,
        "originateType": None,
        "opportunityType": None,
        "trickType": None,
        "createsSpace": None,
        "pressureType": None,
        "pressurePlayerId": None,
        "pressurePlayerName": None,
        "closingDownPlayerId": None,
        "closingDownPlayerName": None,
        "additionalDuelerPlayerId": None,
        "additionalDuelerPlayerName": None,
        "betterOptionPlayerId": None,
        "betterOptionPlayerName": None,
        "betterOptionTime": None,
        "betterOptionType": None,
        "movementPlayerId": None,
        "movementPlayerName": None,
        "positionPlayerId": None,
        "positionPlayerName": None,
        "homeDuelPlayerId": None,
        "homeDuelPlayerName": None,
        "awayDuelPlayerId": None,
        "awayDuelPlayerName": None,
        "dribblerPlayerId": None,
        "dribblerPlayerName": None,
        "dribbleType": None,
    }


def _empty_foul():
    """One per-event ``fouls`` payload — a single dict (NOT a list of dicts).

    Real Gradient Sports event data carries ``fouls`` as a single dict (not a JSON array)
    per event. Synthetic fixture must match this shape so the converter's
    expected JSON reader produces compatible DataFrame columns.
    """
    return {
        "onFieldCulpritPlayerId": None,
        "onFieldCulpritPlayerName": None,
        "finalCulpritPlayerId": None,
        "finalCulpritPlayerName": None,
        "victimPlayerId": None,
        "victimPlayerName": None,
        "foulType": None,
        "onFieldOffenseType": None,
        "finalOffenseType": None,
        "onFieldFoulOutcomeType": None,
        "finalFoulOutcomeType": None,
        "var": None,
        "varReasonType": None,
        "correctDecision": None,
    }


def make_event(
    event_id,
    *,
    ge_type="OTB",
    pe_type=None,
    period=1,
    time_s=10.0,
    team_id=HOME_TEAM_ID,
    player_id=1,
    set_piece="O",
    home_team=True,
    ball_x=0.0,
    ball_y=0.0,
    **pe_overrides,
):
    """Build one Gradient Sports event row."""
    pe = _empty_pe()
    if pe_type is not None:
        pe["possessionEventType"] = pe_type
    pe.update(pe_overrides)
    return {
        "gameId": 99999,
        "gameEventId": event_id,
        "possessionEventId": event_id,
        "startTime": 200.0 + time_s,
        "endTime": 200.0 + time_s,
        "duration": 0.0,
        "eventTime": 200.0 + time_s,
        "sequence": float(event_id),
        "gameEvents": _ge_envelope(
            ge_type,
            period=period,
            time_s=time_s,
            team_id=team_id,
            player_id=player_id,
            set_piece=set_piece,
            home_team=home_team,
        ),
        "initialTouch": {
            "initialBodyType": "R",
            "initialHeightType": "G",
            "facingType": "G",
            "initialTouchType": "S",
            "initialPressureType": "N",
            "initialPressurePlayerId": None,
            "initialPressurePlayerName": None,
        },
        "possessionEvents": pe,
        "fouls": _empty_foul(),
        "grades": [],
        "stadiumMetadata": [],
        "homePlayers": [],
        "awayPlayers": [],
        "ball": [{"visibility": "VISIBLE", "x": ball_x, "y": ball_y, "z": 0.0}],
    }


def _hand_authored(event, *, raw_time):
    """Reproduce the three field-level artifacts of the three ADR-018 goal-capture events.

    Events 52-54 (the RE+G own goal, the CR+G cross-goal, the ``nonEvent`` disallowed shot)
    were appended DIRECTLY to ``synthetic_match.json`` when goal capture landed; this
    generator was never updated. From that day it emitted 51 events against a committed 54,
    and any regeneration silently deleted all three -- including the two that
    ``test_owngoal_crossgoal_captured_disallowed_excluded`` asserts. Nothing in CI ever runs
    the generator, so the drift was invisible.

    Being hand-authored they deviate from :func:`make_event` in three ways (five fields).
    None is read by the converter: ``_load_synthetic_events`` takes only
    ``ball["x"]``/``["y"]`` and ``startGameClock``.

    * ``ball`` carries no ``visibility`` / ``z`` key.
    * ``startTime`` / ``endTime`` / ``eventTime`` hold the raw clock, not ``200.0 + time_s``.
    * ``startFormattedGameClock`` reads ``"00:12"`` at clocks of 1000 / 1100 / 1200 s.

    They are reproduced VERBATIM rather than normalized so that restoring them is provably a
    no-op on the committed bytes. A generator that "improves" the artifact in the same change
    that restores it cannot show which of the two edits moved the file.
    """
    event["startTime"] = raw_time
    event["endTime"] = raw_time
    event["eventTime"] = raw_time
    event["gameEvents"]["startFormattedGameClock"] = "00:12"
    event["ball"] = [{"x": event["ball"][0]["x"], "y": event["ball"][0]["y"]}]
    return event


def build_match():
    """Compose the synthetic match's events list with full dispatch coverage."""
    events: list[dict] = []
    eid = 1

    # Period 1 kickoff (excluded) — also exercises FIRSTKICKOFF count.
    events.append(make_event(eid, ge_type="FIRSTKICKOFF", pe_type=None, period=1, time_s=0.0, ball_x=0.0, ball_y=0.0))
    eid += 1

    # PASS dispatch: open-play / kickoff / FK / corner / throw-in / goal kick.
    for sp, body, outcome, ball in [
        ("O", "R", "C", -10.0),
        ("O", "L", "F", -5.0),
        ("K", "R", "C", 0.0),
        ("F", "R", "C", -25.0),
        ("C", "R", "C", -45.0),
        ("T", "H", "C", -52.0),
        ("G", "R", "C", -50.0),
    ]:
        events.append(
            make_event(
                eid,
                pe_type="PA",
                period=1,
                time_s=10.0 + eid,
                set_piece=sp,
                ball_x=ball,
                ball_y=0.0,
                bodyType=body,
                passOutcomeType=outcome,
            )
        )
        eid += 1

    # CROSS dispatch: open / FK / corner.
    for sp, outcome, ball in [
        ("O", "C", 30.0),
        ("F", "C", -20.0),
        ("C", "F", -45.0),
    ]:
        events.append(
            make_event(
                eid,
                pe_type="CR",
                period=1,
                time_s=20.0 + eid,
                set_piece=sp,
                ball_x=ball,
                ball_y=15.0,
                bodyType="R",
                crossOutcomeType=outcome,
            )
        )
        eid += 1

    # SHOT dispatch: open / FK / penalty + result mapping.
    for sp, outcome, ball in [
        ("O", "G", 40.0),
        ("O", "S", 35.0),
        ("O", "B", 30.0),
        ("O", "W", 38.0),
        ("F", "S", -20.0),
        ("P", "G", 41.0),
        ("O", "O", -10.0),  # own goal
    ]:
        events.append(
            make_event(
                eid,
                pe_type="SH",
                period=1,
                time_s=30.0 + eid,
                set_piece=sp,
                ball_x=ball,
                ball_y=0.0,
                bodyType="R",
                shotOutcomeType=outcome,
            )
        )
        eid += 1

    # CLEARANCE / DRIBBLE / TOUCH-CONTROL.
    events.append(make_event(eid, pe_type="CL", period=1, time_s=50.0, ball_x=-30.0, ball_y=0.0, bodyType="H"))
    eid += 1
    for outcome in ["R", "L"]:
        events.append(
            make_event(
                eid,
                pe_type="BC",
                period=1,
                time_s=51.0 + eid,
                ball_x=10.0,
                ball_y=0.0,
                bodyType="R",
                ballCarryOutcome=outcome,
            )
        )
        eid += 1
    events.append(make_event(eid, pe_type="TC", period=1, time_s=53.0, ball_x=15.0, ball_y=0.0, bodyType="O"))
    eid += 1

    # CHALLENGE: tackle winner = challenger AND tackle winner = carrier.
    events.append(
        make_event(
            eid,
            pe_type="CH",
            period=1,
            time_s=55.0,
            team_id=HOME_TEAM_ID,
            player_id=1,
            ball_x=20.0,
            ball_y=0.0,
            bodyType="R",
            challengerPlayerId=12,
            challengerPlayerName="Player 12",
            challengeWinnerPlayerId=12,
            challengeWinnerPlayerName="Player 12",
        )
    )
    eid += 1
    events.append(
        make_event(
            eid,
            pe_type="CH",
            period=1,
            time_s=56.0,
            team_id=HOME_TEAM_ID,
            player_id=2,
            ball_x=22.0,
            ball_y=0.0,
            bodyType="R",
            challengerPlayerId=13,
            challengerPlayerName="Player 13",
            challengeWinnerPlayerId=2,
            challengeWinnerPlayerName="Player 2",
        )
    )
    eid += 1

    # REBOUND: keeper save (default) and keeper pick-up (catch-class).
    events.append(
        make_event(
            eid,
            pe_type="RE",
            period=1,
            time_s=60.0,
            team_id=AWAY_TEAM_ID,
            player_id=12,
            home_team=False,
            ball_x=-45.0,
            ball_y=0.0,
            bodyType="O",
            keeperTouchType="P",
        )
    )
    eid += 1
    events.append(
        make_event(
            eid,
            pe_type="RE",
            period=1,
            time_s=61.0,
            team_id=AWAY_TEAM_ID,
            player_id=12,
            home_team=False,
            ball_x=-45.0,
            ball_y=0.0,
            bodyType="O",
            keeperTouchType="C",
        )
    )
    eid += 1

    # FOULS: yellow card + red card.
    e = make_event(
        eid,
        pe_type="PA",
        period=1,
        time_s=65.0,
        ball_x=0.0,
        ball_y=0.0,
        bodyType="R",
        passOutcomeType="C",
    )
    e["fouls"] = {
        **_empty_foul(),
        "foulType": "STANDARD",
        "onFieldCulpritPlayerId": 12,
        "onFieldCulpritPlayerName": "Player 12",
        "victimPlayerId": 1,
        "victimPlayerName": "Player 1",
        "finalFoulOutcomeType": "Y",
    }
    events.append(e)
    eid += 1

    e = make_event(
        eid,
        pe_type="PA",
        period=1,
        time_s=66.0,
        ball_x=10.0,
        ball_y=0.0,
        bodyType="R",
        passOutcomeType="C",
    )
    e["fouls"] = {
        **_empty_foul(),
        "foulType": "STANDARD",
        "finalFoulOutcomeType": "R",
    }
    events.append(e)
    eid += 1

    # DEDICATED FOUL EVENT: gameEventType="FOUL" with possessionEventType="FO"
    # plus populated fouls dict. Real Gradient Sports data uses these for some standalone
    # fouls (vs inline fouls on PA/CR/etc rows above). Converter handles
    # via in-place foul-row conversion.
    # Real GS data has NULL startGameClock on all 28 dedicated FOUL events
    # across 13/64 WC2022 matches — the converter must impute time_seconds.
    e = make_event(
        eid,
        ge_type="FOUL",
        pe_type="FO",
        period=1,
        time_s=67.0,
        ball_x=20.0,
        ball_y=0.0,
        bodyType="R",
    )
    e["gameEvents"]["startGameClock"] = None
    e["fouls"] = {
        **_empty_foul(),
        "foulType": "I",
        "onFieldCulpritPlayerId": 13,
        "onFieldCulpritPlayerName": "Player 13",
        "finalFoulOutcomeType": "Y",
    }
    events.append(e)
    eid += 1

    # EXCLUDED: ball out of play, sub, OTB+IT (initial touch / ball receipt),
    # OFF / ON (player-off / player-on substitution metadata),
    # G (game marker), OTB+empty (initialNonEvent markers).
    events.append(make_event(eid, ge_type="OUT", pe_type=None, period=1, time_s=70.0, ball_x=-52.0, ball_y=20.0))
    eid += 1
    events.append(make_event(eid, ge_type="SUB", pe_type=None, period=1, time_s=72.0))
    eid += 1
    events.append(
        make_event(
            eid,
            ge_type="OTB",
            pe_type="IT",
            period=1,
            time_s=73.0,
            ball_x=10.0,
            ball_y=0.0,
        )
    )
    eid += 1
    events.append(make_event(eid, ge_type="OFF", pe_type=None, period=1, time_s=74.0))
    eid += 1
    events.append(make_event(eid, ge_type="ON", pe_type=None, period=1, time_s=74.5))
    eid += 1
    events.append(make_event(eid, ge_type="G", pe_type=None, period=1, time_s=75.0))
    eid += 1
    # OTB with no possession-event (initialNonEvent marker)
    events.append(make_event(eid, ge_type="OTB", pe_type=None, period=1, time_s=75.5))
    eid += 1

    # PERIOD 2: kickoff + small dispatch sample to exercise the cross-period flip.
    events.append(make_event(eid, ge_type="SECONDKICKOFF", pe_type=None, period=2, time_s=0.0, ball_x=0.0, ball_y=0.0))
    eid += 1
    for sp, body, outcome, ball in [
        ("O", "R", "C", 10.0),
        ("O", "L", "F", -5.0),
    ]:
        events.append(
            make_event(
                eid,
                pe_type="PA",
                period=2,
                time_s=10.0 + eid,
                set_piece=sp,
                ball_x=ball,
                ball_y=0.0,
                bodyType=body,
                passOutcomeType=outcome,
            )
        )
        eid += 1
    events.append(
        make_event(
            eid,
            pe_type="SH",
            period=2,
            time_s=30.0,
            set_piece="O",
            ball_x=42.0,
            ball_y=0.0,
            bodyType="R",
            shotOutcomeType="G",
        )
    )
    eid += 1

    # PERIOD 3 / 4 (ET): kickoff markers ensure both excluded vocab paths are
    # exercised (THIRDKICKOFF / FOURTHKICKOFF). One PA per period is enough
    # to exercise the per-period direction flip in ET, given matching
    # home_team_start_left_extratime is supplied at convert time.
    events.append(make_event(eid, ge_type="THIRDKICKOFF", pe_type=None, period=3, time_s=0.0))
    eid += 1
    events.append(
        make_event(
            eid,
            pe_type="PA",
            period=3,
            time_s=5.0,
            set_piece="O",
            ball_x=0.0,
            ball_y=0.0,
            bodyType="R",
            passOutcomeType="C",
        )
    )
    eid += 1
    events.append(make_event(eid, ge_type="FOURTHKICKOFF", pe_type=None, period=4, time_s=0.0))
    eid += 1
    events.append(
        make_event(
            eid,
            pe_type="PA",
            period=4,
            time_s=5.0,
            set_piece="O",
            ball_x=0.0,
            ball_y=0.0,
            bodyType="R",
            passOutcomeType="C",
        )
    )
    eid += 1

    # OUT-OF-BOUNDS COORDINATES: real WC 2022 has ~1.2% actions with
    # ball positions slightly outside the pitch lines (throw-ins, GK
    # overruns, tracking noise). Max overshoot: ~5m x, ~8m y.
    # Centered coords: ball_x=57.5 → SPADL 110.0 (OOB high x by 5m);
    #                  ball_y=44.0 → SPADL 78.0 (OOB high y by 10m);
    #                  ball_x=-58.0 → SPADL -5.5 (OOB low x by 5.5m);
    #                  ball_y=-42.0 → SPADL -8.0 (OOB low y by 8m).
    # Pass with OOB high x (throw-in area near attacking goal line)
    events.append(
        make_event(
            eid,
            pe_type="PA",
            period=2,
            time_s=40.0,
            set_piece="T",
            ball_x=57.5,
            ball_y=0.0,
            bodyType="R",
            passOutcomeType="C",
        )
    )
    eid += 1
    # Cross with OOB high y (sideline overshoot)
    events.append(
        make_event(
            eid,
            pe_type="CR",
            period=2,
            time_s=42.0,
            ball_x=30.0,
            ball_y=44.0,
            bodyType="R",
            crossOutcomeType="C",
        )
    )
    eid += 1
    # Clearance with OOB low x and low y
    events.append(
        make_event(
            eid,
            pe_type="CL",
            period=2,
            time_s=44.0,
            ball_x=-58.0,
            ball_y=-42.0,
            bodyType="H",
        )
    )
    eid += 1
    # Pass (away team) with OOB high y — exercises OOB after LTR flip
    events.append(
        make_event(
            eid,
            pe_type="PA",
            period=2,
            time_s=46.0,
            team_id=AWAY_TEAM_ID,
            player_id=12,
            home_team=False,
            ball_x=0.0,
            ball_y=44.0,
            bodyType="R",
            passOutcomeType="C",
        )
    )
    eid += 1

    # NULL-ACTOR EVENTS: real WC 2022 has ~17 events/match with
    # gameEvents.teamId=NULL and gameEvents.playerId=NULL (10 OTB+CH + 7 FOUL+FO).
    # Converter must produce team_id=0, player_id=0 (Int64 fillna sentinel).
    e = make_event(eid, ge_type="OTB", pe_type="CH", period=2, time_s=2680.0, ball_x=5.0, ball_y=2.0)
    e["gameEvents"]["teamId"] = None
    e["gameEvents"]["teamName"] = None
    e["gameEvents"]["homeTeam"] = None
    e["gameEvents"]["playerId"] = None
    e["gameEvents"]["playerName"] = None
    events.append(e)
    eid += 1

    e = make_event(eid, ge_type="FOUL", pe_type="FO", period=2, time_s=2690.0, ball_x=-10.0, ball_y=5.0)
    e["gameEvents"]["teamId"] = None
    e["gameEvents"]["teamName"] = None
    e["gameEvents"]["homeTeam"] = None
    e["gameEvents"]["playerId"] = None
    e["gameEvents"]["playerName"] = None
    e["fouls"] = {**_empty_foul(), "foulType": "M"}
    events.append(e)
    eid += 1

    # GOAL CAPTURE (ADR-018): the own goal, the cross-goal, and the DISALLOWED shot.
    # These carry ids 52-54 yet sort BEFORE the id-51 END marker, because they were appended
    # directly to the committed JSON rather than emitted here. See _hand_authored(). They do
    # not consume `eid`.
    events.append(
        _hand_authored(
            make_event(
                52,
                pe_type="RE",
                period=1,
                time_s=1000.0,
                team_id=HOME_TEAM_ID,
                player_id=7,
                ball_x=-45.0,
                ball_y=0.0,
                bodyType="R",
                shotOutcomeType="G",
                rebounderPlayerId=7,
                rebounderPlayerName="OG Scorer",
            ),
            raw_time=1000.0,
        )
    )
    events.append(
        _hand_authored(
            make_event(
                53,
                pe_type="CR",
                period=1,
                time_s=1100.0,
                team_id=HOME_TEAM_ID,
                player_id=11,
                ball_x=45.0,
                ball_y=30.0,
                bodyType="R",
                crosserPlayerId=11,
                crosserPlayerName="Crosser",
                crossOutcomeType="I",
                shotOutcomeType="G",
            ),
            raw_time=1100.0,
        )
    )
    events.append(
        _hand_authored(
            make_event(
                54,
                pe_type="SH",
                period=1,
                time_s=1200.0,
                team_id=HOME_TEAM_ID,
                player_id=9,
                ball_x=40.0,
                ball_y=0.0,
                bodyType="R",
                nonEvent=True,
                shooterPlayerId=9,
                shooterPlayerName="Disallowed Scorer",
                shotOutcomeType="G",
            ),
            raw_time=1200.0,
        )
    )

    # INPUT-CONVENTION BLOCK: give `detect_input_convention` a case it can CLASSIFY.
    #
    # Before this block the fixture had shots from ONE team only (team 100: 8 in P1, 1 in P2),
    # so the detector returned `convention=None, confidence="low"` on the
    # fewer-than-2-reliable-groups clause (`orientation.py:410`) and the converter's
    # `validate_input_convention` guard deferred silently. CI therefore never exercised the
    # guard on this provider at all -- neither its agreement path nor its raise path.
    #
    # Shape, measured over all 64 real GS WC2022 matches (`docs/research/gs_input_convention/`):
    # both teams shoot at OPPOSITE ends within a period and SWAP ends between periods. That is
    # the `per_period_absolute` signature the converter declares at `gradientsports.py:756`
    # (match 10502: team 51 `P1 13.3 LOW -> P2 94.0 HIGH`; team 366 `P1 95.8 HIGH -> P2 9.8 LOW`).
    #
    # Three choices the measurement underdetermines, stated rather than made silently:
    #
    # 1. FIVE shots per group, not ten. Five is `min_shots_per_group_medium`, so the detector
    #    classifies at `confidence="medium"`. That is the representative tier: only 6 of 64 real
    #    matches reach two `high`-reliable (>=10) groups, while 50 of 64 reach two at `medium`.
    #    A fixture pinned at the `high` tier would test a shape the corpus almost never has.
    # 2. APPENDED, not interleaved. `make_event` callers above use the `time_s = X + eid` idiom,
    #    so inserting an event mid-stream silently shifts every later event's game clock. These
    #    carry explicit ids (55-74) and explicit clocks and consume no `eid`, so no existing
    #    event's id, clock or converted action changes.
    # 3. Outcome "S" (saved), never "G". These exist to carry a coordinate distribution, so they
    #    must not add goals -- the goal assertions belong to the three ADR-018 events above.
    #
    # The x spread is deliberate rather than one repeated value: a constant column is the
    # degeneracy that has twice made a gate score noise here (see CLAUDE.md on fixture validity).
    conv_eid = 55
    for team_id, player_id, home, period, base_time, side in (
        (HOME_TEAM_ID, 9, True, 1, 1300.0, +1.0),
        (AWAY_TEAM_ID, 20, False, 1, 1400.0, -1.0),
        (HOME_TEAM_ID, 9, True, 2, 1500.0, -1.0),
        (AWAY_TEAM_ID, 20, False, 2, 1600.0, +1.0),
    ):
        for k, spread in enumerate((-3.0, -1.0, 0.0, 1.0, 3.0)):
            events.append(
                make_event(
                    conv_eid,
                    pe_type="SH",
                    period=period,
                    time_s=base_time + 10.0 * k,
                    team_id=team_id,
                    player_id=player_id,
                    home_team=home,
                    set_piece="O",
                    ball_x=side * 45.0 + spread,
                    ball_y=0.0,
                    bodyType="R",
                    shotOutcomeType="S",
                )
            )
            conv_eid += 1

    # END markers.
    events.append(make_event(eid, ge_type="END", pe_type=None, period=2, time_s=2700.0, ball_x=0.0, ball_y=0.0))
    eid += 1

    return events


def main():
    events = build_match()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
        # The committed artifact ends with a newline; `json.dump` does not write one. Without
        # this the generator's own output differs from its committed file by one byte.
        f.write("\n")
    print(f"Wrote {len(events)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
