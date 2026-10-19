"""Toy staged-layout corpora for the stage-contract parity tests and preflight check 5 (ADR-128).

Every builder writes the SAME layout an HF job reads, so a migrated pipeline's toy ``--run-all`` exercises
the real data path:

* flat (``kind = "flat_tc3"``): ``shards/<generation>/<provider>__<game>.parquet`` (frames),
  ``_actions/<provider>__<game>.parquet`` (SPADL actions) and ``home_teams.json`` (``{game_id: home}``);
* per-game (``kind = "pergame"``): ``<provider>-<game>/frames.parquet`` (+ ``actions.parquet``,
  ``shots.parquet``).

Each builder is a pure function of its arguments: two builds with the same arguments are byte-identical
(asserted in ``tests/scripts/test_toy_corpora.py``). Toy dataset names are never in
``datasets.lock.json``, so a toy corpus's identity is its file-listing digest (``_stages._dataset_identity``).

``TOY_PARAMS`` maps each migrated module to the CLI arguments its toy ``--run-all`` needs, and
``TOY_BUILDERS`` to the builder that returns its ``--data-root``; the migration tasks add one entry each
and the compute gate asserts the mapping covers every compliant module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

#: The generation directory name of every toy flat corpus (a staged corpus has exactly one).
TOY_GENERATION = "0123456789abcdef"

#: The id dtypes each provider's STAGED data really carries, measured from the parquet schemas of the staged
#: datasets (tc3, xshot17, xshot115, xcross_probe, sb360; 2026-10-20). ``"int"``: ``game_id`` int64; frames
#: ``player_id`` Int64 (``<NA>`` on the ball row) and ``team_id`` float64 (NaN); actions ids int64.
#: ``"str"``: ``game_id`` / ``player_id`` str (None on the ball row); frames ``team_id`` a category of str;
#: actions ids str. So a combined multi-provider output MIXES int and str ids -- the defect class Cycle 1
#: found only on real HF because every toy carried ONE id type (ADR-128). Every toy unit is written through
#: `staged_frames` / `staged_actions` so the toys reproduce it.
PROVIDER_ID_KIND: Mapping[str, str] = {
    "gradientsports": "int",
    "statsbomb": "int",
    "idsse": "str",
    "sportec": "str",
    "skillcorner": "str",
    # test-only (never staged): the integration tests' "synthetic" provider keeps the original int toy ids
    "synthetic": "int",
}
_ID_COLUMNS = ("game_id", "player_id", "team_id")


def _as_int_id(v):
    """An int id from a toy id: ints stay; ``"p10"`` / ``"a10"`` -> 1010 / 2010 (a stable, disjoint map);
    the ball's ``"ball"`` / -1 / None / NA -> None."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) or v in ("ball", -1):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v)
    s = str(v)
    if s[:1] in ("p", "a") and s[1:].isdigit():
        return (1000 if s[0] == "p" else 2000) + int(s[1:])
    return int(s)


def _as_str_id(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) or v in ("ball", -1):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def staged_frames(frames: pd.DataFrame, kind: str) -> pd.DataFrame:
    """A toy frame table with the id dtypes a provider of ``kind`` really stages (`PROVIDER_ID_KIND`)."""
    out = frames.copy()
    if kind == "int":
        out["game_id"] = out["game_id"].map(_as_int_id).astype("int64")
        out["player_id"] = pd.array(out["player_id"].map(_as_int_id).tolist(), dtype="Int64")
        out["team_id"] = out["team_id"].map(_as_int_id).astype("float64")
    elif kind == "str":
        out["game_id"] = out["game_id"].map(_as_str_id).astype(object)
        out["player_id"] = out["player_id"].map(_as_str_id).astype(object)
        out["team_id"] = out["team_id"].map(_as_str_id).astype("category")
    else:
        raise ValueError(f"unknown id kind {kind!r}")
    return out


def staged_actions(actions: pd.DataFrame, kind: str) -> pd.DataFrame:
    """A toy actions table with the id dtypes a provider of ``kind`` really stages (`PROVIDER_ID_KIND`)."""
    out = actions.copy()
    for c in (c for c in _ID_COLUMNS if c in out.columns):
        if kind == "int":
            out[c] = out[c].map(_as_int_id).astype("int64")
        elif kind == "str":
            out[c] = out[c].map(_as_str_id).astype(object)
        else:
            raise ValueError(f"unknown id kind {kind!r}")
    return out


def staged_home(team, kind: str):
    """A ``home_teams.json`` value in the provider's id type (the map's values are team ids)."""
    return _as_int_id(team) if kind == "int" else _as_str_id(team)


def ghost_gk_frames(
    *,
    home_team_id: int = 1,
    away_team_id: int = 2,
    period_id: int = 1,
    frame_id: int = 1,
    game_id: str = "100",
    timestamp: float = 1.0,
    source_provider: str = "gradientsports",
) -> pd.DataFrame:
    """Synthetic frames with ball, GK, defenders, attackers.

    ORIENTED (ADR-028): every non-ball row carries ``team_attacking_direction``, so
    ``acting_team_attacks_rtl`` can resolve a real per-action flip. The scene is
    home-attacks-right (frame-LTR): the home GK sits at x=5 defending the x=0 goal,
    so the home team attacks x=105 -> "ltr"; the away GK sits at x=100 defending the
    x=105 goal, so the away team attacks x=0 -> "rtl". Ball rows carry None, which is
    what ``convert_to_frames`` emits and what the helper filters out anyway.

    Before this column existed the fixture always took the all-False no-flip path, so
    every away-team action it fed to ``add_ghost_gk`` / ``ghost_gk_xfns`` silently
    mixed coordinate conventions and was never exercised against ADR-028.

    Moved here from ``tests/tracking/test_ghost_gk.py::_make_ghost_gk_frames`` (ADR-128, which keeps that
    name as an alias) so the toy corpora and the unit tests share ONE synthesis.
    """
    rows = []
    base = {
        "game_id": game_id,
        "period_id": period_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "ball_state": "alive",
        "time_seconds": timestamp,
        # A REAL, classified provider (fully-observed): the ghost-GK trainer's detected-only
        # filter (spec 4.3, PR-S115) calls keeper_detection_mask, which fail-closes on an
        # unregistered provider. "gradientsports" is a no-op for the filter (all keepers observed),
        # so these synthetic frames exercise the trainer without a spurious unknown-provider raise.
        "source_provider": source_provider,
    }
    # Ball
    rows.append(
        {
            **base,
            "player_id": "ball",
            "team_id": None,
            "x": 50.0,
            "y": 34.0,
            "vx": 2.0,
            "vy": 0.0,
            "speed": 2.0,
            "is_ball": True,
            "is_goalkeeper": False,
            "team_attacking_direction": None,
        }
    )
    # Home GK (defending goal at x=0 -> home attacks x=105 -> "ltr")
    rows.append(
        {
            **base,
            "player_id": "p1",
            "team_id": home_team_id,
            "x": 5.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": False,
            "is_goalkeeper": True,
            "team_attacking_direction": "ltr",
        }
    )
    # Home defenders (4)
    for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
        rows.append(
            {
                **base,
                "player_id": f"p{10 + i}",
                "team_id": home_team_id,
                "x": float(px),
                "y": float(py),
                "vx": 0.5,
                "vy": 0.0,
                "speed": 0.5,
                "is_ball": False,
                "is_goalkeeper": False,
                "team_attacking_direction": "ltr",
            }
        )
    # Away attackers (4)
    for i, (px, py) in enumerate([(40, 30), (45, 34), (38, 40), (50, 34)]):
        rows.append(
            {
                **base,
                "player_id": f"a{10 + i}",
                "team_id": away_team_id,
                "x": float(px),
                "y": float(py),
                "vx": -1.0,
                "vy": 0.0,
                "speed": 1.0,
                "is_ball": False,
                "is_goalkeeper": False,
                "team_attacking_direction": "rtl",
            }
        )
    # Away GK (defending goal at x=105 -> away attacks x=0 -> "rtl")
    rows.append(
        {
            **base,
            "player_id": "a1",
            "team_id": away_team_id,
            "x": 100.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": False,
            "is_goalkeeper": True,
            "team_attacking_direction": "rtl",
        }
    )
    return pd.DataFrame(rows)


def _game_frames(
    game: str, provider: str, *, n_frames: int, seed: int, dt: float = 0.04, keeper_visibility: bool = False
) -> pd.DataFrame:
    """One toy game: ``n_frames`` ghost scenes ``dt`` seconds apart with a seeded, per-game jitter (so
    folds differ). ``keeper_visibility`` adds a detection-aware ``visibility`` flag: every row is detected
    except the two keepers on every third frame (so a detected-only filter keeps a strict subset)."""
    # A STABLE per-game stream (never `hash()`, which is salted per process).
    rng = np.random.default_rng([seed, int(hashlib.sha256(game.encode("utf-8")).hexdigest()[:8], 16)])
    frames = []
    for f in range(1, n_frames + 1):
        scene = ghost_gk_frames(game_id=game, frame_id=f, timestamp=float(f) * dt, source_provider=provider)
        moving = ~scene["is_ball"]
        scene.loc[moving, "x"] = (scene.loc[moving, "x"] + rng.normal(0.0, 0.5, int(moving.sum()))).round(6)
        scene.loc[moving, "y"] = (scene.loc[moving, "y"] + rng.normal(0.0, 0.5, int(moving.sum()))).round(6)
        if keeper_visibility:
            hidden = scene["is_goalkeeper"].to_numpy(dtype=bool) & (f % 3 == 0)
            scene["visibility"] = pd.array(~hidden, dtype="boolean")
        frames.append(scene)
    return pd.concat(frames, ignore_index=True)


def _game_actions(game: str, *, n_frames: int, home: int, away: int) -> pd.DataFrame:
    """Minimal SPADL actions for one toy game: alternating home/away passes and one shot per side."""
    from silly_kicks.spadl import SPADL_COLUMNS

    rows = []
    for i in range(max(2, n_frames // 10)):
        team = home if i % 2 == 0 else away
        rows.append(
            {
                "game_id": game,
                "original_event_id": f"{game}-{i}",
                "action_id": i,
                "period_id": 1,
                "time_seconds": float(i) * 0.4,
                "team_id": team,
                "player_id": 10 + i % 4,
                "start_x": 40.0 + i,
                "start_y": 34.0,
                "end_x": 50.0 + i,
                "end_y": 34.0,
                "type_id": 11 if i in (2, 3) else 0,  # 11 = shot, 0 = pass
                "result_id": 1,
                "bodypart_id": 0,
                "shot_blocked": pd.NA,
                "cross_blocked": pd.NA,
            }
        )
    # The toy frames key games by string id; the actions keep the SAME key (game_id is not cast).
    return pd.DataFrame(rows).astype({c: d for c, d in SPADL_COLUMNS.items() if c != "game_id"})


def build_flat_corpus(
    root: str | pathlib.Path,
    *,
    n_games: int,
    providers: Sequence[str],
    seed: int = 0,
    n_frames: int = 40,
    dt: float = 0.04,
    heavy_frames: int | None = None,
    detection_aware: frozenset[str] = frozenset(),
) -> pathlib.Path:
    """Write ONE toy flat dataset IN ``root`` and return ``root``.

    Games are numbered ``100, 101, ...`` and assigned to ``providers`` round-robin, so a two-provider corpus
    has both providers in every CV split that ``StratifiedGroupKFold`` can draw. ``heavy_frames`` (if set)
    makes the FIRST game heavy, so at W >= 2 it finishes last in its extract job (the parity non-vacuity);
    a provider in ``detection_aware`` carries the keeper ``visibility`` flag (``_game_frames``).
    """
    root = pathlib.Path(root)
    shards = root / "shards" / TOY_GENERATION
    actions_dir = root / "_actions"
    shards.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    home_teams: dict[str, object] = {}
    for g in range(n_games):
        game = str(100 + g)
        provider = providers[g % len(providers)]
        kind = PROVIDER_ID_KIND[provider]
        stem = f"{provider}__{game}"
        n = heavy_frames if (heavy_frames is not None and g == 0) else n_frames
        frames = _game_frames(
            game, provider, n_frames=n, seed=seed, dt=dt, keeper_visibility=provider in detection_aware
        )
        staged_frames(frames, kind).to_parquet(shards / f"{stem}.parquet", index=False)
        actions = staged_actions(_game_actions(game, n_frames=n, home=1, away=2), kind)
        actions.to_parquet(actions_dir / f"{stem}.parquet", index=False)
        home_teams[game] = staged_home(1, kind)
    (root / "home_teams.json").write_text(json.dumps(home_teams, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def build_pergame_corpus(
    root: str | pathlib.Path,
    *,
    n_games: int,
    provider: str,
    seed: int = 0,
    n_frames: int = 40,
) -> pathlib.Path:
    """Write ONE toy per-game dataset IN ``root`` (``<provider>-<game>/frames.parquet`` + actions + shots)."""
    from scripts.hf_jobs.datasets import pergame_name

    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        game = str(100 + g)
        gdir = root / pergame_name(provider, game)
        gdir.mkdir(parents=True, exist_ok=True)
        kind = PROVIDER_ID_KIND[provider]
        frames = staged_frames(_game_frames(game, provider, n_frames=n_frames, seed=seed), kind)
        frames.to_parquet(gdir / "frames.parquet", index=False)
        actions = staged_actions(_game_actions(game, n_frames=n_frames, home=1, away=2), kind)
        actions.to_parquet(gdir / "actions.parquet", index=False)
        actions[actions["type_id"] == 11].to_parquet(gdir / "shots.parquet", index=False)
    return root


def build_ghost_corpus(
    root: str | pathlib.Path, *, n_games: int = 6, n_frames: int = 40, heavy_frames: int = 400
) -> pathlib.Path:
    """ONE toy flat ghost-GK dataset IN ``root``: games ``100..`` alternating a detection-aware
    ``skillcorner`` (keeper ``visibility`` flag, so the detected-only filter and the selection-bias counters
    run; string ids) and a fully-observed ``gradientsports`` (integer ids), the real tc3 id mix; frames 1 s
    apart (so ``--subsample-fps 1`` keeps every frame); SPADL actions and ``home_teams.json``. The first
    game is HEAVY (``heavy_frames``)."""
    return build_flat_corpus(
        root,
        n_games=n_games,
        providers=("skillcorner", "gradientsports"),
        n_frames=n_frames,
        dt=1.0,
        heavy_frames=heavy_frames,
        detection_aware=frozenset({"skillcorner"}),
    )


def xshot_frame_rows(fid: int, t: float, ball_x: float) -> list[dict]:
    """One frame's full-schema rows: ball + team-1 (defends x=0) + team-2 (attacks, carrier near the ball).
    ``ball_x`` drives the goal distance ``r``, the dominant feature. Moved from
    ``tests/tracking/test_xshot_occurrence_integration.py::_xshot_frame_rows`` (ADR-128; that name stays an
    alias) so the toy corpus and the integration tests share ONE synthesis."""
    rows = [
        dict(
            player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=ball_x, y=34.0, frame_id=fid, time_seconds=t
        ),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0, frame_id=fid, time_seconds=t),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0, frame_id=fid, time_seconds=t),
        # team-2 carrier 0.3 m from the ball -> possession resolves to team 2.
        dict(
            player_id=21,
            team_id=2,
            is_ball=False,
            is_goalkeeper=False,
            x=ball_x + 0.3,
            y=34.0,
            frame_id=fid,
            time_seconds=t,
        ),
    ]
    for k in range(5):  # team-1 outfield defenders
        rows.append(
            dict(
                player_id=11 + k,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=6.0 + k,
                y=28.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    for k in range(4):  # team-2 outfield attackers
        rows.append(
            dict(
                player_id=22 + k,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=ball_x + 2 + k,
                y=30.0 + 2 * k,
                frame_id=fid,
                time_seconds=t,
            )
        )
    return rows


def finalize_xshot_frames(rows: list[dict], game) -> pd.DataFrame:
    """The constant per-game columns of a toy xShot frame table (moved with ``xshot_frame_rows``)."""
    frames = pd.DataFrame(rows)
    frames["game_id"] = game
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    return frames


def xshot_game(game, *, episodes: int = 20, degenerate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One toy xShot game: ``(frames, shots)``.

    LEARNABLE (``degenerate=False``): pre-shot frames put the ball NEAR the attacked goal (small ``r``) and
    quiet frames keep it in the attacking third but farther (larger ``r``), so ``r`` cleanly predicts the
    label -- ``episodes`` x (3 positive near + 3 negative far). DEGENERATE: every frame has IDENTICAL
    features (ball fixed at x=20), so no model beats the base rate and the acceptance gates must fail.
    """
    from silly_kicks.spadl import config as spadlconfig

    rows: list[dict] = []
    shot_times: list[float] = []
    fid = 0
    for ep in range(episodes):
        if degenerate:
            base = ep * 6.0
            shot_times.append(base + 0.3)
            specs = [(base, 20.0), (base + 2.5, 20.0), (base + 3.5, 20.0)]
        else:
            base = ep * 7.0
            shot_times.append(base + 0.9)  # one shot covers the 3 near frames (each within 1 s)
            specs = [(base, 8.0), (base + 0.3, 8.0), (base + 0.6, 8.0)]
            specs += [(base + 2.5, 30.0), (base + 3.0, 30.0), (base + 3.5, 30.0)]
        for t, ball_x in specs:
            rows.extend(xshot_frame_rows(fid, t, ball_x))
            fid += 1
    shots = pd.DataFrame(
        {
            "game_id": [game] * len(shot_times),
            "period_id": [1] * len(shot_times),
            "team_id": [2] * len(shot_times),
            "time_seconds": shot_times,
            "type_id": [spadlconfig.actiontype_id["shot"]] * len(shot_times),
        }
    )
    return finalize_xshot_frames(rows, game), shots


#: The toy xShot corpus's providers, one per game: 6 SkillCorner then 2 Gradient Sports (so a three-arm
#: payload has public / sc_extended / full arms that all differ -- the paired path).
XSHOT_TOY_PROVIDERS = ("skillcorner",) * 6 + ("gradientsports",) * 2


def build_xshot_corpus(
    root: str | pathlib.Path,
    *,
    providers: Sequence[str] = XSHOT_TOY_PROVIDERS,
    episodes: int = 20,
    heavy_episodes: int | None = 120,
    degenerate: bool = False,
) -> pathlib.Path:
    """ONE toy flat xShot dataset IN ``root``: game ``100 + g`` of ``providers[g]``, frames (with
    ``source_provider``) + the shot actions as ``_actions/`` + ``home_teams.json``. The first game is HEAVY
    (``heavy_episodes``), so at W >= 2 it finishes last in its extract job (the parity non-vacuity)."""
    root = pathlib.Path(root)
    shards = root / "shards" / TOY_GENERATION
    actions_dir = root / "_actions"
    shards.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    home: dict[str, object] = {}
    for g, provider in enumerate(providers):
        game = 100 + g
        kind = PROVIDER_ID_KIND[provider]
        n_ep = heavy_episodes if (heavy_episodes is not None and g == 0) else episodes
        frames, shots = xshot_game(game, episodes=n_ep, degenerate=degenerate)
        frames["source_provider"] = provider
        stem = f"{provider}__{game}"
        staged_frames(frames, kind).to_parquet(shards / f"{stem}.parquet", index=False)
        staged_actions(shots, kind).to_parquet(actions_dir / f"{stem}.parquet", index=False)
        home[str(game)] = staged_home(1, kind)
    (root / "home_teams.json").write_text(json.dumps(home, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def xshot_arms(root: str | pathlib.Path, *, paired: bool) -> str:
    """The launcher's neutral ``--arms-json`` for a toy xShot dataset, keyed by the IN-JOB match key
    ``<source_provider>__<provider>-<game>`` (``arm_policy._injob_key``; not imported here, so the
    launcher-only policy module stays out of every import closure but the launcher's). ``paired``: games 0-3
    ``public``, + the other SkillCorner games ``sc_extended``, + everything ``full`` (the three-arm paired
    path); else ONE all-``public`` arm."""
    stems = sorted(p.stem for p in (pathlib.Path(root) / "shards" / TOY_GENERATION).glob("*.parquet"))
    stems.sort(key=lambda s: int(s.partition("__")[2]))
    return _xshot_arms_payload([s.partition("__")[0] for s in stems], paired=paired)


def _xshot_arms_payload(providers: Sequence[str], *, paired: bool) -> str:
    """`xshot_arms` from the per-game providers alone (game ``100 + g`` of ``providers[g]``)."""
    keys = [f"{prov}__{prov}-{100 + g}" for g, prov in enumerate(providers)]
    if not paired:
        return json.dumps({"arms": {"public": keys}, "ship_order": ["public"]}, sort_keys=True)
    public = keys[:4]
    extended = public + [k for k in keys[4:] if k.startswith("skillcorner__")]
    payload = {
        "arms": {"public": public, "sc_extended": extended, "full": keys},
        "ship_order": ["public", "sc_extended", "full"],
    }
    return json.dumps(payload, sort_keys=True)


def xcross_frame_rows(fid: int, t: float, ball_x: float) -> list[dict]:
    """One frame's rows in the WIDE corridor (y=8). Team-1 defends x=0 (GK x=2); team-2 attacks + carries
    the ball near the byline. ``ball_x`` (= carrier gr_x, goal at 0) drives dist_endline, the learnable
    signal (near byline -> cross imminent -> positive). Moved from
    ``tests/tracking/test_xcross_attempt_integration.py::_xcross_frame_rows`` (ADR-128) so the toy corpus
    and the integration tests share ONE synthesis."""

    def _r(pid, tid, x, y, *, is_ball=False, is_gk=False):
        return dict(
            player_id=pid,
            team_id=tid,
            is_ball=is_ball,
            is_goalkeeper=is_gk,
            x=x,
            y=y,
            frame_id=fid,
            time_seconds=t,
        )

    rows = [
        _r(-1, -1, ball_x, 8.0, is_ball=True),
        _r(10, 1, 2.0, 34.0, is_gk=True),
        _r(20, 2, 103.0, 34.0, is_gk=True),
        _r(21, 2, ball_x + 0.3, 8.0),  # team-2 carrier 0.3 m from the ball -> possession = team 2
    ]
    rows += [_r(11 + k, 1, 6.0 + k, 28.0 + 2 * k) for k in range(5)]  # team-1 defenders (box near x=0)
    rows += [_r(22 + k, 2, 8.0 + k, 30.0 + 2 * k) for k in range(4)]  # team-2 attackers (box)
    return rows


def finalize_xcross_frames(rows: list[dict], game, provider: str = "synthetic") -> pd.DataFrame:
    """The constant per-game columns of a toy xCross frame table (moved with ``xcross_frame_rows``)."""
    frames = pd.DataFrame(rows)
    frames["game_id"] = game
    frames["period_id"] = 1
    frames["z"] = 0.0
    frames["frame_rate"] = 10.0
    frames["ball_state"] = "alive"
    frames["source_provider"] = provider
    frames["vx"] = 0.0
    frames["vy"] = 0.0
    return frames


def xcross_game(game, *, episodes: int = 20, learnable: bool = True, provider: str = "synthetic"):
    """One toy xCross game: ``(frames, actions)``. LEARNABLE: near-byline frames (small dist_endline)
    precede a cross -> positive; farther wide frames -> negative. ``learnable=False`` fixes the carrier
    position, so no model beats the base rate (the gates FAIL)."""
    from silly_kicks.spadl import config as spadlconfig

    rows: list[dict] = []
    cross_times: list[float] = []
    fid = 0
    for ep in range(episodes):
        base = ep * 7.0
        cross_times.append(base + 0.9)  # one cross covers the 3 near frames (each within 1 s)
        near_x = 8.0 if learnable else 20.0
        far_x = 30.0 if learnable else 20.0
        specs = [(base, near_x), (base + 0.3, near_x), (base + 0.6, near_x)]
        specs += [(base + 2.5, far_x), (base + 3.0, far_x), (base + 3.5, far_x)]
        for t, ball_x in specs:
            rows.extend(xcross_frame_rows(fid, t, ball_x))
            fid += 1
    actions = pd.DataFrame(
        {
            "game_id": [game] * len(cross_times),
            "period_id": [1] * len(cross_times),
            "team_id": [2] * len(cross_times),
            "time_seconds": cross_times,
            "type_id": [spadlconfig.actiontype_id["cross"]] * len(cross_times),
            "result_id": [spadlconfig.result_id["success"]] * len(cross_times),
        }
    )
    return finalize_xcross_frames(rows, game, provider), actions


def build_xcross_corpus(
    root: str | pathlib.Path,
    *,
    providers: Sequence[str] = XSHOT_TOY_PROVIDERS,
    episodes: int = 20,
    heavy_episodes: int | None = 120,
    learnable: bool = True,
) -> pathlib.Path:
    """ONE toy flat xCross dataset IN ``root``: game ``100 + g`` of ``providers[g]``, frames + the cross
    actions as ``_actions/`` + ``home_teams.json`` (team 1 is home; xcross signs score_differential from
    HOME). The first game is HEAVY (``heavy_episodes``) for the extract non-vacuity."""
    root = pathlib.Path(root)
    shards = root / "shards" / TOY_GENERATION
    actions_dir = root / "_actions"
    shards.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    home: dict[str, object] = {}
    for g, provider in enumerate(providers):
        game = 100 + g
        kind = PROVIDER_ID_KIND[provider]
        n_ep = heavy_episodes if (heavy_episodes is not None and g == 0) else episodes
        frames, actions = xcross_game(game, episodes=n_ep, learnable=learnable, provider=provider)
        stem = f"{provider}__{game}"
        staged_frames(frames, kind).to_parquet(shards / f"{stem}.parquet", index=False)
        staged_actions(actions, kind).to_parquet(actions_dir / f"{stem}.parquet", index=False)
        home[str(game)] = staged_home(1, kind)
    (root / "home_teams.json").write_text(json.dumps(home, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def build_xcross_probe_corpus(
    root: str | pathlib.Path, *, n_games: int = 3, provider: str = "gradientsports", episodes: int = 6
) -> pathlib.Path:
    """ONE toy PER-GAME probe dataset IN ``root`` (the staged ``xcross_probe`` layout:
    ``<provider>-<game>/frames.parquet`` + ``actions.parquet``) of ``provider`` games ``200 + g``."""
    from scripts.hf_jobs.datasets import pergame_name

    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        game = 200 + g
        frames, actions = xcross_game(game, episodes=episodes, provider=provider)
        kind = PROVIDER_ID_KIND[provider]
        gdir = root / pergame_name(provider, str(game))
        gdir.mkdir(parents=True, exist_ok=True)
        staged_frames(frames, kind).to_parquet(gdir / "frames.parquet", index=False)
        staged_actions(actions, kind).to_parquet(gdir / "actions.parquet", index=False)
    return root


#: The committed raw StatsBomb event fixtures a toy open-data tree is built from.
_SB_RAW_EVENTS = pathlib.Path(__file__).resolve().parents[1] / "tests" / "datasets" / "statsbomb" / "raw" / "events"


def _goals(events: list[dict], team_id: int) -> int:
    """Goals scored FOR ``team_id``: its shots that ended in a goal plus the opponents' own goals."""
    n = 0
    for e in events:
        etype = e.get("type", {}).get("name")
        team = e.get("team", {}).get("id")
        if etype == "Shot" and team == team_id and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            n += 1
        if etype == "Own Goal For" and team == team_id:
            n += 1
    return n


#: The synthesized HEAVY toy match: it sorts FIRST among the open-data keys and carries about seven times
#: the events of any fixture (measured warm: 1.8 s against 0.2-0.4 s), so at W >= 2 it finishes last in its
#: job and completion order differs from key order -- the parity tests' non-vacuity (spec 5.2), which a
#: per-worker import warm-up of about 2.5 s would otherwise turn into a coin flip. Built from fixture 7298
#: replayed six times as extra time.
HEAVY_MATCH_ID = "1000000"


def _shift_clock(stamp: str, minutes: int) -> str:
    """``HH:MM:SS.mmm`` plus ``minutes``."""
    h, m, s = stamp.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}:{s}"


def _with_extra_time(events: list[dict], *, copies: int) -> list[dict]:
    """``events`` followed by ``copies`` replays of its periods 1-2 as extra-time periods 3-4.

    Each replay gets fresh ids (links kept), a later block of the period clock (60 minutes per replay), and
    indices after everything before it, so the match stays chronological and only grows.
    """
    out = list(events)
    for k in range(1, copies + 1):
        base = len(out)
        for e in events:
            if e.get("period") not in (1, 2) or e.get("type", {}).get("name") == "Starting XI":
                continue
            c = json.loads(json.dumps(e))
            c["id"] = f"{e['id']}-et{k}"
            c["index"] = int(e["index"]) + base
            c["period"] = int(e["period"]) + 2
            c["minute"] = int(e.get("minute", 0)) + 60 * k
            c["timestamp"] = _shift_clock(str(e.get("timestamp", "00:00:00.000")), 60 * (k - 1))
            if "related_events" in c:
                c["related_events"] = [f"{r}-et{k}" for r in c["related_events"]]
            out.append(c)
    return out


def build_open_data(
    root: str | pathlib.Path,
    *,
    match_ids: Sequence[str] = ("3754058", "7298", "7584"),
    heavy_first: bool = True,
    heavy_id: str = HEAVY_MATCH_ID,
    aliases: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """Write a toy StatsBomb open-data tree (the ``github.com/statsbomb/open-data`` layout) under ``root``.

    ``data/competitions.json`` lists ONE competition (43/106), ``data/matches/43/106.json`` lists the
    matches, ``data/events/<id>.json`` copies the committed raw fixtures, and ``data/lineups/<id>.json`` is
    built from each match's two "Starting XI" events. The first "Starting XI" team is the home team. With
    ``heavy_first`` the tree also holds the heavy match ``heavy_id`` (default ``HEAVY_MATCH_ID``, see its
    comment), listed first. ``aliases`` (``{new id: fixture id}``) adds copies of a fixture under new ids. Serve it with
    ``tests/scripts/_open_data_server.serve_open_data`` and point statsbombpy at it with
    ``SILLY_KICKS_OPEN_DATA_BASE`` (``scripts._sb_open_data.pin_open_data_revision``).
    """
    root = pathlib.Path(root)
    data = root / "data"
    for sub in ("matches/43", "events", "lineups"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    sources = {mid: json.loads((_SB_RAW_EVENTS / f"{mid}.json").read_bytes()) for mid in match_ids}
    if heavy_first:
        heavy = _with_extra_time(json.loads((_SB_RAW_EVENTS / "7298.json").read_bytes()), copies=6)
        sources = {heavy_id: heavy, **sources}
    for new_id, fixture in (aliases or {}).items():
        sources[new_id] = json.loads((_SB_RAW_EVENTS / f"{fixture}.json").read_bytes())
    comp = {
        "competition_id": 43,
        "season_id": 106,
        "country_name": "International",
        "competition_name": "FIFA World Cup",
        "competition_gender": "male",
        "competition_youth": False,
        "competition_international": True,
        "season_name": "2022",
        "match_updated": "2024-01-01T00:00:00",
        "match_available": "2024-01-01T00:00:00",
    }
    matches = []
    for i, (mid, events) in enumerate(sources.items()):
        (data / "events" / f"{mid}.json").write_text(json.dumps(events), encoding="utf-8")
        xi = [e for e in events if e.get("type", {}).get("name") == "Starting XI"][:2]
        lineups = [
            {
                "team_id": e["team"]["id"],
                "team_name": e["team"]["name"],
                "lineup": [
                    {
                        "player_id": p["player"]["id"],
                        "player_name": p["player"]["name"],
                        "player_nickname": None,
                        "jersey_number": p.get("jersey_number"),
                        "country": {"id": 0, "name": "Unknown"},
                        "cards": [],
                        "positions": [],
                    }
                    for p in e["tactics"]["lineup"]
                ],
            }
            for e in xi
        ]
        (data / "lineups" / f"{mid}.json").write_text(json.dumps(lineups, indent=1), encoding="utf-8")
        home, away = xi[0]["team"], xi[1]["team"]
        matches.append(
            {
                "match_id": int(mid),
                "match_date": f"2022-11-{20 + i:02d}",
                "kick_off": "16:00:00.000",
                "competition": {k: comp[k] for k in ("competition_id", "country_name", "competition_name")},
                "season": {"season_id": 106, "season_name": "2022"},
                "home_team": {"home_team_id": home["id"], "home_team_name": home["name"], "home_team_gender": "male"},
                "away_team": {"away_team_id": away["id"], "away_team_name": away["name"], "away_team_gender": "male"},
                "home_score": _goals(events, home["id"]),
                "away_score": _goals(events, away["id"]),
                "match_status": "available",
                "match_week": 1,
            }
        )
    (data / "competitions.json").write_text(json.dumps([comp], indent=1), encoding="utf-8")
    (data / "matches" / "43" / "106.json").write_text(json.dumps(matches, indent=1), encoding="utf-8")
    return root


def _outfield_players() -> list[dict]:
    def r(pid, team, x, y, *, gk=False, ball=False, vx=0.2, vy=-0.1):
        return {
            "period_id": 1,
            "team_id": (pd.NA if ball else team),
            "player_id": (pd.NA if ball else pid),
            "is_ball": ball,
            "is_goalkeeper": gk,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "team_in_possession": 1,
        }

    return [
        r(101, 1, 3.0, 34.0, gk=True),
        r(102, 1, 20.0, 10.0),
        r(103, 1, 22.0, 27.0),
        r(104, 1, 24.0, 44.0),
        r(105, 1, 26.0, 60.0),
        r(106, 1, 45.0, 20.0),
        r(107, 1, 55.0, 40.0),
        r(201, 2, 102.0, 34.0, gk=True),
        r(202, 2, 18.0, 30.0),
        r(203, 2, 30.0, 40.0),
        r(204, 2, 55.0, 25.0),
        r(205, 2, 70.0, 50.0),
        r(206, 2, 85.0, 22.0),
        r(pd.NA, pd.NA, 55.0, 30.0, ball=True, vx=1.5, vy=0.5),
    ]


def outfield_frames(game_id: str, provider: str = "toy", n_frames: int = 25) -> pd.DataFrame:
    """Jittered 2-team frames with ``team_in_possession`` for the ghost-outfield trainer. Moved from
    ``tests/scripts/test_train_ghost_outfield.py::_tiny_frames`` (ADR-128; that name stays an alias) so the
    toy corpus and the unit tests share one synthesis."""
    base = _outfield_players()
    rows = []
    for k in range(n_frames):
        dx = 3.0 * np.sin(0.3 * k)
        dy = 2.0 * np.cos(0.4 * k)
        for p in base:
            row = dict(p)
            row["game_id"] = game_id
            row["frame_id"] = 1000 + k
            row["time_seconds"] = 100.0 + 0.5 * k
            row["source_provider"] = provider  # for the StratifiedGroupKFold(match+provider) CV
            if not row["is_ball"]:
                row["x"] = float(min(104.0, max(1.0, row["x"] + dx)))
                row["y"] = float(min(67.0, max(1.0, row["y"] + dy)))
            rows.append(row)
    return pd.DataFrame(rows)


def build_outfield_corpus(
    root: str | pathlib.Path, *, n_games: int = 6, n_frames: int = 25, heavy_frames: int = 600
) -> pathlib.Path:
    """ONE toy flat ghost-outfield dataset IN ``root``: games ``100..`` alternating ``gradientsports``
    (integer ids) and ``skillcorner`` (string ids) -- the real tc3 id mix, and two providers so
    StratifiedGroupKFold and the per-provider MAE run -- frames + an empty actions table +
    ``home_teams.json``. The first game is HEAVY (``heavy_frames`` frames), so at W >= 2 it finishes last
    in its extract job (the parity non-vacuity)."""
    root = pathlib.Path(root)
    shards = root / "shards" / TOY_GENERATION
    shards.mkdir(parents=True, exist_ok=True)
    (root / "_actions").mkdir(parents=True, exist_ok=True)
    home: dict[str, object] = {}
    for g in range(n_games):
        game = str(100 + g)
        provider = "gradientsports" if g % 2 == 0 else "skillcorner"
        kind = PROVIDER_ID_KIND[provider]
        n = heavy_frames if g == 0 else n_frames
        frames = staged_frames(outfield_frames(game, provider=provider, n_frames=n), kind)
        frames.to_parquet(shards / f"{provider}__{game}.parquet", index=False)
        home[game] = staged_home(1, kind)
    (root / "home_teams.json").write_text(json.dumps(home, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


#: SPADL columns a toy receiver match carries (the shape `tests/scripts/test_train_receiver_model.py` uses).
_RECEIVER_ACTION_COLS = [
    "action_id",
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def receiver_match(game_id: int, *, n_passes: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` of one SB360-style toy match: ``n_passes`` completed passes whose intended
    receiver sits ON the release->reception ray (so the TRAJECTORY strategy labels it) with off-ray
    teammates as negatives, one freeze frame per pass. Moved from
    ``tests/scripts/test_train_receiver_model.py::_match`` (ADR-128) so the toy corpus and that test share
    one synthesis."""
    from silly_kicks.spadl.config import actiontypes, results

    pass_id = {n: i for i, n in enumerate(actiontypes)}["pass"]
    success = {n: i for i, n in enumerate(results)}["success"]
    acts, frames = [], []
    aid, t, fid = 1, 10.0, game_id * 1000
    for k in range(n_passes):
        receiver = 10 + k % 3  # 10, 11, 12
        acts.append((aid, game_id, 1, t, 1, 9, pass_id, success, 50, 34, 70, 34))  # pass toward (70,34)
        acts.append((aid + 1, game_id, 1, t + 0.5, 1, receiver, pass_id, success, 70, 34, 30, 40))  # the touch
        rows = [
            (True, pd.NA, pd.NA, False, 50.0, 34.0),
            (False, 9, 1, False, 50.0, 34.0),  # passer
            (False, receiver, 1, False, 70.0, 34.0),  # intended receiver -- ON the +x ray (perp 0)
            (False, 50, 1, False, 55.0, 55.0),  # off-ray teammate
            (False, 51, 1, False, 55.0, 12.0),  # off-ray teammate
            (False, 20, 2, False, 60.0, 34.0),
            (False, 30, 2, True, 100.0, 34.0),
        ]
        fr = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y"])
        fr["game_id"], fr["period_id"], fr["time_seconds"], fr["frame_id"] = game_id, 1, t, fid
        fr["source_provider"] = "statsbomb"
        frames.append(fr.astype({"player_id": "Int64", "team_id": "Int64"}))
        aid, t, fid = aid + 2, t + 5.0, fid + 1
    return pd.DataFrame(acts, columns=_RECEIVER_ACTION_COLS), pd.concat(frames, ignore_index=True)


def build_receiver_corpus(root: str | pathlib.Path, *, n_games: int = 4, heavy_passes: int = 400) -> pathlib.Path:
    """ONE toy flat SB360-style dataset IN ``root`` (``shards/<gen>/statsbomb__<g>.parquet`` + ``_actions/``;
    no ``home_teams.json``, like the staged sb360 dataset). The first game is HEAVY (``heavy_passes``
    passes, the others 3), so at W >= 2 it finishes last in its job (the parity non-vacuity)."""
    root = pathlib.Path(root)
    shards = root / "shards" / TOY_GENERATION
    actions_dir = root / "_actions"
    shards.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    for g in range(n_games):
        game = 100 + g
        actions, frames = receiver_match(game, n_passes=heavy_passes if g == 0 else 3)
        kind = PROVIDER_ID_KIND["statsbomb"]
        staged_frames(frames, kind).to_parquet(shards / f"statsbomb__{game}.parquet", index=False)
        staged_actions(actions, kind).to_parquet(actions_dir / f"statsbomb__{game}.parquet", index=False)
    return root


@contextlib.contextmanager
def serve_open_data(root: str | pathlib.Path) -> Iterator[str]:
    """Serve a toy open-data tree over local http; yields the base URL (``SILLY_KICKS_OPEN_DATA_BASE``).

    Lives here (not in ``tests/``) because preflight check 5 runs every open-data module's toy
    ``--run-all`` through it; ``tests/scripts/_open_data_server.py`` re-exports it for the tests.
    """
    import functools
    import http.server
    import threading

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):  # the stdlib signature (keeps test output quiet)
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


#: Where an open-data module's toy tree lives under its data root.
_OPEN_DATA_DIR = "open_data"


def _open_data_toy(root: pathlib.Path) -> pathlib.Path:
    build_open_data(root / _OPEN_DATA_DIR)
    return root


#: The census toy: 5 matches sorted 3754058, 7298, 7300000 (heavy), 7584, 7600000 (a copy of 7584). With
#: ``--fit-fraction 0.4`` the leakage-disjoint split fits xT on the two lowest ids -- 7298 has goals, so the
#: xT surface is non-zero (3754058 alone has none) -- and scores three, the HEAVY one first in key order
#: (the parity non-vacuity: a job holding it and a later unit completes out of key order).
CENSUS_HEAVY_MATCH_ID = "7300000"


def _census_toy(root: pathlib.Path) -> pathlib.Path:
    build_open_data(root / _OPEN_DATA_DIR, heavy_id=CENSUS_HEAVY_MATCH_ID, aliases={"7600000": "7584"})
    return root


def _outfield_toy(root: pathlib.Path) -> pathlib.Path:
    build_outfield_corpus(root / "toyof")
    return root


def _receiver_toy(root: pathlib.Path) -> pathlib.Path:
    build_receiver_corpus(root / "toyrecv")
    return root


def _ghost_toy(root: pathlib.Path) -> pathlib.Path:
    build_ghost_corpus(root / "toygk")
    return root


def _xshot_toy(root: pathlib.Path) -> pathlib.Path:
    build_xshot_corpus(root / "toyxs")
    return root


def _xcross_toy(root: pathlib.Path) -> pathlib.Path:
    build_xcross_corpus(root / "toyxc")
    build_xcross_probe_corpus(root / "toyxcprobe")
    return root


#: How many copies of its frames the scaffold toy's first game carries: once a scaffolded module's extract
#: does real per-frame work, that unit is the slow one, so its parity test can assert out-of-order completion.
_SCAFFOLD_HEAVY_COPIES = 120


def scaffold_flat_toy(dataset: str) -> Callable[[pathlib.Path], pathlib.Path]:
    """The toy builder a scaffolded pipeline registers (`python -m scripts.hf_jobs scaffold`): ONE flat
    dataset ``dataset`` of 4 SkillCorner games under the data root; the first game is HEAVY -- its frames
    replicated ``_SCAFFOLD_HEAVY_COPIES`` times along ``frame_id`` (vectorized, so the build stays cheap), for
    the module's later out-of-order non-vacuity. Returns the data root."""

    def build(root: pathlib.Path) -> pathlib.Path:
        root.mkdir(parents=True, exist_ok=True)
        base = build_flat_corpus(root / dataset, n_games=4, providers=["skillcorner"])
        first = sorted((base / "shards").glob("*/*.parquet"))[0]
        frames = pd.read_parquet(first)
        span = int(frames["frame_id"].max()) + 1
        copies = [frames.assign(frame_id=frames["frame_id"] + k * span) for k in range(_SCAFFOLD_HEAVY_COPIES)]
        pd.concat(copies, ignore_index=True).to_parquet(first, index=False)
        return root

    return build


#: Module -> the CLI arguments its toy `--run-all` needs (Tasks 10-16 add one entry each).
TOY_PARAMS: dict[str, tuple[str, ...]] = {
    "scripts.train_win_probability": ("--competition-id", "43", "--season-id", "106"),
    "scripts.train_receiver_model": (
        "--dataset",
        "toyrecv",
        "--provider",
        "statsbomb",
        "--feature-set",
        "public",
        "--min-passes",
        "1",
        "--corpus-visibility",
        "restricted",
        "--corpus-n-matches",
        "4",
    ),
    "scripts.train_ghost_outfield": (
        "--dataset",
        "toyof",
        "--variant",
        "default",
        "--n-estimators",
        "10",
        "--max-depth",
        "3",
        "--cv-folds",
        "4",
    ),
    "scripts.train_ghost_gk": (
        "--dataset",
        "toygk",
        "--variant",
        "default",
        "--n-estimators",
        "10",
        "--max-depth",
        "3",
        "--cv-folds",
        "4",
        "--perm-importance-sample",
        "200",
    ),
    "scripts.train_xshot_occurrence": (
        "--dataset",
        "toyxs",
        "--arms-json",
        _xshot_arms_payload(XSHOT_TOY_PROVIDERS, paired=True),
        "--n-trials",
        "4",
    ),
    "scripts.train_xcross_attempt": (
        "--dataset",
        "toyxc",
        "--probe-dataset",
        "toyxcprobe",
        "--arms-json",
        _xshot_arms_payload(XSHOT_TOY_PROVIDERS, paired=True),
        "--n-trials",
        "4",
    ),
    # two fit matches (the lowest ids; one has goals), three scored (the heavy one first)
    "scripts.build_territory_ranking_census": (
        "--competition-id",
        "43",
        "--season-id",
        "106",
        "--fit-fraction",
        "0.4",
    ),
    # --- scaffold: toys above ---
}

#: Module -> the builder that writes its toy dataset under a data root and returns that root.
TOY_BUILDERS: dict[str, Callable[[pathlib.Path], pathlib.Path]] = {
    "scripts.train_win_probability": _open_data_toy,
    "scripts.train_receiver_model": _receiver_toy,
    "scripts.train_ghost_outfield": _outfield_toy,
    "scripts.train_ghost_gk": _ghost_toy,
    "scripts.train_xshot_occurrence": _xshot_toy,
    "scripts.train_xcross_attempt": _xcross_toy,
    "scripts.build_territory_ranking_census": _census_toy,
    # --- scaffold: toy builders above ---
}

#: Modules whose toy input is a served open-data tree (``toy_session`` starts the server).
TOY_OPEN_DATA: frozenset[str] = frozenset({"scripts.train_win_probability", "scripts.build_territory_ranking_census"})


@contextlib.contextmanager
def toy_session(module: str, root: str | pathlib.Path) -> Iterator[tuple[pathlib.Path, dict[str, str]]]:
    """Build ``module``'s toy input under ``root`` and yield ``(data_root, extra_env)`` for its CLI run."""
    data = build_toy_for(module, root)
    if module not in TOY_OPEN_DATA:
        yield data, {}
        return
    with serve_open_data(data / _OPEN_DATA_DIR) as url:
        yield data, {"SILLY_KICKS_OPEN_DATA_BASE": url}


def build_toy_for(module: str, root: str | pathlib.Path) -> pathlib.Path:
    """Build ``module``'s toy dataset under ``root`` (the ``--data-root``) and return ``root``."""
    if module not in TOY_BUILDERS:
        raise KeyError(f"{module} has no toy builder in scripts/_toy_corpora.TOY_BUILDERS")
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return TOY_BUILDERS[module](root)
