"""The idsse penalties of the raw stage matches -> shot_penalty rows; goal rows by the shared goal rule.

Input (read only): the raw provider cache (--cache-dir, idsse). Run from a clone root with PYTHONPATH=. (it imports
scripts._loader_pining)."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import _loader_pining as lp
from silly_kicks.spadl import config as c
from silly_kicks.spadl._goals import goal_mask, owngoal_mask

ap = argparse.ArgumentParser(description="The idsse penalty rows and goal rows.")
ap.add_argument("--cache-dir", required=True, help="raw provider cache root (<provider>/<match>/)")
root = Path(ap.parse_args().cache_dir) / "idsse"
tot_goals = 0
for d in sorted(root.iterdir()):
    mid = d.name
    paths = {"metadata": d / f"idsse_{mid}_metadata.xml", "events": d / f"idsse_{mid}_events.xml"}
    actions, _home = lp._build_match_actions("idsse", mid, paths)
    pen = actions[actions["type_id"] == c.actiontype_id["shot_penalty"]]
    tree = ET.parse(d / f"idsse_{mid}_events.xml")  # noqa: S314 -- the provider's raw file
    raw_shooters = [s.get("Player") for p in tree.getroot().iter("Penalty") for s in p.findall("ShotAtGoal")]
    n_goal = int((goal_mask(actions) | owngoal_mask(actions)).sum())
    tot_goals += n_goal
    for r in pen.to_dict("records"):
        print(mid, "shot_penalty", c.results[int(r["result_id"])], r["player_id"], r["team_id"], r["time_seconds"],
              r["time_source"], "raw_shooters", raw_shooters)
    if len(pen) == 0:
        print(mid, "no shot_penalty; raw Penalty>ShotAtGoal shooters:", raw_shooters)
    print(mid, "goal rows (the shared goal rule):", n_goal)
print("TOTAL goal rows (the shared goal rule, idsse stage, events-only):", tot_goals)
