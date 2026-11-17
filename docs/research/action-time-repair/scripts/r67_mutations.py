"""R67 / bar A29 point 4: each rule of the probe has a mutation that turns the gate red (scratch copy, not the repo)."""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\Development\karstenskyt__silly-kicks_part-deux")
SCR = Path(r"D:\Temp\xshot187\r67_mut")
if SCR.exists():
    shutil.rmtree(SCR)
shutil.copytree(REPO / "silly_kicks", SCR / "silly_kicks", ignore=shutil.ignore_patterns("__pycache__", "*_weights"))
for d in ("tests", "tests/tracking"):
    (SCR / d).mkdir(parents=True, exist_ok=True)
    (SCR / d / "__init__.py").write_text("", encoding="utf-8")
for f in ("tests/tracking/_match_context_probe.py", "tests/tracking/test_match_context_revision_gate.py"):
    shutil.copy(REPO / f, SCR / f)
(SCR / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

G = "silly_kicks/tracking/_ghost_gk.py"
O = "silly_kicks/spadl/_goals.py"
MUTATIONS = [
    ("R51 period key (score)", G, "_PERIOD_KEY_SCALE = 1.0e6", "_PERIOD_KEY_SCALE = 0.0"),
    ("R51 period key (phase)", G, "key = (str(game_id), int(period_id))", "key = (str(game_id), 1)"),
    ("R52 own goal counts for the other team", G, "if own.any() and len(all_teams) == 2:", "if False:"),
    ("R52 shoot-out row is no goal", G, '(g["period_id"].to_numpy() != 5)', '(g["period_id"].to_numpy() != 99)'),
    ("R52 direct free-kick goal", O, 'frozenset({"shot", "shot_penalty", "shot_freekick"})',
     'frozenset({"shot", "shot_penalty"})'),
    ("R57/R58 penalty goal", O, 'frozenset({"shot", "shot_penalty", "shot_freekick"})',
     'frozenset({"shot", "shot_freekick"})'),
    ("R56 effect at the NEXT native row", G, "ev_keys.append(float(keys[i + 1]))", "ev_keys.append(float(keys[i]))"),
    ("R56 derived rows not read", G, 'g[(g["action_provenance"] == "native").to_numpy()]',
     'g[(g["action_provenance"] != "x").to_numpy()]'),
    ("R56 a decrease falls back to the goal rows", G, "if np.any(np.diff(home) < 0) or np.any(np.diff(away) < 0):",
     "if False:"),
    ("R50 order_only row opens no window", G, "actions = actions[~is_order_only(actions)]", "actions = actions"),
    ("R50 the 10 s window", G, "_SET_PIECE_DECAY_SECONDS = 10.0", "_SET_PIECE_DECAY_SECONDS = 11.0"),
    ("R50 the phase starts AT its row's time (side=right)", G,
     'np.searchsorted(times, time_s, side="right")', 'np.searchsorted(times, time_s, side="left")'),
    ("R50 goal-kick phase code 2", G, 'sp["_phase_code"] = np.where(sp["type_id"].to_numpy() == gk_id, 2, 1)',
     'sp["_phase_code"] = np.where(sp["type_id"].to_numpy() == gk_id, 1, 1)'),
]


def run() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/tracking/test_match_context_revision_gate.py", "-q",
                        "-p", "no:cacheprovider", "-k", "current_revision"], cwd=SCR, capture_output=True, text=True)
    return "RED" if r.returncode else "green"


print("baseline (no mutation):", run())
for name, rel, a, b in MUTATIONS:
    p = SCR / rel
    orig = p.read_text(encoding="utf-8")
    n = orig.count(a)
    if n != 1:
        print(f"{name}: anchor found {n} times -- NOT RUN")
        continue
    p.write_text(orig.replace(a, b), encoding="utf-8")
    print(f"{name}: {run()}")
    p.write_text(orig, encoding="utf-8")
print("baseline again:", run())
