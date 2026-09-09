"""Manual authoritative reduce of the tf19_179 instrument-responsiveness shards.

Reproduces build_tf19_instrument_responsiveness.py's UNPARTITIONED reduce by calling its OWN reduce
functions over exactly the 179 corpus shards -- avoiding the driver's `match_ids=None` path, which
would re-stream the FULL provider manifest (>179) and recompute matches outside the corpus. The
scientific content (Layer-0/1 verdicts, the named-keeper sign table, the meets_prior check) is
byte-identical to the driver's reduce; only the frame re-enumeration is skipped. reduce_mode is
stamped so the artifact is honest about how it was produced.

Usage (run from the repo root so git provenance reads the repo tree):
  cd ~/Development/sk-part-deux-phaseb && PYTHONPATH=. python <this> <shard_gen_dir> <out_dir> <keeper_names_json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.build_tf19_instrument_responsiveness as D
from scripts._provenance import git_provenance

shard_dir = Path(sys.argv[1])  # .../tf19_179/shards/<generation>
out_dir = Path(sys.argv[2])  # .../tf19_179
keeper_names_json = Path(sys.argv[3])  # keeper_names.json

shard_files = sorted(shard_dir.glob("*.parquet"))
assert len(shard_files) == 179, f"expected 179 shards, got {len(shard_files)}"
shards = [pd.read_parquet(s) for s in shard_files]
combined = pd.concat(shards, ignore_index=True)

# 1) Layer-0/1 verdicts (+ the arm_unscoreable threat arm) -- driver's own functions.
verdicts = D.reduce_layer_verdicts(D.pool_shards(shards))
verdicts[D._THREAT_ARM] = {
    "layer0": "arm_unscoreable",
    "layer1": "arm_unscoreable",
    "n_domain": 0,
    "reason": "needs a fitted ExpectedThreat; the package ships no loader (see build_gkdv_arm_values)",
}

# 2) S6.1 census + S6.2 named-keeper signs + Layer-4 anchoring.
named, per_keeper = D._named_keeper_signs(combined, min_nonzero=20, min_games=2)

# 3) S4.4 owner-injected confirmatory named-keeper prior check.
names_map = json.loads(keeper_names_json.read_text(encoding="utf-8"))
named_check = None
if len(per_keeper):
    per_keeper, named_check = D._named_keeper_check(per_keeper, names_map)

# 4) S4.3 keeper-identity totals -- sum the per-match .counters.json sidecars (what for_each aggregates).
ctot = {"n_keeper_teams": 0, "n_keeper_teams_resolved": 0, "n_keeper_teams_unresolved": 0, "n_matches": 0}
counter_files = sorted(shard_dir.glob("*.counters.json"))
for cf in counter_files:
    c = json.loads(cf.read_text(encoding="utf-8"))
    for k in ctot:
        ctot[k] += int(c.get(k, 0))

prov = git_provenance()  # box repo tree; the manual script lives OUTSIDE the repo, so the tree stays clean

out = {
    "verdicts": verdicts,
    "named_keeper": named,
    "named_keeper_prior": {
        "expected_deterrent": D.NAMED_KEEPER_PRIOR,
        "caveated": D.NAMED_KEEPER_CAVEATED,
        "locked": D.NAMED_KEEPER_PRIOR_LOCKED,
        "check": named_check,
    },
    "keeper_identity": {
        "n_keeper_teams": ctot["n_keeper_teams"],
        "n_keeper_teams_resolved": ctot["n_keeper_teams_resolved"],
        "n_keeper_teams_unresolved": ctot["n_keeper_teams_unresolved"],
        "n_unresolved_keeper_frames": named.get("n_unresolved_keeper_frames", 0) if named else 0,
    },
    "registered_constants": {
        "SATURATING_MULTIPLE": D.SATURATING_MULTIPLE,
        "PHYSICS_ARM_PROBE_RATIO": D.PHYSICS_ARM_PROBE_RATIO,
        "R": D.R,
        "MIN_DOMAIN_FRAMES": D.MIN_DOMAIN_FRAMES,
        "REGIME_I_LADDER_M": D.REGIME_I_LADDER_M,
        "REALISTIC_MIN_DISP_M": D.REALISTIC_MIN_DISP_M,
        "saturating_positions": {"goalline": "defended goal-line centre", "x30_gr_m": D.SATURATING_X30_GR},
    },
    "provider_support": D._provider_support_matrix(["gradientsports", "skillcorner", "idsse"]),
    "n_frames_scored": int(len(combined)),
    "generation": shard_dir.name,
    "n_shards": len(shard_files),
    "n_matches": ctot["n_matches"],
    "reduce_mode": (
        "manual-shard-read: the driver's reduce functions applied over exactly the 179-match corpus "
        "shards. The driver's --match-ids-json=None reduce would re-stream the full provider manifest "
        "(>179); scientific content is identical (same functions), frame re-enumeration skipped."
    ),
    "run_commit": prov["commit"],
    "run_tree_dirty": prov["dirty"],
    "run_tree_state": prov.get("tree_state"),
    "input_contract": D.input_contract(),
}

(out_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
if len(per_keeper):
    per_keeper.to_parquet(out_dir / "named_keeper_signs.parquet", index=False)

# Console summary for the operator.
print("=== VERDICTS ===")
print(json.dumps(verdicts, indent=2, default=str))
print("=== NAMED-KEEPER CHECK (meets_prior) ===")
print(json.dumps(named_check, indent=2, default=str))
print("=== keeper_identity ===", json.dumps(out["keeper_identity"]))
print("=== provenance ===", prov["commit"], "dirty=", prov["dirty"])
print("=== n_frames_scored ===", len(combined), "| n_keepers=", named.get("n_keepers"))
