"""Owner-gated e2e: reconcile_goals on the owner SkillCorner corpus, 0 wrong (Task 2.8 Step 3, G3).

``@pytest.mark.e2e`` + token-gated, so it is skipped in the default ``-m "not e2e"`` CI. It runs the
shipped converter over every owner SkillCorner match that carries the ``match.json`` goal truth and
asserts per-player goals equal ``match.json``, except the pinned allow-list (the 3 score-defect matches,
spec section 10.2). Needs the owner pining token; run on the owner/DGX with ``SILLY_KICKS_RUN_PINING_E2E=1``
and ``PINING_FOR_THE_DATA_TOKEN`` set.
"""

import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

#: Score-defect matches excluded from the zero-wrong bar (spec section 10.2 G3 allow-list).
_ALLOW_LIST = {"2053105", "2050969", "2052202"}


@pytest.mark.skipif(
    not os.environ.get("SILLY_KICKS_RUN_PINING_E2E"),
    reason="owner pining e2e; set SILLY_KICKS_RUN_PINING_E2E=1 + PINING_FOR_THE_DATA_TOKEN (owner/DGX)",
)
def test_owner_corpus_goal_reconciliation_is_zero_wrong():
    import pandas as pd

    from scripts._loader_pining import (
        _base_url,
        _download_artifacts,
        _read_json,
        _resolve_token,
        list_match_refs,
    )
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.spadl import config as cfg
    from silly_kicks.spadl import skillcorner

    def cid(value: object) -> str:
        return str(canonical_id(value))

    token, base = _resolve_token(None), _base_url()
    refs = list_match_refs(providers=["skillcorner"], token=token, base_url=base)
    shot_ids = {cfg.actiontype_id[k] for k in ("shot", "shot_penalty", "shot_freekick")}

    wrong: list[tuple[str, str, int, int]] = []
    checked = 0
    for ref in refs:
        if ref.match_id in _ALLOW_LIST:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            paths = _download_artifacts(
                "skillcorner", ref.match_id, dict(ref.artifacts), token, base, Path(tmp), events_only=True
            )
            meta = _read_json(paths["metadata"])
            if not meta.get("players"):
                continue  # no per-player goal truth -> cannot score G3 on this match
            actions, _ = skillcorner.convert_to_actions(pd.read_csv(paths["events"], low_memory=False), meta)
        checked += 1
        credit: dict[str, int] = {}
        for tid, rid, player in zip(actions["type_id"], actions["result_id"], actions["player_id"], strict=True):
            if tid in shot_ids and rid == cfg.result_id["success"]:
                credit[cid(player)] = credit.get(cid(player), 0) + 1
        mj = {cid(p["id"]): int(p.get("goal") or 0) for p in meta["players"]}
        for pid in set(credit) | set(mj):
            if credit.get(pid, 0) != mj.get(pid, 0):
                wrong.append((ref.match_id, pid, credit.get(pid, 0), mj.get(pid, 0)))

    assert checked > 0, "no owner SkillCorner matches with goal truth were loaded"
    assert not wrong, f"{len(wrong)} wrong per-player goal assignment(s) over {checked} matches: {wrong[:10]}"
