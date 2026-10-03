"""Hub smoke over the org's whole frame-geometry population (combined-cycle-completion spec section 9).

Anonymous downloads only (no token): each repo loads fail-closed (chirality + feature contract inside
load) and scores the canonical float32 frame to finite values. Publishes nothing.
    python scripts/validate_hub_variants.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORG = "silly-kicks"
#: repo -> (silly_kicks.tracking class, role). role: hf_only | mirror (a wheel variant republished) | event_only.
HUB_REGISTRY: dict[str, tuple[str, str]] = {
    "silly-kicks/xshot-occurrence-v1": ("XShotOccurrenceModel", "hf_only"),
    "silly-kicks/xshot-occurrence-position-only-v1": ("XShotOccurrenceModel", "hf_only"),
    "silly-kicks/xcross-attempt-v1": ("XCrossAttemptModel", "hf_only"),
    "silly-kicks/xcross-attempt-position-only-v1": ("XCrossAttemptModel", "hf_only"),
    "silly-kicks/ghost-gk-v1": ("GhostGkModel", "hf_only"),  # the Hub-only `full` variant (spec 0.10)
    "silly-kicks/ghost-gk-sweeper-v1": ("GhostGkModel", "mirror"),
    "silly-kicks/ghost-gk-sweeper-position-only-v1": ("GhostGkModel", "mirror"),
    "silly-kicks/ghost-outfield-v1": ("GhostOutfieldModel", "mirror"),
    "silly-kicks/ghost-outfield-position-only-v1": ("GhostOutfieldModel", "mirror"),
    "silly-kicks/xsuccess-v1": ("", "event_only"),
}


def check_population(listed: set[str]) -> None:
    """ADR-056: the org listing must equal the registry exactly."""
    unregistered, missing = sorted(listed - set(HUB_REGISTRY)), sorted(set(HUB_REGISTRY) - listed)
    if unregistered:
        raise SystemExit(f"unregistered Hub repo(s): {unregistered} -- classify them in HUB_REGISTRY")
    if missing:
        raise SystemExit(f"missing Hub repo(s): {missing} -- the registry names repos the org no longer lists")


def readme_matches_card(repo_id: str, *, download, root: Path = Path(".")) -> bool:
    """D9: the Hub README equals the registered in-repo card, both LF-normalized (spec section 9)."""
    from scripts._hub_publish import CARD_SOURCE, card_bytes

    hub = Path(download(repo_id, "README.md")).read_bytes().replace(b"\r\n", b"\n")
    return hub == card_bytes(Path(root) / CARD_SOURCE[repo_id])


def smoke_repo(repo_id: str, cls_name: str, *, from_hub, score, refusals: tuple[type[BaseException], ...] = ()) -> dict:
    """Load + score one repo. A fail-closed load REFUSAL (one of ``refusals``: the library's own integrity
    errors, e.g. a pre-ADR-089 chirality fingerprint) is recorded as a fact about the published artifact,
    never a crash that loses every other repo's result; any other exception propagates."""
    try:
        model = from_hub(repo_id)
    except refusals as exc:
        first = (str(exc).splitlines() or [""])[0]
        return {
            "loaded": False,
            "load_error": f"{type(exc).__name__}: {first}",
            "n_scores": 0,
            "finite": False,
            "training_commit": None,
            "class": cls_name,
        }
    vals = [float(v) for v in score(model)]
    return {
        "loaded": True,
        "n_scores": len(vals),
        "finite": bool(vals) and all(math.isfinite(v) for v in vals),
        "training_commit": getattr(model, "training_commit", None),
        "class": cls_name,
    }


def _float32_canonical_frame():
    """The driver's own copy of tests/test_bundled_models_load_on_float32_commit1.py::_float32_canonical_frame
    (scripts must not import tests -- the same rule validate_das_native_parity._run_native follows).
    test_hub_variants_driver asserts the two stay byte-identical."""
    import numpy as np
    import pandas as pd

    rows = [
        dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=20.0, y=34.0),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0),
        dict(player_id=11, team_id=1, is_ball=False, is_goalkeeper=False, x=10.0, y=30.0),
        dict(player_id=12, team_id=1, is_ball=False, is_goalkeeper=False, x=12.0, y=38.0),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0),
        dict(player_id=21, team_id=2, is_ball=False, is_goalkeeper=False, x=20.3, y=34.0),
        dict(player_id=22, team_id=2, is_ball=False, is_goalkeeper=False, x=25.0, y=30.0),
    ]
    df = pd.DataFrame(rows)
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    df["time_seconds"], df["frame_rate"] = 0.0, 25.0
    for c in ("x", "y"):
        df[c] = df[c].astype("float32")
    for c in ("z", "vx", "vy", "speed"):
        df[c] = np.float32(0.0)
    df["ball_state"] = "alive"
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64").astype("category")
    return df


def _score_fn(cls_name: str):
    """The model's own serve path on the canonical float32 frame."""
    from silly_kicks import tracking

    frame = _float32_canonical_frame()

    def _xy(out):  # the served ghost coordinates only (never ids/frame keys, which are always finite)
        return out.filter(regex=r"^ghost_.*_[xy]$").stack().dropna()

    if cls_name == "XShotOccurrenceModel":
        return lambda m: tracking.compute_xshot_occurrence(frame, model=m, home_team_id=1)["xshot_occurrence"].dropna()
    if cls_name == "XCrossAttemptModel":
        return lambda m: tracking.compute_xcross_attempt(frame, model=m, home_team_id=1)["xcross_attempt"].dropna()
    if cls_name == "GhostGkModel":
        return lambda m: _xy(tracking.serve_ghost_gk_positions(frame, model=m, home_team_id=1))
    return lambda m: _xy(tracking.serve_ghost_outfield_positions(frame, model=m, home_team_id=1))


#: mirror repo -> the wheel bundle dir it republishes (spec 0.10, D9). The post-push gate compares their
#: training_commit (spec 9 step 4).
MIRROR_BUNDLE: dict[str, str] = {
    "silly-kicks/ghost-gk-sweeper-v1": "_ghost_gk_weights/sweeper",
    "silly-kicks/ghost-gk-sweeper-position-only-v1": "_ghost_gk_weights/sweeper_position_only",
    "silly-kicks/ghost-outfield-v1": "_ghost_outfield_weights/default",
    "silly-kicks/ghost-outfield-position-only-v1": "_ghost_outfield_weights/position_only",
}
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_refusals() -> tuple[type[BaseException], ...]:
    """The fail-closed load refusals of the frame-geometry model classes (xcross reuses xshot's)."""
    from silly_kicks.tracking import _ghost_gk, _ghost_outfield, _xshot_occurrence

    return (_ghost_gk.IntegrityError, _ghost_outfield.IntegrityError, _xshot_occurrence.IntegrityError)


def _training_commit(path) -> str | None:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("training_commit")


def run(
    *,
    listed: set[str],
    revision,
    download,
    load_model,
    score_for,
    prov: dict,
    root: Path,
    out: Path,
    load_refusals: tuple[type[BaseException], ...] = (),
    require_cards_match: bool = False,
    require_mirrors_match_wheel: bool = False,
) -> dict:
    """The whole smoke over injected Hub access (offline-testable). Writes ``out/hub_smoke.json``; the two
    ``require_*`` flags turn the recorded card / mirror comparisons into gates (the post-push D9 check).

    A repo the library refuses to load is recorded under ``load_refused``. A refused Hub-only repo always
    fails (this cycle never republishes those); a refused mirror is recorded pre-push and gated by
    ``require_mirrors_match_wheel`` (the D9 republish is what replaces it)."""
    check_population(listed)
    results: dict[str, dict] = {}
    for repo_id, (cls_name, role) in sorted(HUB_REGISTRY.items()):
        rec: dict = {"role": role, "readme_matches_card": readme_matches_card(repo_id, download=download, root=root)}
        if role == "event_only":
            results[repo_id] = {**rec, "skipped": "event-only model: no frame geometry"}
            continue
        rec["revision"] = revision(repo_id)
        rec["hub_training_commit"] = _training_commit(download(repo_id, "metadata.json"))
        if role == "mirror":
            rec["wheel_training_commit"] = _training_commit(
                Path(root) / "silly_kicks" / "tracking" / MIRROR_BUNDLE[repo_id] / "metadata.json"
            )
            rec["mirror_matches_wheel"] = rec["hub_training_commit"] == rec["wheel_training_commit"]
        results[repo_id] = {
            **rec,
            **smoke_repo(
                repo_id,
                cls_name,
                from_hub=lambda r, c=cls_name: load_model(c, r),
                score=score_for(cls_name),
                refusals=load_refusals,
            ),
        }
    load_refused = sorted(r for r, v in results.items() if v.get("loaded") is False)
    ok = all(v.get("finite", True) for v in results.values() if v.get("loaded") is not False)
    cards_mismatched = sorted(r for r, v in results.items() if not v["readme_matches_card"])
    mirrors_mismatched = sorted(r for r, v in results.items() if v.get("mirror_matches_wheel") is False)
    doc = {
        "org": ORG,
        "repos": results,
        "all_finite": ok,
        "load_refused": load_refused,
        "cards_mismatched": cards_mismatched,
        "mirrors_mismatched": mirrors_mismatched,
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "hub_smoke.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit("a Hub variant did not score finite values -- see hub_smoke.json")
    refused_hub_only = [r for r in load_refused if results[r]["role"] == "hf_only"]
    if refused_hub_only:
        raise SystemExit(f"Hub-only repo(s) refused to load: {refused_hub_only} -- see hub_smoke.json")
    if require_mirrors_match_wheel and load_refused:
        raise SystemExit(f"mirror(s) refused to load: {load_refused} -- see hub_smoke.json")
    if require_cards_match and cards_mismatched:
        raise SystemExit(f"Hub README differs from its in-repo card for {cards_mismatched} -- see hub_smoke.json")
    if require_mirrors_match_wheel and mirrors_mismatched:
        raise SystemExit(
            f"mirror training_commit differs from the wheel for {mirrors_mismatched} -- see hub_smoke.json"
        )
    return doc


def _live_run(**kw) -> dict:
    """``run`` against the live Hub, ANONYMOUSLY (B r4 CCC-PLAN-35): ``from_hub`` calls snapshot_download
    without a token argument, so the implicit login is disabled before huggingface_hub is imported.
    Otherwise a gated or private repo would load under the operator's login and read as public."""
    import os

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # noqa: S105 -- an env FLAG name, not a secret
    os.environ.pop("HF_TOKEN", None)
    from huggingface_hub import HfApi, hf_hub_download

    from silly_kicks import tracking

    api = HfApi()
    return run(
        listed={m.id for m in api.list_models(author=ORG, token=False)},
        revision=lambda repo_id: api.model_info(repo_id, token=False).sha,
        download=lambda repo_id, f: hf_hub_download(repo_id, f, token=False, force_download=True),
        load_model=lambda cls_name, repo_id: getattr(tracking, cls_name).from_hub(repo_id),
        score_for=_score_fn,
        load_refusals=_load_refusals(),
        root=_REPO_ROOT,
        **kw,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--require-cards-match",
        action="store_true",
        help="fail unless every Hub README equals its registered in-repo card (the post-release D9 check)",
    )
    ap.add_argument(
        "--require-mirrors-match-wheel",
        action="store_true",
        help="fail unless every mirror's Hub training_commit equals its wheel bundle's (the post-release D9 check)",
    )
    args = ap.parse_args(argv)

    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)  # the entry-point gate
    doc = _live_run(
        prov=prov,
        out=args.out,
        require_cards_match=args.require_cards_match,
        require_mirrors_match_wheel=args.require_mirrors_match_wheel,
    )
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
