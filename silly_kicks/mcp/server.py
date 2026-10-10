"""silly-kicks agent-support MCP server (Phase 2): three read-only tripwire/verdict tools.

Pure adapter. Each tool composes a fail-loud read-only seam (match-load ``_load`` or memo-read
``_memo``) with a bound compute seam, and returns a JSON-serializable dict. The server owns no
analysis logic beyond arg-normalization + one geometry-vs-label comparison (``check_orientation``).
Requires the ``mcp`` extra; the FastMCP import is guarded so the core library is unaffected.

Run: ``python -m silly_kicks.mcp.server`` (stdio).
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from typing import Any

import numpy as np

from . import _load, _memo  # importing _load adds scripts/ to sys.path (for the lazy binds below)

try:  # guarded: the core lib + non-MCP consumers never need FastMCP
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        "silly_kicks.mcp.server requires the 'mcp' optional extra: pip install 'silly-kicks[mcp]' "
        "(or: uv sync --extra mcp)."
    ) from exc

app = FastMCP("silly-kicks")

_NO_ANCHOR_PERIOD = re.compile(r"period=(\d+)")
_KNOWN_PERIODS = (1, 2, 3, 4)


# --------------------------------------------------------------------------- serialization


def _json_safe(obj: Any) -> Any:
    """Recursively coerce a diagnosis (dataclass / dict / numpy) to JSON-serializable Python.

    Dataclasses -> ``asdict``; TUPLE dict keys -> ``"a|b"`` strings (``GkClampDiagnosis`` keys its
    per-unit dicts by ``(game_id, team_id)``); numpy scalars/arrays -> Python (``DetectionResult``
    ``diagnostics`` carries numpy group means).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {_json_key(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def _json_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(str(_json_safe(k)) for k in key)
    if isinstance(key, np.generic):
        return str(key.item())
    return str(key) if not isinstance(key, str) else key


def tool_names() -> set[str]:
    """The registered tool names.

    Reads FastMCP's ``_tool_manager.list_tools()`` — a PRIVATE attribute, used because it is the only
    SYNC tool listing in mcp 1.x (``app.list_tools()`` is async). The coupling is deliberate and guarded
    (P2-IMPL-01): the ``mcp[cli]>=1,<2`` pin blocks the 2.x FastMCP move, and
    ``tests/mcp/test_registration.py`` fails loudly if this private surface changes. Swap to a public
    sync accessor if mcp ever ships one.
    """
    return {t.name for t in app._tool_manager.list_tools()}


# --------------------------------------------------------------------------- check_orientation


def _periods_without_anchor(caught: list[warnings.WarningMessage]) -> set[int]:
    out: set[int] = set()
    for w in caught:
        msg = str(w.message)
        if "no GK anchor" in msg:
            m = _NO_ANCHOR_PERIOD.search(msg)
            if m:
                out.add(int(m.group(1)))
    return out


def _geometry_agrees(loaded: Any) -> bool:
    """Compare the stored per-period home label to the bound function's reflect/no-reflect decision.

    Binds ``orient_frames_to_ltr_by_geometry`` (away-GK fallback / period-5 / no-anchor skip all live
    inside it); reads the per-period decision from input-vs-output coord reflection; excludes period 5
    and periods the function could not anchor. No present label to contradict -> agrees.
    """
    frames = getattr(loaded, "frames", None)
    if frames is None or "team_attacking_direction" not in frames.columns:
        return True
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking import orient_frames_to_ltr_by_geometry

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = orient_frames_to_ltr_by_geometry(
            frames, home_team_id=loaded.home_team_id, on_missing_home="warn", copy=True
        )
    no_anchor = _periods_without_anchor(caught)

    in_x = frames["x"].to_numpy(dtype="float64")
    out_x = out["x"].to_numpy(dtype="float64")
    period = frames["period_id"].to_numpy()
    home = ids_match(frames["team_id"], loaded.home_team_id).fillna(False).to_numpy(dtype=bool)
    tad = frames["team_attacking_direction"]
    present = tad.notna().to_numpy()

    for p in _KNOWN_PERIODS:
        if p in no_anchor:
            continue
        pmask = period == p
        hp = pmask & home & present
        if not hp.any():
            continue
        reflected = not np.allclose(in_x[pmask], out_x[pmask], equal_nan=True)
        geometric = "rtl" if reflected else "ltr"
        stored = {str(v) for v in tad.to_numpy()[hp]}
        if stored != {geometric}:
            return False
    return True


def check_orientation(match_ref: str, provider: str | None = None) -> dict:
    """Orientation tripwire: is the match's direction resolved, and does the label match geometry?

    ``verdict``: ``UNORIENTED`` (direction unresolved — the RC4 NULL-``team_attacking_direction``
    case), ``MISMATCH`` (resolved but a present per-period label contradicts the geometry), else
    ``OK``. A tokenless/empty load RAISES (never returns a verdict).
    """
    loaded = _load.load_match(match_ref, provider)
    import measure_rc4_orientation as _mrc4  # scripts/ (path added by _load import)

    raw = _mrc4.measure(loaded)
    direction_resolved = bool(raw["unlabelled_fraction"] < 1.0)
    geometry_agrees = _geometry_agrees(loaded)
    verdict = "UNORIENTED" if not direction_resolved else ("MISMATCH" if not geometry_agrees else "OK")
    return {
        "match_ref": match_ref,
        "provider": provider,
        "direction_resolved": direction_resolved,
        "geometry_agrees": geometry_agrees,
        "verdict": verdict,
        "measure": _json_safe(raw),
    }


# --------------------------------------------------------------------------- diagnose_provider


def _diagnose(loaded: Any, aspect: str) -> Any:
    from silly_kicks.spadl import detect_input_convention, diagnose_coordinates
    from silly_kicks.tracking import validate_gk_position_clamp, validate_id_dtypes

    if aspect == "keeper":
        return validate_gk_position_clamp(loaded.frames)
    if aspect == "id_dtype":
        return validate_id_dtypes(loaded.actions, loaded.frames, on_mismatch="warn")
    if aspect == "convention":
        from silly_kicks.spadl import config as _cfg

        return detect_input_convention(loaded.actions, match_col="game_id", x_max=float(_cfg.field_length))
    if aspect == "coords":
        return diagnose_coordinates(getattr(loaded, "actions", None), getattr(loaded, "frames", None))
    raise ValueError(f"unknown aspect {aspect!r}; known: keeper|convention|id_dtype|coords")


def _flags(aspect: str, diag: Any) -> list[str]:
    if aspect == "coords":  # the lib seam already produces the frozen flag tokens; adapter passes them through
        return list(getattr(diag, "flags", []))
    flags: list[str] = []
    if aspect == "keeper" and getattr(diag, "clamped", False):
        flags.append("gk_clamped")
    if aspect == "id_dtype" and getattr(diag, "has_mismatch", False):
        flags.append("id_dtype_mismatch")
    if aspect == "convention" and getattr(diag, "convention", "x") is None:
        flags.append("convention_ambiguous")
    return flags


def diagnose_provider(provider: str, match_ref: str, aspect: str) -> dict:
    """Provider data-quality probe. ``aspect`` ∈ {keeper, convention, id_dtype, coords} (lib binds).

    ``coords`` binds :func:`silly_kicks.spadl.diagnose_coordinates` — a scale/units + bounds/NaN
    tripwire over the SPADL actions and/or tracking frames. It does NOT measure orientation
    (use ``check_orientation``) and treats off-pitch TRACKING positions as INFO, not a defect.
    """
    loaded = _load.load_match(match_ref, provider)
    diag = _diagnose(loaded, aspect)
    return {
        "provider": provider,
        "match_ref": match_ref,
        "aspect": aspect,
        "findings": _json_safe(diag),
        "flags": _flags(aspect, diag),
    }


# --------------------------------------------------------------------- validate_construct_validity


def validate_construct_validity(metric_family: str, *, research_root: Any = None) -> dict:
    """Surface a metric family's RECORDED construct-validity verdict (memo reader, read-only).

    Fails loud (raises) on an absent / dirty / unprovenanced memo — never synthesises a verdict.
    """
    memo = _memo.read_validity_memo(metric_family, research_root=research_root)
    memo_path = _memo.memo_path_for(metric_family, research_root=research_root)
    return {
        "metric_family": metric_family,
        "memo_path": str(memo_path),
        "run_commit": memo["run_commit"],
        "run_tree_dirty": memo["run_tree_dirty"],
        "run_commit_is_head": _run_commit_is_head(memo["run_commit"]),
        "verdict": _memo.recorded_verdict(memo),
    }


def _run_commit_is_head(run_commit: Any) -> bool:
    """Informational staleness hint (ADR-056 declare-not-enforce): does the memo's commit == HEAD?"""
    try:
        from _provenance import git_provenance  # scripts/ (path added by _load import)

        head = git_provenance().get("commit")
    except (ImportError, OSError):
        return False
    return bool(run_commit) and bool(head) and str(run_commit) == str(head)


# --------------------------------------------------------------------------- registration


app.add_tool(check_orientation)
app.add_tool(diagnose_provider)
app.add_tool(validate_construct_validity)


if __name__ == "__main__":  # pragma: no cover - stdio entry
    app.run()
