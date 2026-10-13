"""TF-64 mechanical population gate (spec 2026-10-01 §4 test 8; the ADR-056 "derive the population
from code, assert it EXACTLY" idiom).

A keeper-position reader that ships ungated-and-unlisted silently reintroduces anti-pattern E7
(a metric computed on an ~80%-extrapolated SkillCorner keeper). This test DERIVES, from the source,
every production module that touches a keeper-row token, and asserts each is **either** detection-
gated (routes through a TF-64 gate symbol) **or** present in an explicit, reasoned out-of-scope
allowlist. It is deliberately NOT a hand-list of the gated set (that is self-certifying -- the very
failure the `pre_shot_gk_*`/`rd_zone_occupancy`/`gk_closing_time_mean_s` misses proved); the gated set
is derived, and only the out-of-scope exemptions are enumerated (each with its stated reason).

Caveat (recorded for the next author): a literal-token population scan is blind to a consumer sitting
downstream of a shared resolver that already stripped the raw column -- e.g. the `pre_shot_gk_*` chain
reads `ctx.defending_gk_rows`, gated once upstream in ``tracking/utils.py``. That is correct here (the
gate is applied at the resolver), but the scan proves gating of the readers it can SEE, not a global
absence of E7. Keep the gate applied at the resolvers, not only at leaf readers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import silly_kicks

_PKG_ROOT = Path(silly_kicks.__file__).parent

# The keeper-row tokens whose readers must be gated-or-exempt (spec §4 test 8).
_KEEPER_TOKENS = ("is_goalkeeper", "defending_gk_player_id", "_resolve_a_keeper_id")

# A module is GATED iff it references one of the TF-64 detection-gate symbols.
_GATE_SYMBOLS = (
    "keeper_detected",
    "resolve_detected_keeper_xy",
    "detection_aware_provider",
    "detected_mask",
    "has_detected_defending_gk",
)

# Out-of-scope allowlist: a keeper-token module that legitimately does NOT gate, each with its reason.
# Keyed by path relative to the silly_kicks package root (POSIX). Derived from the TF-64 keeper-
# position-reader classification; every entry is either an exclusion-only use (the keeper is filtered
# OUT, never read), an identity/presence resolver (resolves WHO, never consumes x/y), a build/parse/
# schema site, a physics-over-all-players surface (detection handled at corpus admission), a
# model training-target, a published-faithful feature set, or a docstring-only mention.
_OUT_OF_SCOPE: dict[str, str] = {
    # -- exclusion-only: `~is_goalkeeper` builds an outfield-only pool; keeper position never read --
    "positioning/_compute.py": "exclusion-only: keeper filtered out of the movable candidate pool",
    "tracking/_cover_shadows.py": "exclusion-only: keeper excluded from blocker/receiver pools",
    "tracking/_defensive_line.py": "exclusion-only: defensive line is outfield-only",
    "tracking/_line_breaking.py": "exclusion-only: non-ball non-GK unit",
    "tracking/_off_ball_runs.py": "exclusion-only: outfield runs",
    "tracking/_opponent_resolution.py": "exclusion-only: opponents are non-GK",
    "tracking/_player_influence.py": "exclusion-only: outfield player influence",
    "tracking/_receiver.py": "exclusion-only: opponent outfield receivers",
    "tracking/_structural_pass.py": "exclusion-only: defenders = opponent outfield",
    "tracking/_team_shape.py": "exclusion-only: team shape is outfield-only",
    "tracking/_collective.py": "opt-in include_goalkeeper, default-excludes; TF-31/44 collective variables",
    "tracking/defensive_credit/_line_break_signal.py": "exclusion-only: ~is_goalkeeper",
    "tracking/defensive_credit/_resolution.py": "exclusion-only: non-GK opponents",
    # NOTE: tracking/_ghost_gk.py (keeper_detection_mask -> detected_mask), coordination/_signals.py
    # (TF-58 outfield detection), and tracking/_provider_visibility.py (the gate definition) all
    # reference a gate symbol, so they classify as GATED and are intentionally NOT listed here.
    # -- identity / presence resolvers: resolve WHO the keeper is, never consume x/y as a metric --
    "keeper_identity.py": "identity resolver: resolves player_id/rung, never reads x/y",
    "tracking/_gk_resolve.py": "identity + resolve_defended_goals orientation primitive (ADR-055)",
    "tracking/_run_features.py": "orchestration docstring; wires the already-gated pre_shot_gk_* family",
    "restdefense/_compute.py": "presence guard via the gated _resolve_a_keeper_id (restdefense/_danger.py)",
    "shot_stopping/_compute.py": "event-derived: pre-stamped defending_gk_player_id + psxg, no frame read",
    "feature_glossary.py": "glossary declaration of the defending_gk_player_id column; not a reader",
    "atomic/spadl/utils.py": "atomic SPADL converter stamps defending_gk_player_id (build); no frame read",
    "atomic/tracking/features.py": "atomic faithful mirror; shares the gated resolver _resolve_action_frame_context",
    # -- build / parse / schema / fixture: derive or declare the flag, no metric compute --
    "providers/gradientsports/appearances.py": "roster classification at parse time",
    "providers/sportec/parse.py": "build-time row tagging",
    "providers/statsbomb/parse.py": "snapshot shaping at parse time",
    "reflection.py": "reflection-kind registry entry",
    "tracking/_chirality.py": "synthetic probe-frame construction",
    "tracking/_feature_contract.py": "synthetic fixture construction",
    "tracking/_gk_identification.py": "DERIVES the is_goalkeeper flag (frame construction)",
    "tracking/_snapshot.py": "frame construction from provider snapshot",
    "tracking/gradientsports.py": "build-time flag derivation",
    "tracking/kloppy.py": "build-time frame conversion",
    "tracking/metrica.py": "build-time frame conversion",
    "tracking/schema.py": "schema dtype/domain declaration",
    "tracking/skillcorner.py": "build-time frame conversion",
    "tracking/sportec.py": "build-time frame conversion",
    "vaep/features/core.py": "empty-frame schema for xfn column discovery; no row data",
    # -- physics over ALL players (keeper is one of 22; detection handled at corpus admission) --
    "tracking/_das_pack.py": "DAS native physics input port; all-player simulation (ADR-107)",
    "tracking/pitch_control/_spearman.py": "pitch-control physics; keeper is one simulated player (TF-7)",
    # -- model training targets / validation harnesses (keeper position is the TARGET, not an input) --
    "tracking/_ghost_outfield.py": "ghost-outfield model: keeper separated from the modeled outfield pool",
    "tracking/_model_eval.py": "ghost-GK substitution-delta validation harness, not a production metric",
    # -- orientation primitive --
    "tracking/direction.py": "orientation primitive: keeper median x fixes whole-frame orientation (ADR-029)",
    # -- published-faithful feature sets (GK block is part of the paper's established contract) --
    "tracking/_xshot_occurrence.py": "xShot (Anzer & Bauer) faithful 27-feature set; GK is published contract",
    "tracking/_xcross_attempt.py": "xCross faithful feature set; GK block is the published contract",
    # -- opt-in, default-exclude (not an E7 pre_shot metric) --
    "tracking/_packing.py": "TF-49 opt-in include_gk, default-excludes; no default xfns",
    "coordination/_compute.py": "TF-58 opt-in include_goalkeeper per family (cluster incl. GK by Duarte's method)",
    # -- counting primitive: gating is delegated to the (gated) caller's include_gk flag --
    "restdefense/_counting.py": "count_goalside primitive; the gated caller (_structure.py) decides include_gk",
    # -- physics-arm probes + Layer-3 counterfactual: restricted to the engine's gated scored domain --
    "gkdv/_probe.py": "physics-arm dose probe over the engine's drop_reason-gated scored domain",
    "restdefense/_probe.py": "physics-arm instrument-validity probe; engine-domain restricted",
    "restdefense/_counterfactual.py": "Layer-3 ghost-substitution engine; gated by the engine domain filter",
    # -- comment / docstring-only mentions (never filtered on) --
    "spadl/utils.py": "comment only; delegates to the gated add_pre_shot_gk_* chain",
}


def _module_text_outside_docstrings(path: Path) -> tuple[str, bool]:
    """Return (scannable source with module/func/class docstrings blanked, parsed_ok)."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, False
    # Blank out docstrings so a required-columns docstring mention is not a false reader.
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                ds = node.body[0]
                for ln in range(ds.lineno - 1, ds.end_lineno or ds.lineno):
                    lines[ln] = ""
    return "\n".join(lines), True


def _keeper_token_modules() -> dict[str, str]:
    """DERIVE the population: production modules whose non-docstring source references a keeper
    token. Returns {relpath: text}."""
    out: dict[str, str] = {}
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text, _ok = _module_text_outside_docstrings(path)
        if any(tok in text for tok in _KEEPER_TOKENS):
            out[path.relative_to(_PKG_ROOT).as_posix()] = text
    return out


def _is_gated(text: str) -> bool:
    return any(sym in text for sym in _GATE_SYMBOLS)


def test_every_keeper_position_reader_is_gated_or_allowlisted():
    """The gate: each derived keeper-token module is detection-gated OR reasoned-allowlisted."""
    population = _keeper_token_modules()
    offenders = [rel for rel, text in population.items() if not _is_gated(text) and rel not in _OUT_OF_SCOPE]
    assert not offenders, (
        "ungated-and-unlisted keeper-position reader(s) -- gate them (route through a TF-64 gate "
        f"symbol) or add a reasoned _OUT_OF_SCOPE entry: {offenders}"
    )


def test_allowlist_has_no_stale_entries():
    """Both-directions (ADR-056): every _OUT_OF_SCOPE entry is still in the derived population, and
    is still ungated (a gated module must not sit in the exemption list)."""
    population = _keeper_token_modules()
    stale = [rel for rel in _OUT_OF_SCOPE if rel not in population]
    assert not stale, f"stale _OUT_OF_SCOPE entries no longer in the population: {stale}"
    wrongly_listed = [rel for rel in _OUT_OF_SCOPE if rel in population and _is_gated(population[rel])]
    assert not wrongly_listed, (
        f"these _OUT_OF_SCOPE modules now reference a gate symbol -- drop them from the allowlist: {wrongly_listed}"
    )


def test_the_gate_catches_an_ungated_reader_plant():
    """Non-vacuity (ADR-056): a synthetic keeper-position reader with no gate symbol and no allowlist
    entry MUST be flagged -- otherwise the gate above could pass vacuously."""
    plant = "df = frames[frames['is_goalkeeper']]\ngx = df.iloc[0]['x']\n"
    assert not _is_gated(plant)
    assert "plant_module.py" not in _OUT_OF_SCOPE
    # the classifier flags it: has a keeper token, not gated, not allowlisted
    assert any(tok in plant for tok in _KEEPER_TOKENS)


def test_the_named_gated_sites_are_actually_gated():
    """Guards the derivation itself: the wired choke-point modules MUST classify as gated (if one
    silently lost its gate symbol, the offender test would still pass by allowlisting -- this
    asserts they are gated, not exempted)."""
    population = _keeper_token_modules()
    # NOTE: gk_decision/_reconstruct.py IS gated (keeper_detected on a["player_id"]) but is NOT a
    # keeper-token-population member -- it resolves the keeper from the ACTION row, not a literal
    # is_goalkeeper/defending_gk_player_id token -- so the literal-token scan cannot see it. Its gate
    # is covered by its own per-site counted-drop test (keeper_undetected), not by this population gate.
    must_be_gated = (
        "restdefense/_structure.py",
        "restdefense/_danger.py",
        "gkdv/_engine.py",
        "tracking/utils.py",
        "tracking/features.py",
        "tracking/_gk_geometry.py",
    )
    for rel in must_be_gated:
        assert rel in population, f"{rel} dropped out of the keeper-token population"
        assert _is_gated(population[rel]), f"{rel} lost its TF-64 gate symbol"
        assert rel not in _OUT_OF_SCOPE, f"{rel} must be gated, not allowlisted"
