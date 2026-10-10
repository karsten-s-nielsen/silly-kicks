# How-to: the silly-kicks agent-support MCP server

> Class-2 procedural runbook (`docs/howto`). The MCP server (`silly_kicks/mcp/`) exposes three
> **read-only** tripwire/verdict tools that a document cannot enforce because they must *execute*. It
> is a pure adapter — each tool binds an existing lib/compute seam or reads a committed memo, owns no
> analysis logic, and never writes an artifact or touches the tree. Any MCP client can use it (it is
> not Claude-specific).

## Install (optional extra)

The `mcp` dependency is opt-in; the core library never imports it.

```bash
uv sync --extra mcp        # managed; updates uv.lock
# or: pip install -e ".[mcp]"
```

Run it directly (stdio transport):

```bash
python -m silly_kicks.mcp.server
```

## Register it with a client

**Claude Code** — the repo ships `.mcp.json`; nothing more is needed when the repo venv (with the
`mcp` extra) is active:

```json
{ "mcpServers": { "silly-kicks": { "command": "python", "args": ["-m", "silly_kicks.mcp.server"], "cwd": "." } } }
```

**Other MCP clients (Codex / Cursor / …)** — register the identical command
`python -m silly_kicks.mcp.server` (working directory = the repo root, repo venv with the `mcp`
extra). The server is the same for every client.

## The three tools

All three are **read-only** and **fail loud** (they RAISE, never return a verdict, on a bad load /
untrustworthy memo — silent degradation is the exact defect the tripwire exists to catch, ADR-052 D14):

- **`check_orientation(match_ref, provider?)`** — is the match's direction resolved, and does the
  stored per-period `team_attacking_direction` label match the data geometry? Binds
  `scripts/measure_rc4_orientation.measure()` (raw metrics) + `orient_frames_to_ltr_by_geometry`
  (data-only per-period reflect/no-reflect decision). `verdict`: `UNORIENTED` (direction unresolved —
  the RC4 NULL-label no-op), `MISMATCH` (label present but contradicts geometry — a defensive
  future-regression check; consistent adapters ship coord-label-consistent frames), else `OK`.
- **`diagnose_provider(provider, match_ref, aspect)`** — a provider data-quality probe.
  `aspect ∈ {keeper, convention, id_dtype, coords}`: `keeper`→`validate_gk_position_clamp`
  (`GkClampDiagnosis`), `convention`→`detect_input_convention`, `id_dtype`→`validate_id_dtypes`
  (the `str()`-on-float ADR-019 trap), `coords`→`diagnose_coordinates` (a scale/units + bounds/NaN
  tripwire over the SPADL actions and/or tracking frames — NOT orientation, use `check_orientation`
  for that; off-pitch TRACKING positions are reported as INFO, never a defect). `findings` is
  JSON-safe for every aspect; `flags` surfaces the diagnosis booleans.
- **`validate_construct_validity(metric_family)`** — surfaces a family's RECORDED
  construct-validity verdict from its committed `docs/research/<family>/…json` memo (no corpus re-run).
  Families: `gk_decision`, `territorial_defense` (both return the recorded verdict + provenance);
  `xtgk_possession_value` currently **RAISES** — its `gate.json` carries no `run_commit`/
  `run_tree_dirty`, so the reader refuses until the memo is regenerated with provenance
  (a deferred owner-corpus run).

## Notes

- The server binds a PURE read-only seam per tool — never a driver's `run()`/`main()`, no
  `require_clean_tree`, no artifact write. Safe to call mid-work on a dirty tree.
- All ids route through `silly_kicks.id_compat` (never raw `str()`/`==`).
- `.claude/` activation and the neutral runbooks are Phase 1 (`docs/howto/authoring-a-metric.md` etc.).
