"""silly-kicks agent-support MCP server (Phase 2, ADR-052 D14 tripwires).

Opt-in: requires the ``mcp`` extra (``mcp[cli]>=1,<2``). The core library never imports this
package; the server binds existing read-only compute seams + reads committed validity memos and
owns no analysis logic of its own. Run it with ``python -m silly_kicks.mcp.server`` (stdio).

Nothing here is imported at core-library import time — importing ``silly_kicks`` does NOT import
``silly_kicks.mcp`` (guarded by the extra; the server module guards the FastMCP import).
"""
