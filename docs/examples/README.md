# silly-kicks examples

End-to-end usage examples for silly-kicks public API.

Each script is **documentation, not a test**. They run against user-supplied data directories (paths passed at the command line) and demonstrate the canonical conversion → enrichment → labels pipeline for a given provider.

## Scripts

| Script | Provider | Demonstrates |
|---|---|---|
| `pff_wc2022_walkthrough.py` | PFF FC | JSON parsing → SPADL → Atomic-SPADL → coverage / boundary metrics → VAEP labels |

## Convention for new examples

When adding a new provider walkthrough:

1. Name: `<provider>_<dataset_or_release>_walkthrough.py`.
2. Take dataset path as a CLI argument (no hard-coded paths, no env-var gating, no test-discovery hooks).
3. Demonstrate the full pipeline: events → SPADL → Atomic-SPADL → `coverage_metrics` → `boundary_metrics` → VAEP labels.
4. Print progress to stdout — these run interactively.
