# Contributing to silly-kicks

## Development Setup

```bash
git clone https://github.com/karsten-s-nielsen/silly-kicks.git
cd silly-kicks
pip install -e ".[kloppy,xgboost,test,dev]"
```

## Running Tests

```bash
# Unit tests (fast, no external data needed)
python -m pytest tests/ -m "not e2e" -v

# With coverage
python -m pytest tests/ -m "not e2e" --cov=silly_kicks
```

## Code Quality

```bash
# Lint
ruff check silly_kicks/ tests/
ruff format silly_kicks/ tests/

# Type check
pyright silly_kicks/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD preferred)
3. Ensure all CI checks pass: `ruff check`, `ruff format --check`, `pyright`, `pytest`
4. Keep commits focused — one logical change per commit
5. Include a clear description of what and why

## Architecture Guidelines

- **Hexagonal / pure-core**: All core functions are pure — pandas in, pandas out.
  Zero I/O, zero global state mutation.
- Converters return `tuple[pd.DataFrame, ConversionReport]`
- Use `np.select` for vectorized dispatch, not `apply(axis=1)`
- Add `stacklevel=2` to all `warnings.warn()` calls
- New public functions need docstrings with Parameters/Returns sections

For the full system-level view, open
[`docs/c4/architecture.html`](docs/c4/architecture.html) in a browser (C4
Container diagram). Architectural Decision Records live under
`docs/superpowers/adrs/` — each captures the context, decision, and
consequences for a significant design choice (e.g. ADR-001 on converter
identifier conventions, ADR-005 on tracking-VAEP composition).

## Test Categories

| Marker | Command | What it covers |
|--------|---------|----------------|
| *(none)* | `pytest tests/ -m "not e2e"` | Unit + integration — runs in CI on every push |
| `e2e` | `pytest tests/ -m e2e` | End-to-end tests requiring downloaded datasets (not in CI) |
| `slow` | `pytest tests/ -m slow` | Longer-running fits or sweeps |
| `benchmark` | `pytest tests/ --benchmark-only` | `pytest-benchmark` micro-benchmarks with hard CI budgets |

Tests with fixtures committed to the repo should **not** be marked `e2e` — they
run in the regular suite.

## Benchmark Expectations

Performance-sensitive code ships with `pytest-benchmark` tests under `tests/`.
CI enforces hard time budgets (typically 100 ms Linux / 150 ms Windows).
When modifying a benchmarked function, run the benchmark locally before pushing:

```bash
python -m pytest tests/ -k "benchmark" --benchmark-only
```

If your change regresses a benchmark beyond its budget, investigate before
opening a PR.
