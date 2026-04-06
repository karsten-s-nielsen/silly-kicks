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

- All core functions are pure: pandas in, pandas out
- Converters return `tuple[pd.DataFrame, ConversionReport]`
- Use `np.select` for vectorized dispatch, not `apply(axis=1)`
- Add `stacklevel=2` to all `warnings.warn()` calls
- New public functions need docstrings with Parameters/Returns sections
