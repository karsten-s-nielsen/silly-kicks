# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Documentation

| # | Size | Item | Context |
|---|------|------|---------|
| D-8 | Large | Add docstring `Examples` sections to public functions | 49 public functions, zero examples. Start with the 10 most-used: `convert_to_actions` (×4 providers), `VAEP.fit`, `VAEP.rate`, `gamestates`, `add_names`, `validate_spadl`, `HybridVAEP` |

## Architecture

| # | Size | Item | Context |
|---|------|------|---------|
| A9 | Medium | Reduce `atomic/vaep/features.py` coupling to `vaep/features` (12 imports) | Legitimate delegation today, but tight coupling will fight if atomic features need to diverge independently |
| — | Medium | Decompose `vaep/features.py` (809 lines) | Natural split: spatial features, temporal features, categorical features. Do when next adding features to this file |

## Observability

| # | Size | Item | Context |
|---|------|------|---------|
| — | Small | Log kloppy intra-type silent drops at DEBUG level | Aerial duels and unrecognized GK subtypes silently disappear inside "mapped" event types. ConversionReport counts the event type but individual sub-events vanish without trace |
