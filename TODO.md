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
| — | Small | Add `preserve_native` parity to Atomic-SPADL converter | Atomic-SPADL has different action types (receival, interception, out, etc.); the SPADL `preserve_native` kwarg landed in 1.1.0 across all four standard SPADL converters (statsbomb / wyscout / opta / kloppy) but Atomic-SPADL was deferred. Apply the same passthrough mechanism (probably via `_finalize_output(extra_columns=...)` extension to atomic) when there's a concrete consumer asking for it. |

