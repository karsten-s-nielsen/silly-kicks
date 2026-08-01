# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 4.x     | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue**
2. [Create a GitHub Security Advisory](https://github.com/karsten-s-nielsen/silly-kicks/security/advisories/new)
3. Include: description, reproduction steps, and potential impact

We aim to acknowledge reports within 48 hours and provide a fix timeline within 7 days.

## Security Considerations

silly-kicks is a computation library. The installed wheel contains `silly_kicks/` only — `scripts/`
(loaders, trainers, corpus drivers) is **not** packaged, so nothing in it is reachable from an install.

**Never, anywhere in the library:**
- Executes subprocess commands
- Imports `pickle`, or calls `yaml.load`

**Does, and worth knowing about:**
- **Reads and writes files.** Model artifacts via `save()` / `load()` on the bundled models;
  `calibration.save_xt` / `load_xt` for the frozen xT artifact; `feature_glossary.dump_glossary`.
  `load()` on the bundled models **verifies a SHA256 digest before use** and raises on mismatch, so
  a tampered weights directory fails closed. `load_xt` also checks a digest, but *after* it has
  already deserialized — see the next point; that check detects corruption, it does not stop an
  attack.
- **One deserialization path is NOT pickle-free**, and calling it on an untrusted file is unsafe:
  `calibration.load_xt` uses `np.load(..., allow_pickle=True)`. The SHA256 check happens *after*
  the load, so it does not protect against a hostile file — it detects corruption, not attack.
  **Only load xT artifacts you produced.** Every *bundled model* artifact is plain JSON / `.npz`
  loaded with `allow_pickle=False`; that pickle-free property is a deliberate, load-bearing design
  constraint for the shipped weights and should not be relaxed.
- **Opens network connections, but only on an explicit opt-in call**: the `from_hub()` classmethods
  fetch Hub-distributed weights from HuggingFace. Nothing on the default import or inference path
  reaches the network — the `default` variants ship inside the wheel.

The primary security surface remains input validation on provider DataFrames. Six of the eight SPADL
converters enforce it via `_validate_input_columns()` at their entry point — `statsbomb`, `wyscout`,
`opta`, `sportec`, `metrica` and `gradientsports`. The other two do not. `kloppy` is
structural — it takes a typed `EventDataset`, not a DataFrame, so kloppy's own parser has already
constrained the shape. **`skillcorner` is a genuine gap**: `convert_to_actions(events: pd.DataFrame,
match_metadata: dict)` takes a caller-supplied raw provider DataFrame and indexes it directly, so a
missing column surfaces as a bare `KeyError` rather than the converter's validation error. Two
earlier versions of this line were wrong — first claiming "every converter entry point", then
excusing SkillCorner as structural — and the honest statement is that it is unvalidated.
