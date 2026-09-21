# SK-XT-SER — `ExpectedThreat` serialize seam (`to_dict`/`from_dict`/`save`/`load`)

**Status:** DRAFT (for lakehouse review) · **Proposed version:** <next-free version> / PR-Snnn / **ADR-NNN** (re-derive at commit-prep) · **Base:** branch off `origin/main` (silly-kicks **4.120.0**), NOT the stale 4.119.1 working-tree checkout.

**Origin:** luxury-lakehouse sk4118 adoption (ExT-v2 fold) requested a cross-process transport for a fitted `ExpectedThreat` — handoff `D:\Development\_reviews\2026-09-20-sk-expectedthreat-serialize-seam-handoff.md` (§1, §8 are consumer context only; this spec is written against §2–§7). Same collaboration pattern as SK-EXPORT (4.119.0): silly-kicks ships the seam, the lakehouse consumes it.

---

## 1. Problem

`silly_kicks.xthreat.ExpectedThreat` has `__init__`, `fit`, `interpolator`, `rate`, and plain attributes (`l`/`w`/`eps`/`method`/`params`/`grid`/`xT`/…) — **no serialization** (`values_at_points`/`destination_profiles` are module functions in `_physical.py`/`_counterfactual_seam.py` that read a model, not `ExpectedThreat` members). A fitted model cannot cross a process boundary except by re-`fit()`.

The TF-54b `territory` counterfactual (`compute_territorial_dominance(method="counterfactual", ...)`) and `xthreat.destination_profiles` both require an **injected, fitted** `ExpectedThreat` (`require_fitted_xt` raises `NotFittedError` when `not np.any(model.xT)`). silly-kicks ships **no** xT — the consumer fits it once (on HF Jobs, CPU) and must reconstruct it elsewhere (Databricks serverless) without re-fitting (a full-fact `.toPandas()` fit violates the consumer's boundedness gate). So `ExpectedThreat` needs a serialize/deserialize round-trip.

## 2. Scope

**Additive public API on `ExpectedThreat`** (nothing existing changes; the `fit`/`rate`/`interpolator` paths and the SK-xT-1 frozen-oracle parity are untouched):

| Method | Signature | Behaviour |
|---|---|---|
| `to_dict` | `def to_dict(self) -> dict[str, Any]` | Serialize the complete fitted state to a JSON-round-trippable dict (ndarrays → nested Python lists, NumPy scalars → Python `float`/`int`). **Raises `sklearn.exceptions.NotFittedError`** on an unfitted model (`not np.any(self.xT)` — the same predicate `require_fitted_xt`/`rate` use). |
| `from_dict` | `@classmethod def from_dict(cls, d: Mapping[str, Any]) -> "ExpectedThreat"` | Reconstruct a fitted instance **without calling `.fit()`**. Passes `require_fitted_xt`; produces byte-identical `rate`/`values_at_points`/`destination_profiles`. Fail-closed (§5). |
| `save` | `def save(self, path: str \| os.PathLike) -> None` | Thin wrapper: `json.dump(self.to_dict(), ...)` to a single UTF-8 JSON file. |
| `load` | `@classmethod def load(cls, path: str \| os.PathLike) -> "ExpectedThreat"` | Thin wrapper: `cls.from_dict(json.load(...))`. |

`save`/`load` are **wrappers only** — `to_dict`/`from_dict` are the single source of the serialization logic (no duplicated field lists). All four are **methods on `ExpectedThreat`**, reachable via the already-exported class (`xthreat/__init__.py:32`) — **no `__all__` change** (`__all__` lists module-level names, not methods; adding method names would create undefined entries and break `import *` / `test_public_api_examples.py`).

**Out of scope (recorded, not silently dropped):** SHA256SUMS sidecar / chirality / feature-contract (Decision D5). A future bundled xT variant (the reserved `require_fitted_xt(model: str)` door) is unaffected.

## 3. State to serialize

Verified against `silly_kicks/xthreat/_model.py` (`__init__` :73-93, `fit` :95-131).

**Constructor args (rebuild `__init__`):** `l: int`, `w: int`, `eps: float`, `method: Method` (`"singh_counts"`|`"kde_smoothed"`), `params: XtParams | None`.
- `params`: `None`, or `dataclasses.asdict(params)` (a flat dict — `SinghParams` or `KDEParams`; `KDEParams.kernel` is a `KdeKernel` string literal, JSON-safe). Rebuilt via `_METHOD_TO_PARAMS_TYPE[method](**d)` then `validate_params_for_method(method, params)` (both in `xthreat/_params.py`).

**Fitted arrays (all set by `fit`):**
- `xT` — `(w, l)` float.
- `scoring_prob_matrix`, `shot_prob_matrix`, `move_prob_matrix`, `transition_matrix` — each `NDArray[float64]` (each `None` on an unfitted model).
- `heatmaps` — `list[NDArray[float64]]` (value-iteration trace; `[]` on an unfitted model).

**Derived — NOT serialized:** `grid = GridSpec(n_zones_x=l, n_zones_y=w)`. `from_dict` gets it for free by passing `l`, `w` to the constructor; `GridSpec.__post_init__` validates positivity.

### Schema (`format_version: 1`)
```json
{
  "format_version": 1,
  "l": 16, "w": 12, "eps": 1e-05,
  "method": "singh_counts",
  "params": null,
  "xT": [[...], ...],
  "scoring_prob_matrix": [[...], ...],
  "shot_prob_matrix": [[...], ...],
  "move_prob_matrix": [[...], ...],
  "transition_matrix": [[...], ...],
  "heatmaps": [[[...], ...], ...]
}
```

## 4. Correctness constraints

- **Raw orientation, verbatim (ADR-041).** `ExpectedThreat.xT` stores rows **y-inverted** (neutralized only in `_physical.py`, never in the model). `to_dict`/`from_dict` round-trip **every array bit-for-bit** — no orientation normalization, transpose, or "fix" on either leg. This is a correctness invariant, not a preference: `rate`/`interpolator` assume the internal convention. Pinned by the functional-equivalence test (§6.2).
- **`from_dict` populates every attribute `fit` sets** — `rate`, `interpolator`, `values_at_points`, `destination_profiles` behave identically to the original. dtype and shape preserved across the round trip.
- **Additive, non-perturbing.** The four methods add no state and touch no existing code path. The SK-xT-1 `singh_counts` frozen-oracle parity — asserted in CI by `tests/test_xthreat.py::test_singh_path_byte_identical_to_legacy` (+ `test_singh_path_byte_identical_on_worldcup`), which import the reference-values module `tests/xthreat_legacy_reference.py` — is unaffected (those tests stay green).

## 5. Fail-closed contract (`from_dict` / `load`)

Mirrors the trained-artifact load discipline (ADR-011/040/050) at the level that applies to a non-bundled JSON model:

1. **`format_version` gate FIRST.** Missing or `!= 1` (unknown/newer) → **raise `ValueError`**. Never silently best-effort-parse. This is the forward-compat door: a future v2 schema is rejected loudly by a v1 reader.
2. **Structural completeness.** A missing required key (any array, or `l`/`w`/`eps`/`method`) → raise `KeyError`/`ValueError`. `from_dict` does NOT fabricate a default for an absent fitted array.
3. **`params` validity.** Rebuild + `validate_params_for_method` (raises `TypeError` on a method/params mismatch — reuses the existing guard).
4. **Post-reconstruction fitted check.** The reconstructed model must pass `require_fitted_xt` (i.e. `np.any(xT)`); an all-zero `xT` payload → the model is unfitted and the round-trip guard is non-vacuous (§6.6).
5. **No `pickle`.** JSON-safe primitives only (consumer policy forbids `pickle.loads`). `to_dict` emits only `int`/`float`/`str`/`bool`/`None`/`list`/`dict`.

## 6. TDD — red-green, required (write failing first)

All in `tests/xthreat/test_serialize.py` (new). Each covers both `method`s where relevant.

1. **Round-trip fidelity:** `xt2 = ExpectedThreat.from_dict(xt.to_dict())` → `np.array_equal` on `xT` + the 4 prob matrices + each `heatmaps` entry; equal `l`/`w`/`eps`/`method`/`params`; `xt2.grid == xt.grid`; dtypes + shapes preserved.
2. **Functional equivalence:** `np.allclose(xt.rate(actions), xt2.rate(actions))` AND `destination_profiles(xt2, xs, ys)` equals `destination_profiles(xt, xs, ys)` on all three fields (`zone_centres`/`zone_values`/`probabilities`). (This is where a silent orientation flip would show up.)
3. **Unfitted raises:** `ExpectedThreat().to_dict()` → `NotFittedError`.
4. **Both methods:** `singh_counts` + `kde_smoothed` (proves `params` round-trips, incl. `KDEParams.kernel`).
5. **JSON boundary:** `ExpectedThreat.from_dict(json.loads(json.dumps(xt.to_dict())))` passes tests 1+2 — proves no non-JSON type leaks.
6. **Non-vacuity (fail-closed):** (a) a dict with `transition_matrix` dropped → `from_dict` raises; (b) an unknown `format_version` (e.g. `2`) → raises; (c) an all-zero `xT` payload → `from_dict` raises (or the model fails `require_fitted_xt`). Proves the round-trip guard cannot pass vacuously.
7. **`save`/`load` file round-trip:** `xt.save(tmp); ExpectedThreat.load(tmp)` passes tests 1+2; `load` of a JSON file with a bad `format_version` raises (wrapper inherits the fail-closed contract).

## 7. Decisions (for the reviewer to confirm/challenge)

- **D1 — `to_dict`/`from_dict` are the primitive; `save`/`load` are thin JSON-file wrappers.** Single serialization source; no duplicated field list. (Owner: gold-standard → include the wrappers.)
- **D2 — JSON, not pickle.** Consumer policy forbids `pickle.loads`; JSON is inspectable + cross-language.
- **D3 — Raw-orientation verbatim round-trip (ADR-041).** No normalization on either leg. Correctness invariant.
- **D4 — `format_version` forward-compat, fail-closed on unknown.** A newer schema is rejected loudly by an older reader.
- **D5 — NO SHA256SUMS sidecar / chirality / feature-contract on `save`/`load`.** sk's SHA/chirality machinery (DependenceModel, GhostGk, GkCompletion) exists to detect tampering of **wheel-bundled** artifacts loaded by pip users. `ExpectedThreat` is **consumer-fitted and consumer-persisted** (to HF Hub, with the consumer's own integrity); there is no sk bundle to tamper. The `format_version` gate + structural completeness + `require_fitted_xt` are the integrity contract. Adding a SHA sidecar would be cargo-culting a bundled-artifact control onto a non-bundled model. **Reviewer: challenge if the consumer wants sk-side integrity instead of owning it.**
- **D6 — `from_dict` bypasses `fit`.** Reconstruction sets state directly (no re-computation), so a consumer never pays the fit cost twice and gets a bit-identical model.

## 8. Ship checklist

- Spec (this) → lakehouse review → plan → lakehouse review → implement → unbiased impl review (all reports to `D:\Development\_reviews\`).
- Branch off `origin/main` (4.120.0); `feat/expectedthreat-serialize-seam`.
- `CHANGELOG.md` `Added`; **ADR-NNN** (records D1–D6); `_version.py` → `<next-free version>` (single line); `uv lock` follows; annotated tag on merge (owner).
- `NOTICE`: no new citation (Singh xT already cited; SK-xT-1/ADR-021).
- Additive — **no VAEP/tracking retrain, no re-materialize, C4-free** (methods on an existing class, not a new aggregator; aggregator count 33 unchanged, glossary unchanged).
- Kept on its own cycle — not entangled with any in-flight tracking work.

## 9. Non-goals

- No bundled xT (silly-kicks still ships none).
- No change to `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles` behaviour or the SK-xT-1 parity oracle.
- No lakehouse-side code (handoff §8 is consumer context only).
