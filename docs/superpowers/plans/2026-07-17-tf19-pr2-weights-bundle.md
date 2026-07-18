# TF-19 PR-2 Weights Bundle + Chirality Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **STOP at the final commit — it is user-gated (present + WAIT; never commit without explicit approval; never create/offer the commit sentinel).**

**Goal:** Ship the deferred TF-19 PR-2: bundle the DGX-retrained (chirality-corrected) default xS/xCross/ghost weights, add fail-closed chirality enforcement to `load()` across all three models, implement the xS `from_hub` + route the HF-only `sc_extended` variant, fix the ghost model-card prose, and record the decision-table verdict.

**Architecture:** The currently-bundled `default` weights are chirality-mis-served (trained y-mirrored pre-ADR-031, served y-correct) — a live correctness bug for `pre_shot_gk_full_default_xfns` consumers. This PR replaces them with the corrected retrains (which carry chirality fingerprints) and adds `load()` enforcement that re-runs each model's own `_chirality_block` and compares to the stored fingerprint, raising on a mismatch AND on a missing one (every pre-PR-2 artifact = exactly the mis-served ones), with an explicit `legacy_override` escape hatch. `sc_extended` (trained on the 98 owner SkillCorner matches) is HF-only.

**Tech Stack:** Python, xgboost (xS/xCross boosters), numpy (ghost npz), huggingface_hub (Hub download), pytest.

**Silly-kicks conventions to honor:** `warnings.warn(..., stacklevel=2)`; pickle-free serialization (already in place); SHA256SUMS CRLF→LF normalization; version at 3 code sites (pyproject.toml, silly_kicks/__init__.py, uv.lock) + CHANGELOG; CI = full-tree pyright + ruff + `pytest -m "not e2e"`.

**Verified weight artifacts** staged at `C:\Users\Karsten\AppData\Local\Temp\claude\D--Development-karstenskyt--silly-kicks-part-deux\e14c809d-84c7-4487-992d-d7b587dcaed0\scratchpad\weights\` (subdirs: `xs_default`, `xcross_default`, `ghost_default`, `xs_sc_extended`, `xcross_sc_extended`; each with model.json/rfcde_weights.npz + metadata.json + SHA256SUMS + metrics.json; all carry `chirality` version `chirality-probe-1`, frame_sha `60ac605a…`, valid SHA256SUMS).

**KEY RISK (validated by Task 9's golden test):** the chirality fingerprint was computed on the DGX (aarch64) at save-time; `load()` re-verifies on the user's platform (x86). Output comparison MUST tolerate cross-platform float noise (~1e-6) while catching a y-mirror (gross, O(0.01–1)). If the golden test fails, the tolerance is wrong OR xgboost/numpy predict isn't cross-platform-reproducible — STOP and escalate, do not just loosen until green.

---

## File Structure

- **Modify** `silly_kicks/tracking/_chirality.py` — add `np.isfinite` guard to `chirality_fingerprint`; add the shared `verify_chirality(recomputed, stored, *, legacy_override, model_name)` enforcement helper.
- **Modify** `silly_kicks/tracking/_xshot_occurrence.py` — `load()` gains `legacy_override` + calls `verify_chirality`; `from_hub` implemented; `from_variant` routes `sc_extended`→`from_hub`.
- **Modify** `silly_kicks/tracking/_xcross_attempt.py` — same `load()` enforcement + `from_variant` `sc_extended`→`from_hub` (its `from_hub` already works).
- **Modify** `silly_kicks/tracking/_ghost_gk.py` — `load()` gains `legacy_override` + `verify_chirality`; fix model-card prose in `save()` metadata / MODEL_CARD.
- **Replace** `silly_kicks/tracking/_xshot_weights/default/`, `_xcross_weights/default/`, `_ghost_gk_weights/default/` — corrected retrain artifacts.
- **Create** `tests/tracking/test_chirality_enforcement.py` — load() raises on mismatch + missing (all 3), legacy_override loads, isfinite guard.
- **Create/extend** `tests/tracking/test_weights_bundle_golden.py` — bundled default weights load + chirality re-verifies on THIS platform.
- **Modify** xS/xCross `from_variant` sc_extended routing test in the enforcement test file.
- **Create** `docs/research/tf19_pr2/decision_table.md` — recorded verdict + probe availability.
- **Create** `docs/research/tf19_pr2/hf_upload_instructions.md` — owner deliverable for the sc_extended + full-ghost Hub upload.
- **Modify** `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`; **create** `docs/superpowers/adrs/ADR-040-*.md` OR amend ADR-037 (Task 12 decides).

---

### Task 1: `chirality_fingerprint` finiteness guard + shared `verify_chirality` helper

**Files:**
- Modify: `silly_kicks/tracking/_chirality.py`
- Test: `tests/tracking/test_chirality_enforcement.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_chirality_enforcement.py
import numpy as np
import pytest
from silly_kicks.tracking._chirality import chirality_fingerprint, verify_chirality
from silly_kicks.tracking._xshot_occurrence import IntegrityError


def test_chirality_fingerprint_raises_on_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        chirality_fingerprint(lambda frame: np.array([np.nan, 0.5]))


def test_verify_chirality_passes_identical():
    fp = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5, 0.25]}
    verify_chirality(fp, dict(fp), legacy_override=False, model_name="xS")  # no raise


def test_verify_chirality_tolerates_float_noise():
    stored = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5000000, 0.2500000]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5000004, 0.2499997]}
    verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")  # no raise (< tol)


def test_verify_chirality_raises_on_mirror_scale_mismatch():
    stored = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.80, 0.20]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.20, 0.80]}
    with pytest.raises(IntegrityError, match="chirality"):
        verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")


def test_verify_chirality_raises_on_missing_stored():
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5]}
    with pytest.raises(IntegrityError, match="missing"):
        verify_chirality(recomputed, None, legacy_override=False, model_name="xS")


def test_verify_chirality_legacy_override_allows_missing():
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "abc", "outputs": [0.5]}
    with pytest.warns(UserWarning, match="legacy"):
        verify_chirality(recomputed, None, legacy_override=True, model_name="xS")  # no raise


def test_verify_chirality_raises_on_frame_sha_change():
    stored = {"version": "chirality-probe-1", "frame_sha256": "OLDSHA", "outputs": [0.5]}
    recomputed = {"version": "chirality-probe-1", "frame_sha256": "NEWSHA", "outputs": [0.5]}
    with pytest.raises(IntegrityError, match="probe frame"):
        verify_chirality(recomputed, stored, legacy_override=False, model_name="xS")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_chirality_enforcement.py -x -q`
Expected: FAIL (`verify_chirality` not importable; `chirality_fingerprint` doesn't guard non-finite).

- [ ] **Step 3: Implement in `_chirality.py`**

Add the finiteness guard inside `chirality_fingerprint` (after `outputs = np.asarray(...).ravel()`, before the return):

```python
    if not np.all(np.isfinite(outputs)):
        raise ValueError(f"chirality fingerprint produced non-finite outputs: {outputs!r}")
```

Add the shared enforcement helper (import `IntegrityError` lazily to avoid a cycle — it lives in `_xshot_occurrence`; use a local import inside the function). Tolerance: `atol=1e-3, rtol=1e-2` — a y-mirror on the deliberately-asymmetric probe frame moves outputs by O(0.01–1); cross-platform xgboost/numpy float noise is ~1e-6. `frame_sha256` and `version` are frame/code-derived (platform-independent) → exact match.

```python
# Tolerance catches a y-mirror (gross) but tolerates cross-platform float noise (~1e-6).
# See ADR-037 §9 + the 2026-07-17-tf19-pr2 plan's KEY RISK note.
_CHIRALITY_ATOL = 1e-3
_CHIRALITY_RTOL = 1e-2


def verify_chirality(recomputed: dict, stored: dict | None, *, legacy_override: bool, model_name: str) -> None:
    """Fail-closed chirality check at load() (ADR-037 §9, TF-19 PR-2).

    ``recomputed`` = ``chirality_fingerprint`` re-run on the just-loaded model.
    ``stored`` = the ``chirality`` block from the artifact's metadata.json (``None`` if absent).
    Raises ``IntegrityError`` on a MISSING fingerprint (every pre-PR-2 artifact = the mis-served
    ones) unless ``legacy_override``; raises on a probe-frame change or an output mismatch beyond
    the cross-platform tolerance.
    """
    from silly_kicks.tracking._xshot_occurrence import IntegrityError

    if stored is None:
        if legacy_override:
            warnings.warn(
                f"{model_name}: loading a weights artifact with NO chirality fingerprint under "
                "legacy_override=True. Every pre-TF-19-PR-2 artifact is y-mirror-mis-served; only "
                "override for an artifact you have independently verified.",
                stacklevel=2,
            )
            return
        raise IntegrityError(
            f"{model_name}: weights artifact has NO chirality fingerprint. Every pre-TF-19-PR-2 "
            "artifact is the y-mirror-mis-served class of bug (ADR-037). Refusing to load; pass "
            "legacy_override=True only if independently verified."
        )
    if recomputed.get("frame_sha256") != stored.get("frame_sha256"):
        raise IntegrityError(
            f"{model_name}: chirality probe frame changed (stored {stored.get('frame_sha256','')[:8]} "
            f"vs library {recomputed.get('frame_sha256','')[:8]}). Version skew; refusing to load."
        )
    a = np.asarray(recomputed.get("outputs", []), dtype=float)
    b = np.asarray(stored.get("outputs", []), dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=_CHIRALITY_ATOL, rtol=_CHIRALITY_RTOL):
        raise IntegrityError(
            f"{model_name}: chirality mismatch — served outputs {a.tolist()} do not match the "
            f"trained fingerprint {b.tolist()} within tol (atol={_CHIRALITY_ATOL}). This is the "
            "y-mirror-mis-serving signature; refusing to load."
        )
```

Add `import warnings` at the top of `_chirality.py` if absent.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_chirality_enforcement.py -q`
Expected: 7 passed.

- [ ] **Step 5: Stage (no commit — final commit is user-gated at Task 13)**

Run: `git add silly_kicks/tracking/_chirality.py tests/tracking/test_chirality_enforcement.py`

---

### Task 2: xS `load()` chirality enforcement + `legacy_override`

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py:468-522` (the `load` classmethod)
- Test: extend `tests/tracking/test_chirality_enforcement.py`

- [ ] **Step 1: Write the failing test** (a round-trip: save a model, tamper the metadata chirality, assert load raises; and the missing-fingerprint case)

```python
# append to tests/tracking/test_chirality_enforcement.py
import json
from pathlib import Path
import pandas as pd


def _tiny_xshot_model():
    # Minimal fitted xS model for round-trip. Reuse the project's fixture pattern.
    from tests.tracking.test_xshot_occurrence import _make_fitted_xshot_model  # existing helper
    return _make_fitted_xshot_model()


def test_xshot_load_raises_on_tampered_chirality(tmp_path):
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, IntegrityError
    m = _tiny_xshot_model()
    m.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["chirality"]["outputs"] = [v + 5.0 for v in meta["chirality"]["outputs"]]  # gross mirror-scale
    (tmp_path / "metadata.json").write_text(json.dumps(meta), newline="\n")
    # re-sum so the SHA guard doesn't fire first
    import hashlib
    with open(tmp_path / "SHA256SUMS", "w", newline="\n") as f:
        for fn in ["model.json", "metadata.json"]:
            raw = (tmp_path / fn).read_bytes().replace(b"\r\n", b"\n") if fn.endswith(".json") else (tmp_path / fn).read_bytes()
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")
    with pytest.raises(IntegrityError, match="chirality"):
        XShotOccurrenceModel.load(tmp_path)


def test_xshot_load_raises_on_missing_chirality(tmp_path):
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel, IntegrityError
    m = _tiny_xshot_model()
    m.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    del meta["chirality"]
    (tmp_path / "metadata.json").write_text(json.dumps(meta), newline="\n")
    import hashlib
    with open(tmp_path / "SHA256SUMS", "w", newline="\n") as f:
        for fn in ["model.json", "metadata.json"]:
            raw = (tmp_path / fn).read_bytes().replace(b"\r\n", b"\n") if fn.endswith(".json") else (tmp_path / fn).read_bytes()
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")
    with pytest.raises(IntegrityError, match="chirality"):
        XShotOccurrenceModel.load(tmp_path)
    XShotOccurrenceModel.load(tmp_path, legacy_override=True)  # override loads
```

(If `_make_fitted_xshot_model` doesn't exist, the implementer must locate the existing xS test fixture in `tests/tracking/test_xshot_occurrence*.py` and reuse it; do NOT invent a new training path.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_chirality_enforcement.py -k xshot_load -q`
Expected: FAIL (load has no chirality check; `legacy_override` kwarg unknown).

- [ ] **Step 3: Implement** — change the `load` signature to `def load(cls, path: Path, *, legacy_override: bool = False)` and, AFTER `model._booster = booster` (line 521, before `return model`), add:

```python
        from silly_kicks.tracking._chirality import verify_chirality
        from silly_kicks.tracking._xshot_occurrence import _chirality_block  # same module; direct call

        verify_chirality(
            _chirality_block(model), meta.get("chirality"),
            legacy_override=legacy_override, model_name="xShotOccurrence",
        )
```

(`_chirality_block` is a module-level function in the same file — call it directly, no import needed; the import line above is illustrative — use the bare name.) `from_variant` and `from_hub` call `cls.load(dir)` with the default `legacy_override=False` — no change needed there for strictness.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_chirality_enforcement.py -k xshot_load -q`
Expected: PASS.

- [ ] **Step 5: Stage**

Run: `git add silly_kicks/tracking/_xshot_occurrence.py tests/tracking/test_chirality_enforcement.py`

---

### Task 3: xCross `load()` chirality enforcement + `legacy_override`

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py` (its `load` classmethod — same shape as xS)
- Test: extend `tests/tracking/test_chirality_enforcement.py` (mirror Task 2 with the xCross fixture)

- [ ] **Step 1: Write the failing test** — mirror `test_xshot_load_raises_on_tampered_chirality` / `_missing_` for `XCrossAttemptModel`, reusing the existing xCross fitted-model fixture from `tests/tracking/test_xcross_attempt*.py`.

- [ ] **Step 2: Run to verify failure** — `pytest -k xcross_load` FAILS.

- [ ] **Step 3: Implement** — identical change: `load(cls, path, *, legacy_override=False)` + after the booster loads and model fields are set, call `verify_chirality(_chirality_block(model), meta.get("chirality"), legacy_override=legacy_override, model_name="xCrossAttempt")`. Reuse the same import pattern.

- [ ] **Step 4: Run to verify pass** — `pytest -k xcross_load` PASSES.

- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_chirality_enforcement.py`

---

### Task 4: ghost-GK `load()` chirality enforcement + `legacy_override`

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:1798-1911` (the `load` classmethod — already has serve_estimator + npz fail-closed checks)
- Test: extend `tests/tracking/test_chirality_enforcement.py` (mirror with the ghost fixture)

- [ ] **Step 1: Write the failing test** — mirror the tamper + missing cases for `GhostGkModel`, reusing the existing ghost fitted-model fixture from `tests/tracking/test_ghost_gk.py` (e.g. a small `fit()`ed model). Ghost outputs are 2 coordinates (~0–105); a mirror flips gk_y (34±) → gross.

- [ ] **Step 2: Run to verify failure** — `pytest -k ghost_load_...chirality` FAILS.

- [ ] **Step 3: Implement** — add `legacy_override: bool = False` to `load`'s signature; after the model is fully reconstructed (baselines + trees + metadata fields set, just before `return model`), call `verify_chirality(_chirality_block(model), meta.get("chirality"), legacy_override=legacy_override, model_name="GhostGk")`. Follow the existing fail-closed guard style in that method.

- [ ] **Step 4: Run to verify pass** — `pytest -k ghost_load` PASSES.

- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_chirality_enforcement.py`

---

### Task 5: xS `from_hub` implementation + `sc_extended`→Hub routing (xS + xCross)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py:551-561` (from_hub) + `:524-549` (from_variant)
- Modify: `silly_kicks/tracking/_xcross_attempt.py:553-572` (from_variant)
- Test: extend `tests/tracking/test_chirality_enforcement.py`

- [ ] **Step 1: Write the failing test** (route `sc_extended` to `from_hub`; since the Hub repos don't exist, assert it ATTEMPTS the hub path — i.e. raises the hub-not-found / import error, NOT the "no bundled variant" FileNotFoundError):

```python
def test_xshot_sc_extended_routes_to_hub(monkeypatch):
    from silly_kicks.tracking import _xshot_occurrence as X
    called = {}
    def fake_from_hub(repo_id=X._HF_REPO_ID):
        called["repo"] = repo_id
        raise FileNotFoundError("hub attempted")
    monkeypatch.setattr(X.XShotOccurrenceModel, "from_hub", classmethod(lambda cls, repo_id=X._HF_REPO_ID: fake_from_hub(repo_id)))
    X._VARIANT_CACHE.pop("sc_extended", None)
    with pytest.raises(FileNotFoundError, match="hub attempted"):
        X.XShotOccurrenceModel.from_variant("sc_extended")
    assert called["repo"] == "silly-kicks/xshot-occurrence-v1"
```

Mirror for xCross (`test_xcross_sc_extended_routes_to_hub`).

- [ ] **Step 2: Run to verify failure** — `pytest -k sc_extended_routes` FAILS (sc_extended currently hits the `else: raise FileNotFoundError("No bundled...")`).

- [ ] **Step 3: Implement** —

In BOTH `from_variant` methods, change the routing condition from `elif variant == "public":` to also catch `sc_extended`:
```python
        elif variant in ("public", "sc_extended"):  # HF-only variants
            model = cls.from_hub(_HF_REPO_ID)
```
Implement the xS `from_hub` by mirroring xCross's working version (replace the inert stub at `_xshot_occurrence.py:551-561`):
```python
    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> XShotOccurrenceModel:
        """Download published weights from HuggingFace Hub and load. Requires ``pip install silly-kicks[xshot]``."""
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("xShotOccurrence Hub weights require: pip install silly-kicks[xshot]") from None
        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))
```
Confirm `[xshot]` extra exists in `pyproject.toml`; if only `[xgboost]` exists, use that extra name in the message (match the actual optional-dependency group — read pyproject first).

- [ ] **Step 4: Run to verify pass** — `pytest -k sc_extended_routes -q` PASSES.

- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xshot_occurrence.py silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_chirality_enforcement.py`

---

### Task 6: Bundle the corrected default weights (the live-bug fix)

**Files:**
- Replace: `silly_kicks/tracking/_xshot_weights/default/{model.json,metadata.json,SHA256SUMS}`
- Replace: `silly_kicks/tracking/_xcross_weights/default/{model.json,metadata.json,SHA256SUMS}`
- Replace: `silly_kicks/tracking/_ghost_gk_weights/default/{rfcde_weights.npz,metadata.json,SHA256SUMS}`

- [ ] **Step 1: Copy the retrain artifacts over the bundled defaults** (drop `metrics.json` — not read by loader, keep the dir minimal; but bundling it is harmless — match whatever the existing default dir contains: check `ls silly_kicks/tracking/_xshot_weights/default/` first and mirror that file set).

```bash
W="/c/Users/Karsten/AppData/Local/Temp/claude/D--Development-karstenskyt--silly-kicks-part-deux/e14c809d-84c7-4487-992d-d7b587dcaed0/scratchpad/weights"
cp "$W/xs_default/model.json"        silly_kicks/tracking/_xshot_weights/default/model.json
cp "$W/xs_default/metadata.json"     silly_kicks/tracking/_xshot_weights/default/metadata.json
cp "$W/xs_default/SHA256SUMS"        silly_kicks/tracking/_xshot_weights/default/SHA256SUMS
cp "$W/xcross_default/model.json"    silly_kicks/tracking/_xcross_weights/default/model.json
cp "$W/xcross_default/metadata.json" silly_kicks/tracking/_xcross_weights/default/metadata.json
cp "$W/xcross_default/SHA256SUMS"    silly_kicks/tracking/_xcross_weights/default/SHA256SUMS
cp "$W/ghost_default/rfcde_weights.npz" silly_kicks/tracking/_ghost_gk_weights/default/rfcde_weights.npz
cp "$W/ghost_default/metadata.json"     silly_kicks/tracking/_ghost_gk_weights/default/metadata.json
cp "$W/ghost_default/SHA256SUMS"        silly_kicks/tracking/_ghost_gk_weights/default/SHA256SUMS
```

- [ ] **Step 2: Verify the bundled SHA256SUMS match the copied files** (LF-normalized json):

```bash
.venv/Scripts/python.exe - <<'PY'
import hashlib, pathlib
for d in ["_xshot_weights","_xcross_weights","_ghost_gk_weights"]:
    p = pathlib.Path("silly_kicks/tracking")/d/"default"
    for line in (p/"SHA256SUMS").read_text().splitlines():
        exp, fn = line.split("  ",1)
        raw = (p/fn).read_bytes()
        if fn.endswith(".json"): raw = raw.replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest()==exp, f"{d}/{fn} SHA mismatch"
    print(d, "OK")
PY
```
Expected: three `OK` lines. (If mismatch: the scp may have introduced CRLF on json — re-copy with binary mode; the plan's Task 9 golden test also guards this.)

- [ ] **Step 3: Stage** — `git add silly_kicks/tracking/_xshot_weights/default silly_kicks/tracking/_xcross_weights/default silly_kicks/tracking/_ghost_gk_weights/default`

---

### Task 7: Ghost-GK model-card prose fix

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (the MODEL_CARD / metadata prose describing the training filter)

- [ ] **Step 1: Locate the prose** — `grep -n "active defensive" silly_kicks/tracking/_ghost_gk.py` (and any `MODEL_CARD` constant / docstring). The claim is the training filter is "GK outside penalty area during active defensive actions"; the code is a pure geometric box (goal-relative x∈[0,30], y∈[18,50], NO action condition).

- [ ] **Step 2: Write the corrected prose** — replace with an accurate description, e.g.: "training targets are GK positions in a fixed goal-relative box (x∈[0,30] m from the defended goal line, y∈[18,50] m), with NO action/possession condition — a purely geometric filter." Grep-confirm no other file repeats the wrong claim (`grep -rn "active defensive" silly_kicks/ docs/`).

- [ ] **Step 3: Stage** — `git add silly_kicks/tracking/_ghost_gk.py`

---

### Task 8: Decision-table verdict record

**Files:**
- Create: `docs/research/tf19_pr2/decision_table.md`

- [ ] **Step 1: Check whether the xS dose-banded probe is runnable now** — `grep -n "def evaluate_xs_probe\|def regate_verdict\|PROBE_WRAPPERS" silly_kicks/tracking/_model_eval.py`. Per ADR-037, the xS probe consumes the PR-3 gkdv engine; if the engine isn't importable, the xS probe is PR-3-gated (record that, do NOT stub it).

- [ ] **Step 2: Write the record** with what IS available from the completed DGX runs (do NOT fabricate): the Stage-B xCross frozen probe `tf19_ready=False` with GK-sub `gk_median 0.00970`, `ratio 2.21×` (clears the 2.0 prong), dose 4m/2m `2.36×` [use the Stage A number if that is the registered gated leg — read `sk_stageB_448/xcross/xcross_attempt_v1/metrics.json`'s probe block for the authoritative gated-vs-comparison values before writing]; the CROSS decision-table row via `regate_verdict` = `gated_clean_fail`; the SHOT row = pending the PR-3 xS probe. State that per ADR-037 the xS-probe row is PR-3-gated and this PR records the xCross verdict + the entanglement inputs only.

- [ ] **Step 3: Stage** — `git add docs/research/tf19_pr2/decision_table.md`

---

### Task 9: Golden test — bundled default weights load + chirality re-verifies on THIS platform

**Files:**
- Create: `tests/tracking/test_weights_bundle_golden.py`

- [ ] **Step 1: Write the test** (this VALIDATES the whole cross-platform enforcement — the KEY RISK):

```python
import pytest


@pytest.mark.parametrize("loader", [
    ("silly_kicks.tracking._xshot_occurrence", "XShotOccurrenceModel"),
    ("silly_kicks.tracking._xcross_attempt", "XCrossAttemptModel"),
    ("silly_kicks.tracking._ghost_gk", "GhostGkModel"),
])
def test_bundled_default_loads_and_chirality_reverifies(loader):
    import importlib
    mod, cls_name = loader
    cls = getattr(importlib.import_module(mod), cls_name)
    # from_variant("default") loads the bundled dir AND runs verify_chirality with the STRICT
    # (non-override) path — so a passing load == the DGX-trained fingerprint reproduces on THIS
    # (x86) platform within tolerance. This is the cross-platform-determinism gate.
    m = cls.from_variant("default")
    assert m is not None
```

- [ ] **Step 2: Run**

Run: `.venv\Scripts\python.exe -m pytest tests/tracking/test_weights_bundle_golden.py -q`
Expected: 3 passed. **If it FAILS with a chirality mismatch, STOP** — the DGX (aarch64) fingerprint does not reproduce on x86 within `_CHIRALITY_ATOL`. Do NOT blindly widen the tolerance; escalate to the user with the actual output deltas (a genuine y-mirror is O(0.01–1); if the delta is ~1e-5 the tolerance is merely too tight, if it's ~0.1 something is wrong). This is the make-or-break validation of the enforcement design.

- [ ] **Step 3: Stage** — `git add tests/tracking/test_weights_bundle_golden.py`

---

### Task 10: HF-upload deliverable (owner follow-up)

**Files:**
- Create: `docs/research/tf19_pr2/hf_upload_instructions.md`

- [ ] **Step 1: Write the instructions** — the sc_extended xS/xCross + full-ghost weights go to HF; the repos `silly-kicks/xshot-occurrence-v1`, `silly-kicks/xcross-attempt-v1`, `silly-kicks/ghost-gk-v1` must be CREATED (they don't exist), and the artifacts uploaded, by the owner (needs HF auth). Document: the exact local staging paths of the artifacts (`scratchpad/weights/xs_sc_extended/`, `xcross_sc_extended/`, and the box's `sk_stageB_448/ghost_full/` for the full ghost), the `HfApi.create_repo` + `upload_folder` calls (exclude any `_feature_cache/`), the repo_ids that `from_hub` points at, and a post-upload verification (`from_variant("sc_extended")` resolves + its chirality re-verifies). Note this is NOT executed in this PR (no HF auth in the code path).

- [ ] **Step 2: Stage** — `git add docs/research/tf19_pr2/hf_upload_instructions.md`

---

### Task 11: Version bump + CHANGELOG

**Files:**
- Modify: `pyproject.toml:7`, `silly_kicks/__init__.py:7`, `uv.lock`, `CHANGELOG.md`

- [ ] **Step 1: Bump to 4.51.0** at `pyproject.toml` and `silly_kicks/__init__.py`; regenerate `uv.lock` via `uv lock`; verify all three read 4.51.0.

- [ ] **Step 2: Add a `## [4.51.0]` CHANGELOG section** — PR-S118, TF-19 PR-2: the chirality-mis-serving fix (corrected default weights + `load()` fail-closed enforcement + legacy_override + finiteness), the HF-only `sc_extended` routing (Hub upload = owner follow-up), the ghost model-card fix, the decision-table verdict. **Flag the Hyrum/retrain trigger:** `pre_shot_gk_full_default_xfns` consumers' xS/xCross columns change (the corrected weights) → retrain trigger for opted-in VAEP consumers; the public arm is GS-free so unaffected by the 4.49/4.50 GS fixes. Note the wheel SHRINKS (ghost 12M→7.2M).

- [ ] **Step 3: Stage** — `git add pyproject.toml silly_kicks/__init__.py uv.lock CHANGELOG.md`

---

### Task 12: ADR (chirality-enforcement contract)

**Files:**
- Create `docs/superpowers/adrs/ADR-040-chirality-load-enforcement.md` OR amend ADR-037.

- [ ] **Step 1: Decide** — the fail-closed `load()` chirality contract (raise on missing = every pre-PR-2 artifact, + legacy_override, + cross-platform tolerance) is a new cross-cutting consumer/serialization contract. ADR-037 already owns the chirality *design*; PR-2 delivers the enforcement. Prefer an **ADR-037 amendment** (the enforcement is the completion of an existing decision), unless the operator judges the cross-platform-tolerance + legacy-override policy warrants its own ADR-040. Record: the missing-fingerprint-fails-closed rule, the legacy_override escape hatch, the `atol/rtol` tolerance rationale (cross-platform aarch64→x86), and that this is CI-gated by the golden test.

- [ ] **Step 2: Stage** — `git add docs/superpowers/adrs/`

---

### Task 13: Full gate + final review + present commit (STOP)

- [ ] **Step 1: Full gate** — `ruff check` + `ruff format --check` (whole tree) + `pyright silly_kicks/ scripts/ tests/` (0 errors) + `pytest -m "not e2e and not slow" --benchmark-skip -q`. All green.
- [ ] **Step 2: Confirm C4-free** — no new aggregator/backend/model *class* (new bundled weights + a load-enforcement helper are not a new C4 model node; the enumerated count stays 29). Confirm.
- [ ] **Step 3: Run `/final-review`** on the staged tree.
- [ ] **Step 4: Present the commit** — the staged diff + a drafted commit message (`fix(tracking)!: TF-19 PR-2 — corrected default weights + chirality load() enforcement + sc_extended HF routing -- silly-kicks 4.51.0`) to the USER and **WAIT** for explicit approval. Do NOT commit. Do NOT create/offer the sentinel. The HF upload (Task 10 doc) is a separate owner step after merge.

---

## Self-Review

**Spec coverage:** (1) default weights bundle → Task 6; (2) load() enforcement all 3 → Tasks 2/3/4; (3) isfinite → Task 1; (4) xS from_hub + sc_extended routing → Task 5; (5) ghost model-card → Task 7; (6) decision-table → Task 8; (7) tests (mismatch+missing+override+routing+golden) → Tasks 1–5, 9; (8) version+CHANGELOG+ADR → Tasks 11/12. HF-upload deliverable → Task 10. All covered.

**Cross-platform risk** is isolated to Task 9's golden test with an explicit STOP-and-escalate — the single most likely failure and the one place a subagent must NOT paper over.

**Type/name consistency:** `verify_chirality(recomputed, stored, *, legacy_override, model_name)` is defined in Task 1 and called identically in Tasks 2/3/4; `_chirality_block` is the existing per-model helper; `IntegrityError` is the existing xS exception reused by xCross (imported lazily in `_chirality.py` to avoid a cycle).
