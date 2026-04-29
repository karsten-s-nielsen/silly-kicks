# Kloppy test fixtures

These files are vendored from the [kloppy](https://github.com/PySport/kloppy) test suite for use as test fixtures by `tests/spadl/test_kloppy.py`.

## Files

| File | Origin | License |
|---|---|---|
| `sportec_events.xml` | [PySport/kloppy `kloppy/tests/files/sportec_events.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/sportec_events.xml) | BSD-3-Clause (kloppy synthetic test data) |
| `sportec_meta.xml` | [PySport/kloppy `kloppy/tests/files/sportec_meta.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/sportec_meta.xml) | BSD-3-Clause (kloppy synthetic test data) |
| `metrica_events.json` | [PySport/kloppy `kloppy/tests/files/metrica_events.json`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/metrica_events.json) | Originally from [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data) Sample Game 2 — **CC-BY-NC-4.0** (test-data-only; not redistributed in the wheel) |
| `epts_metrica_metadata.xml` | [PySport/kloppy `kloppy/tests/files/epts_metrica_metadata.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/epts_metrica_metadata.xml) | BSD-3-Clause (kloppy synthetic test data) |

## Notes

- The Metrica events JSON and the EPTS metadata are from **different matches**. This pairing matches kloppy's own test fixture pairing (see `kloppy/tests/test_metrica_events.py`'s FIXME) and is sufficient for converter contract testing — it exercises the parsing pipeline and event-type mapping without depending on perfect alignment between event coordinates and player metadata.
- These fixtures are **test-data only**. They are excluded from the published `silly-kicks` wheel via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in `pyproject.toml`, which packages only the `silly_kicks/` source tree.
- License compliance: kloppy is BSD-3-Clause (attribution preserved by linking back). The Metrica file's CC-BY-NC license restricts commercial **redistribution**; using it for testing in a non-commercial open-source library is permitted, and we do not redistribute it in the published wheel.
