# StatsBomb open-data fixtures

Vendored from https://github.com/statsbomb/open-data under the StatsBomb
Public Data License (non-commercial). See
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf for the
full license text.

Used for offline e2e validation of silly-kicks's SPADL converters and
post-conversion enrichments. The three matches under `raw/events/` are
the same ones measured during the luxury-lakehouse PR-LL2 boundary-
metrics campaign that produced the empirical baselines published in
`silly_kicks.spadl.add_possessions`'s docstring:

- `7298.json` — Women's World Cup
- `7584.json` — Champions League
- `3754058.json` — Premier League

## `three-sixty/` — SB360 freeze-frame slice

Golden-parity fixture for `silly_kicks.providers.statsbomb`, added with the
parse port. **Women's World Cup 2023, match 3893795**, reduced to 6 freeze-frames
(including a `Goal Keeper` event, ~19 players each) plus the 6 events they join
to. Byte digests are pinned in `SOURCE_SHA` and asserted by
`tests/providers/statsbomb/test_parse_golden.py`, which reads the slice with
stdlib `json` — never `statsbombpy`, so the gate cannot skip where that optional
package is absent.

Reduced rather than whole-match: the port's contract is per-record, so 6 records
exercise it, and a full 3418-frame payload would be ~10 MB of fixture for no
extra coverage.

License compliance: this is non-commercial use; redistribution is
permitted under the same license.
