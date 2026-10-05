# DAS native-engine corpus parity + performance (ADR-107/108)

Owner-run DGX artifacts for the native DAS engine, produced at the release commit
(`run_commit` in `metrics.json` / `performance.json`; combined-cycle C2, 4.128.0). Aggregates only —
no owner-tier rows (ADR-038); `tests/test_research_artifacts_carry_no_ids.py` guards that.

## Corpus bound

The full pining corpus at the owner token: **980 matches** — gradientsports 64, idsse 7,
skillcorner 909. 14 skillcorner entries are velocity-less SB360 freeze-frames (ADR-063),
structurally unscoreable by DAS; they are excluded and recorded in `n_excluded`, never silently
dropped (895 skillcorner matches scored). The
native numpy/numba engines reproduce the pinned `accessible-space==2.0.15` reference within the
golden bounds (`np.allclose`, rtol=atol 1e-12 numpy / 1e-10 numba) on every scored `reason==OK` row,
with zero finite-mask mismatches (`metrics.json`).

## D-KEY

`d_key_frames` = **0 of 980 matches** reuse a `frame_id` across periods, so the collision-free frame
key changed no value on this corpus. The keying code path is exercised by
`tests/tracking/test_das_pack.py` (D-KEY) and `test_das_divergences.py`. Direction (GoalMap vs the
reference's own inference) agreed on all 902 184 compared frames (0 disagreements).

## Timings

`metrics.json`'s per-match `timings_ms_per_frame` are from the **contended** corpus map (many workers
at once) and are **NOT for ratios**. The speed record is `performance.json`: a dedicated `--benchmark`
run, alone on the box, warm-up + best-of-3 per leg, with a `contention.foreign_cpu_fraction` gate.
Headline (`performance.json` `summary`): reference / numba-serial 15.3×, reference / numpy 5.05×;
`prange` efficiency 0.576 at 16 threads; `add_das` 6.69× and `das_xfns` 120× over the finished pairs
(the old `das_xfns` path exceeds the 100 GiB ceiling on gradientsports and idsse — recorded in
`old_path_over_memory`, so `das_xfns_speedup` rests on the 3 skillcorner matches that fit).
