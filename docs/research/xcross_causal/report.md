# xCross causal validation (TF-17 PR-C)

- Opportunities: 23966 (669 treated; base Y=0.043)
- ATT without GK: 0.0747 (SE 0.0167)
- ATT with GK:    0.0927 (SE 0.0156)
- GK ablation shift: 0.0179; placebo band p95: 0.0239
- **GK clears placebo band: False** (reported, not a gate)
- NaN fraction GK/base: 0.000/0.000; PS overlap: 1.000
- SMD max pre/post: 0.511 / 0.078; **claim supported: True**
- Caveat: state-vs-sender + tracking-only opportunity detection; league/era differ from paper. Common support = treated-within-control-PS-range (no density trimming). Treated/control Y-windows are time-shifted (treated anchored at t_cross, control at entry). Z is a same-team cross within T of entry, clamped to possession continuity.
