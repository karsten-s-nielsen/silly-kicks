# xCross causal validation (TF-17 PR-C)

- Opportunities: 52978 (4193 treated; base Y=0.049)
- ATT without GK: 0.0079 (SE 0.0050)
- ATT with GK:    0.0019 (SE 0.0051)
- GK ablation shift: -0.0060; placebo band p95 cluster (GATE): 0.0091; row (4.18.0-comparable): 0.0096
- **GK clears placebo band: False** (reported, not a gate; cluster band)
- NaN fraction GK/base: 0.000/0.000; PS overlap: 1.000
- SMD max pre/post: 0.468 / 0.038; **claim supported: True**
- Caveat: state-vs-sender + tracking-only opportunity detection; league/era differ from paper. Common support = treated-within-control-PS-range (no density trimming). Treated/control Y-windows are time-shifted (treated anchored at t_cross, control at entry). Z is a same-team cross within T of entry, clamped to possession continuity.
