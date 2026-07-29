# Shot-arm (xS) causal validation (TF-19, ADR-037)

- Opportunities: 98789 (2143 treated; base Y=0.010)
- Control Y (R10): 730/96646 conversions (rate 0.0076; floor 30)
- ATT without GK: 0.0952 (SE 0.0070)
- ATT with GK:    0.0882 (SE 0.0074)
- GK ablation shift: -0.0070; placebo band p95 cluster (GATE): 0.0047; row (comparability): 0.0047
- **Entanglement verdict: inside_band** (supportive context, not causal deterrence)
- NaN fraction GK/base: 0.000/0.007; PS overlap: 1.000
- SMD max pre/post: 0.881 / 0.055; **claim supported: True**
- Caveat: GK-confounder ENTANGLEMENT, not causal deterrence: the shot arm measures whether the GK block carries Z/Y-aligned signal beyond the xS positional confounders. Anchor-INCLUSIVE 6 s success-shot outcome (second re-registration, P1); treated/control Y-windows are time-shifted (treated anchored at the in-spell shot, control at entry). The cluster placebo band gates (match-level exchangeability); the row band is reported for comparability.
