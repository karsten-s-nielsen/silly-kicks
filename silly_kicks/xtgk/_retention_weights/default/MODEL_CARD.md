# GK retention (rho) model — variant `default`

P(retain | GK distribution) for xT-GK v2 (ADR-036 §Part 3). Logistic, pure-numpy serve.

- Provider: **gradientsports** (64 matches, 396 GK-distribution actions)
- Label: `retains(window_seconds=10)` (truncated windows excluded)
- Marts-native 8 features (geometry + `pressure_on_actor__andrienko_oval`); tracking-frames deprecated so the frames-only receiver-density feature is absent
- Out-of-fold (GroupKFold by match): **AUC 0.776**, **ECE 0.090**, reliability slope 1.01
- Calibration gate (ECE<=0.10 AND |slope-1|<=0.25): **PASS**
