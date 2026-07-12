# GK retention (rho) model — variant `skillcorner`

P(retain | GK distribution) for xT-GK v2 (ADR-036 §Part 3). Logistic, pure-numpy serve.

- Provider: **skillcorner** (108 matches, 5477 GK-distribution actions)
- Label: `retains(window_seconds=10)` (truncated windows excluded)
- Marts-native 8 features (geometry + `pressure_on_actor__bekkers_pi`); tracking-frames deprecated so the frames-only receiver-density feature is absent
- Out-of-fold (GroupKFold by match): **AUC 0.662**, **ECE 0.029**, reliability slope 0.92
- Calibration gate (ECE<=0.10 AND |slope-1|<=0.25): **PASS**
