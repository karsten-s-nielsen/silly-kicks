# Silly Kicks

![The Modern SPADL Analyst vs The Chief SPADL Evaluator & Classifier](https://raw.githubusercontent.com/karsten-s-nielsen/silly-kicks/main/assets/silly-kicks.jpg)
<sup>Comic by NanoBanana &mdash; inspired by Monty Python's <em>Ministry of Silly Walks</em></sup>

[![CI](https://github.com/karsten-s-nielsen/silly-kicks/actions/workflows/ci.yml/badge.svg)](https://github.com/karsten-s-nielsen/silly-kicks/actions/workflows/ci.yml)

*The Ministry requires that all football actions be properly classified and valued.*

**silly-kicks** is a Python library for objectively quantifying the impact of
individual actions performed by football players using event stream data.

It is an independently maintained successor to
[socceraction](https://github.com/ML-KULeuven/socceraction), originally
developed by Tom Decroos and Pieter Robberechts at KU Leuven. Built under the
MIT license with full attribution preserved.

## Features

- **SPADL** -- Soccer Player Action Description Language: a unified schema for
  on-ball actions with dedicated DataFrame converters for StatsBomb, Wyscout,
  Opta, Sportec / IDSSE Bundesliga, Metrica Sports, and PFF FC — plus a kloppy
  gateway for raw-provider-data consumers (StatsBomb, Sportec, Metrica)
- **VAEP** -- Valuing Actions by Estimating Probabilities: a framework for
  quantifying the value of individual actions
- **Atomic SPADL** -- continuous (non-discretized) action representation

## Installation

```bash
pip install silly-kicks
```

Requires Python 3.10 or later.

With optional provider support:

```bash
pip install "silly-kicks[kloppy,xgboost]"
```

## Quick Start

```python
import silly_kicks.spadl as spadl

# Convert StatsBomb events to SPADL actions
actions, report = spadl.statsbomb.convert_to_actions(events, home_team_id=123)

# Add human-readable names
actions = spadl.add_names(actions)
```

## VAEP Workflow

The full pipeline: convert provider events to SPADL, train a VAEP model, and
rate individual actions.

```python
from silly_kicks.spadl import statsbomb
from silly_kicks.vaep import VAEP

# 1. Convert provider events to SPADL
actions, report = statsbomb.convert_to_actions(
    events_df, home_team_id=home_team_id,
    xy_fidelity_version=2, shot_fidelity_version=2,
)

# 2. Train a VAEP model
model = VAEP(nb_prev_actions=3)
features = model.compute_features(game, actions)
labels = model.compute_labels(game, actions)
model.fit(features, labels, learner="xgboost", random_state=42)

# 3. Rate actions
ratings = model.rate(game, actions)
# Returns DataFrame with offensive_value, defensive_value, vaep_value
```

### Hybrid-VAEP

Standard VAEP includes the action's result (success/fail) as a feature, which
creates information leakage. HybridVAEP removes result information from the
current action while preserving it for previous actions.

```python
from silly_kicks.vaep import HybridVAEP

# HybridVAEP removes result leakage from current-action features
model = HybridVAEP(nb_prev_actions=3)
# Same fit/rate API as standard VAEP
```

### Multi-Provider Support

All converters share the same output schema, so downstream code works
identically regardless of the data provider.

```python
from silly_kicks.spadl import opta, wyscout

actions_opta, _ = opta.convert_to_actions(opta_events, home_team_id)
actions_wyscout, _ = wyscout.convert_to_actions(wyscout_events, home_team_id)
```

## Architecture

Open [`docs/c4/architecture.html`](docs/c4/architecture.html) in a browser to explore the C4 architecture diagrams (System Context, Containers).

## Attribution

This project builds on the foundational research by the KU Leuven Machine
Learning Research Group. If you use this library in academic work, please cite
the original papers:

```bibtex
@inproceedings{Decroos2019VAEP,
  title     = {Actions Speak Louder than Goals: Valuing Player Actions in Soccer},
  author    = {Tom Decroos and Lotte Bransen and Jan Van Haaren and Jesse Davis},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference
               on Knowledge Discovery \& Data Mining},
  pages     = {1851--1861},
  year      = {2019},
  doi       = {10.1145/3292500.3330758}
}

@inproceedings{Decroos2020AtomicSPADL,
  title     = {Interpretable Prediction of Goals in Soccer},
  author    = {Tom Decroos and Jesse Davis},
  booktitle = {Proceedings of the AAAI-20 Workshop on Artificial Intelligence
               in Team Sports},
  year      = {2020}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards,
and PR process. Open items and planned work are tracked in [TODO.md](TODO.md).

## License

MIT License. See [LICENSE](LICENSE) for details.
