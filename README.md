# BBO Capstone Project — Imperial College London AI/ML Programme

Black-box optimisation challenge: maximise the output of eight unknown functions across eleven rounds using sequential model-based optimisation.

## Contents

| File | Description |
|------|-------------|
| [`capstone_project.ipynb`](capstone_project.ipynb) | Main notebook — all rounds, strategies, and query generation |
| [`DATASHEET.md`](DATASHEET.md) | Datasheet for the BBO query dataset |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model card for the GP-UCB optimisation approach |
| [`Initial_data_points_starter/`](Initial_data_points_starter/) | Initial function evaluations provided by the course portal (.npy files) |

## Approach Summary

- **Rounds 1–3:** Exploration — GP with UCB acquisition over random Latin Hypercube candidates
- **Rounds 4–8:** Hybrid — GP-UCB and neural network gradient ascent, function-specific strategy
- **Rounds 9–11:** Tight local exploitation — pure scikit-learn GP-UCB with Matern kernel ensemble, anchored on confirmed all-time best inputs per function

## Results Summary (after Round 10)

| Function | Best Output | Round |
|----------|------------|-------|
| F1 | ~0 (confirmed zero) | Retired |
| F2 | 0.6686 | R7 |
| F3 | -0.0129 | R8 |
| F4 | 0.4394 | R10 |
| F5 | 8662.48 | R9 |
| F6 | -0.1778 | R6 |
| F7 | 2.2164 | R10 |
| F8 | 9.9663 | R10 |

## Documentation

- [Datasheet](DATASHEET.md) — motivation, composition, collection process, preprocessing, distribution
- [Model Card](MODEL_CARD.md) — approach overview, intended use, strategy details, performance, assumptions and limitations, ethical considerations
