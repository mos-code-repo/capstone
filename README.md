# BBO Capstone Project — Imperial College London AI/ML Programme

Black-box optimisation challenge: maximise the output of eight unknown functions across twelve rounds using sequential model-based optimisation.

## Contents

| File | Description |
|------|-------------|
| [`capstone_project.ipynb`](capstone_project.ipynb) | Main notebook — all rounds, strategies, query generation and F6 clustering analysis |
| [`DATASHEET.md`](DATASHEET.md) | Datasheet for the BBO query dataset |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model card for the GP-UCB optimisation approach |
| [`Initial_data_points_starter/`](Initial_data_points_starter/) | Initial function evaluations provided by the course portal (.npy files) |

## Approach Summary

- **Rounds 1--3:** Exploration -- GP with UCB acquisition over random Latin Hypercube candidates
- **Rounds 4--8:** Hybrid -- GP-UCB and neural network gradient ascent, function-specific strategy
- **Rounds 9--11:** Tight local exploitation -- pure scikit-learn GP-UCB with Matern kernel ensemble, anchored on confirmed all-time best inputs per function
- **Round 12:** Hierarchical clustering (Ward linkage) applied to F6 input history to identify unexplored regions; GP-UCB exploitation continues for F2, F3, F4, F7, F8

## Results Summary (after Round 11)

| Function | Best Output | Round | Notes |
|----------|------------|-------|-------|
| F1 | ~0 (confirmed zero) | Retired R7 | Output effectively zero across all inputs |
| F2 | 0.6947 | R11 | Surpassed R7 peak after 4 rounds of recovery |
| F3 | -0.0108 | R11 | Best round ever; active improvement seam |
| F4 | 0.4394 | R10 | R11 regressed; re-anchored for R12 |
| F5 | 8662.48 | R9 | Super-linear boundary scaling; ceiling confirmed R9-R11 |
| F6 | -0.1778 | R5 | Sub-0.0001 spike; local search exhausted; R12 probes new basin |
| F7 | 2.2957 | R11 | Consistent upward trend across 6 rounds |
| F8 | 9.9671 | R11 | Near ceiling; marginal but consistent gains |

## Key Findings

- **F5 boundary scaling:** super-linear output growth as all dimensions approach 1.0, confirmed ceiling at [1,1,1,1] = 8662.48
- **F6 pathological peak:** all-time best -0.1778 achieved at R5; subsequent rounds confirmed a sub-0.0001 spike -- even 0.000099 away the output collapses to -0.337. Hierarchical clustering used in R12 to identify a completely unexplored region of the 5D input space as the final probe
- **F2 recovery:** R7 best (0.6686) held for 4 rounds before being surpassed in R11 (0.6947), demonstrating that GP-UCB with tight radii can eventually recover and improve on sharp peaks with sufficient rounds
- **R11 best single round:** 4 new all-time bests (F2, F3, F7, F8) in one submission

## Documentation

- [Datasheet](DATASHEET.md) -- motivation, composition, collection process, preprocessing, distribution
- [Model Card](MODEL_CARD.md) -- approach overview, intended use, strategy details, performance, assumptions and limitations, ethical considerations
