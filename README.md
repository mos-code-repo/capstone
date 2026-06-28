# BBO Capstone Project — Imperial College London AI/ML Programme

Black-box optimisation challenge: maximise the output of eight unknown functions across thirteen rounds using sequential model-based optimisation.

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
- **Round 13 (Final):** Pure exploitation anchored on all-time bests; F6 midpoint probe between R5 best and R2 cluster as final landscape survey

## Results Summary -- Final (13 Rounds)

| Function | Best Output | Round | Notes |
|----------|------------|-------|-------|
| F1 | ~0 (confirmed zero) | Retired R7 | Output effectively zero across all inputs |
| F2 | 0.6947 | R11 | Sharp local maximum; R12-R13 oscillated below |
| F3 | -0.0108 | R11 | Sharp local maximum; R12-R13 oscillated below |
| F4 | 0.5962 | R13 | Active seam; +0.157 cumulative gain R10-R13 |
| F5 | 8662.4825 | R9 | Super-linear boundary scaling; ceiling confirmed R9-R13 |
| F6 | -0.1778 | R5 | Sub-0.0001 spike; unexplored basin confirmed empty R12 |
| F7 | 2.4330 | R13 | New best every round R6-R13 -- 8 consecutive improvements |
| F8 | 9.9724 | R13 | Approaching ceiling at 10.0 |

## Key Findings

- **F7 sustained improvement:** new all-time best in every round from R6 to R13 -- 8 consecutive improvements across a 6-dimensional space
- **F4 active seam:** largest cumulative gain in the final phase -- 0.439 (R10) to 0.542 (R12) to 0.596 (R13), driven by GP-UCB exploitation of a well-defined local gradient
- **F5 boundary scaling:** super-linear output growth as all dimensions approach 1.0; ceiling at [1,1,1,1] = 8662.4825 confirmed stable across 5 rounds (R9-R13)
- **F6 pathological peak:** all-time best -0.1778 at R5; sub-0.0001 spike confirmed R11; hierarchical clustering in R12 identified unexplored basin (returned -2.85); midpoint probe in R13 returned -0.669 -- no second peak exists
- **R11 best single round:** 4 new all-time bests (F2, F3, F7, F8) in one submission
- **R13 final round:** 3 new all-time bests (F4, F7, F8)

## Documentation

- [Datasheet](DATASHEET.md) -- motivation, composition, collection process, preprocessing, distribution
- [Model Card](MODEL_CARD.md) -- approach overview, intended use, strategy details, performance, assumptions and limitations, ethical considerations
