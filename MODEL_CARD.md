# Model Card: BBO Capstone Optimisation Approach

## Overview

**Name:** GP-UCB Local Exploitation with Adaptive Radius  
**Type:** Sequential model-based optimisation (SMBO) using Gaussian Process surrogate with Upper Confidence Bound acquisition  
**Implementation:** scikit-learn `GaussianProcessRegressor` with Matern kernel ensemble  
**Version:** v4 (rounds 9--12; see Details for full evolution)

---

## Intended Use

**Suitable for:**
- Black-box optimisation of continuous functions where gradients are unavailable
- Settings with extremely limited evaluation budgets (10–20 total queries per function)
- Functions where prior evaluations suggest a known promising region — this approach is exploitation-first and benefits from a warm start
- Low-to-moderate dimensional input spaces (2–5 dimensions); performance degrades significantly in higher dimensions with this data volume

**Use cases to avoid:**
- Global optimisation from scratch with no prior information — the tight local radius will miss the global optimum if it is distant from the initial best
- Functions with multiple well-separated optima of similar magnitude — the strategy commits to a single local region and does not maintain a diverse portfolio of candidates
- High-dimensional functions (≥6D) with fewer than 50 training points — the GP surrogate will be extrapolating across the vast majority of the search space, and UCB scores will be dominated by uncertainty rather than meaningful signal
- Any setting where the function landscape changes over time (non-stationarity) — the Matern kernel assumes a stationary covariance structure

---

## Details: Strategy Evolution Across Twelve Rounds

### Phase 1 — Exploration (R1–R3)
Initial queries used simple perturbation of the provided best input (R1), followed by a single GP with UCB acquisition over random candidates drawn via Latin Hypercube Sampling across the full input domain (R2–R3). Results were mixed: F5 showed strong response to boundary-seeking; most other functions showed high variance.

### Phase 2 — Hybrid Exploitation (R4–R8)
A function-specific strategy was adopted: GP-UCB for functions showing smooth landscapes, TensorFlow neural network gradient ascent for functions showing sharp or non-linear structure. The NN approach used a small dense network (2 hidden layers, ReLU) fit to accumulated data, with gradient ascent from the current best. F6 achieved its all-time best in R6 (−0.178) using this approach; F5 showed super-linear gain from boundary-seeking across all rounds.

TensorFlow was removed in R9 after runtime instability — sequential `model.fit` calls across 6 functions caused 5+ minute hangs per execution. All functions reverted to sklearn GP-UCB.

### Phase 3 -- Tight Local Exploitation (R9--R11)
All active functions (excluding F1 and F5) used GP-UCB with tight radii anchored on their confirmed all-time best inputs:

| Function | Radius | Rationale |
|----------|--------|-----------|
| F2 | +/-0.005 | Recovered and surpassed R7 best (0.6686) in R11 (0.6947) |
| F3 | +/-0.005 | Best round ever in R11 (-0.0108); active improvement seam |
| F4 | +/-0.005 | R10 new best (+0.110 jump); R11 regressed; re-anchored R10 best for R12 |
| F6 | +/-0.003 to +/-0.0001 | Pathological spike; even 0.000099 from R5 best collapses output -0.178 to -0.337 |
| F7 | +/-0.010 | Consistent upward trend R6--R11; new best each round |
| F8 | +/-0.005 | Near ceiling (~10.0); marginal but consistent gains |

### Phase 4 -- Clustering-Guided Exploration (R12)
F6 local search definitively exhausted after confirming sub-0.0001 spike via +/-0.0001 random probe. Hierarchical clustering (Ward linkage, k=3) applied to all 11 F6 input-output pairs to identify unexplored regions. The three clusters revealed that all post-R5 queries (R6--R11) lie within 0.000099 of the R5 best -- confirming the strategy had been searching an infinitesimally small neighbourhood. The most unexplored point (max-min-distance from all known inputs) was identified at [0.8538, 0.119, 0.035, 0.0172, 0.9531], 1.31 Euclidean units from all known F6 queries. R12 submits this as a final basin exploration. All other functions continue GP-UCB exploitation anchored on R11 bests.

**Kernel:** Matern ensemble — `C(1.0)*Matern(ls=0.05, ν=2.5) + C(1.0)*Matern(ls=0.10, ν=2.5)`  
**Candidates:** 5000 per function per round  
**Acquisition:** UCB with β=2.0  
**Seed:** 42 (reproducible)

---

## Performance

Results across all rounds (Init + R1--R11):

| Fn | Initial | Best achieved | Round | Trend |
|----|---------|--------------|-------|-------|
| F1 | ~0 | ~0 | All | Confirmed zero -- retired R7 |
| F2 | 0.611 | **0.695** | R11 | Surpassed R7 peak after 4 rounds of recovery |
| F3 | -0.035 | **-0.011** | R11 | Best round ever in R11; active seam |
| F4 | -4.026 | **0.439** | R10 | Breakthrough R10; R11 regressed |
| F5 | 1089 | **8662** | R9 | Super-linear boundary scaling; ceiling R9--R11 |
| F6 | -0.714 | **-0.178** | R5 | Sub-0.0001 spike; local search exhausted |
| F7 | 1.365 | **2.296** | R11 | Consistent upward trend across 6 rounds |
| F8 | 9.598 | **9.967** | R11 | Near ceiling; marginal but consistent gains |

**Metric used:** Raw function output from the course oracle. No normalisation applied across functions — outputs are on incomparable scales (F5: ~8662 vs F3: ~-0.013).

---

## Assumptions and Limitations

**Assumptions:**
1. Functions are deterministic — the same input always returns the same output. This justifies not re-querying known points.
2. The Matern kernel's stationarity assumption holds — covariance between outputs depends only on distance between inputs. F6 violates this: its landscape has a pathologically narrow peak that the GP cannot represent with length scale 0.05.
3. The local optimum identified in early rounds is near the global optimum. This is unverifiable with the available data but drives the tight-radius exploitation strategy.

**Limitations:**
- **Curse of dimensionality:** With 10 training points in 8 dimensions (F8), the GP is extrapolating for virtually the entire search space. Uncertainty estimates are unreliable beyond the immediate vicinity of known points.
- **No global recovery:** The strategy does not include mechanisms for escaping local optima. F6's all-time best from R5 has not been recovered in 6 subsequent rounds. Hierarchical clustering applied in R12 identified that all post-R5 queries lie within 0.000099 of the R5 best -- the strategy was trapped in a sub-0.0001 neighbourhood for 6 rounds without realising it. R12 pivots to a completely unexplored basin as a last resort.
- **Computational constraints:** 5000 candidates per function with GP fitting is the practical limit for single-session notebook execution. Larger candidate sets or multi-restart optimisation would be more thorough but slower.
- **Single query per round:** One evaluation per function per round severely limits the rate of information gain, particularly in high dimensions.

---

## Ethical Considerations

**Transparency and reproducibility:** The full strategy is documented in a public notebook with all training data embedded inline. Random seeds are fixed. A second researcher can reproduce every query by running the notebook cells in order. The markdown cell preceding each round's code states the explicit reasoning for method and radius choices, making the decision logic auditable.

**Real-world adaptation:** The GP-UCB approach used here is directly applicable to real-world experimental optimisation (materials discovery, hyperparameter tuning, clinical dose-finding) where function evaluations are expensive and gradients are unavailable. The key adaptations required would be: (1) replacing the course oracle with the real evaluation function, (2) scaling the candidate generation to match the evaluation budget, and (3) incorporating domain knowledge into the prior (kernel choice and initial length scales). The F6 case study — where the GP surrogate failed to represent a sharp landscape feature and had to be abandoned in favour of direct random search — is a practical illustration of the importance of validating surrogate assumptions before trusting acquisition function guidance.
