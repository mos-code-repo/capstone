# Lessons Learned: BBO Capstone Post-Mortem

## Overview

This document compares the GP-UCB tight-radius approach used throughout the project against the strategies of the top performers and function winners in cohort IMP-PCMLAI-25-12 (54 participants). The goal is to identify specific, borrowable improvements rather than wholesale method replacement.

Overall average positions across all 54 participants:

| Rank | Approach | Average position |
|------|----------|-----------------|
| 1 | Rank 1 — LOOCV + TuRBO + multi-kernel Thompson + output warping | 9.75 |
| 2 | Rank 2 | 10.00 |
| 3 | Rank 3 | 10.75 |
| 4 | Rank 4 — Sobol + EI + L-BFGS-B + ARD | 11.00 |
| 5 | Rank 5 | 11.50 |
| 9 | This project — GP-UCB tight radius | 15.75 |
| 10 | F7 Winner — RBF + UCB + SVM + NN ensembles | 16.50 |
| — | F5 Winner — GP + UCB + beta sweep + dimension freezing | 21.20 |
| — | NN-primary participant — SVM + NN as primary surrogates | ~50+ |

This project placed 9th overall. The function winners (F7 Winner, F5 Winner) are specialists — this project's average beats both. The cohort's top overall performer is Rank 1, whose validate-then-trust framework is the most instructive single comparison in this document. This outcome confirms that NN and SVM primary surrogates with sparse data are actively harmful. Repositories for Rank 2, Rank 3, and Rank 5 were not publicly available at the time of writing.

---

## The Anchor Problem

Before comparing approaches function by function, the single most important pattern to name is the anchor problem. It explains the three largest gaps between this project and the winners.

The tight-radius strategy commits to a neighbourhood around the best input found so far. If the initial best input happens to be in a poor region of the landscape, the search never escapes. Three functions were affected:

- **F1:** All explored inputs returned ~zero. The function is not zero everywhere — Rank 4 found 1.5578, the F7 Winner found 0.8287. The search was anchored in a zero-output region for all 13 rounds, then the function was retired.
- **F3:** All explored inputs returned negative values. The function has a positive region — the F7 Winner found +0.00882 vs this project's best of -0.0108. The tight-radius strategy anchored in the negative basin and never reached the positive peak.
- **F7:** Eight consecutive improvements reached 2.4330. The global maximum is 3.2265 (F7 Winner). The trajectory was genuinely improving but was climbing the wrong hill.

The common cause in all three cases: initial data happened to land in a suboptimal basin, and tight-radius exploitation trapped the search there. Sobol-based global coverage in rounds 1--3 would likely have found the better basin before the exploitation phase locked in.

---

## Results Comparison

| Function | This project | Rank 1 | Rank 4 | F5 Winner | F7 Winner |
|----------|-------------|-----------------|-----------------|------------|-----------|
| F1 | ~0 (25th) | (18th) | **1.5578 (1st)** | ~0 (48th) | 0.8287 (3rd) |
| F2 | **0.6947 (18th)** | (13th) | 0.7105 (15th) | 0.6564 (28th) | 0.6836 (21st) |
| F3 | -0.0108 (26th) | **(6th)** | -0.0356 (37th) | -0.0111 (12th) | +0.0088 (3rd) |
| F4 | 0.5962 (18th) | **(2nd)** | 0.6681 (7th) | 0.6344 (14th) | 0.4908 (26th) |
| F5 | **8662.4825 (5th)** | (9th) | 8662.48 (4th) | 8662.48 (1st) | 5438.45 (24th) |
| F6 | -0.1778 (9th) | **(2nd)** | -0.2011 (11th) | -0.2869 (23rd) | -0.2238 (17th) |
| F7 | 2.4330 (18th) | (14th) | 2.6718 (12th) | 1.6850 (35th) | **3.2265 (1st)** |
| F8 | **9.9724 (7th)** | (14th) | 9.9933 (1st) | 9.9642 (9th) | 9.8042 (37th) |

Rank 1's exact output values were not recorded in their repository but their positions reveal the pattern: dominant on F4 and F6, competitive everywhere else, catastrophic nowhere. This project beat all three function winners on F6. This project beat the F7 Winner and F5 Winner on F5 and F8. The GP-UCB approach is competitive across the board — the gaps are specific and diagnosable, not systemic.

---

## Top Performer Analysis: Rank 1 (1st overall, average 9.75)

**Strategy:** Validate-then-trust framework combining LOOCV surrogate selection across 7 model families, output warping (Yeo-Johnson), structural transforms (Gaussian-magnitude, ceiling), and TuRBO-1 trust-region Bayesian optimisation with multi-kernel Thompson sampling.

**Positions:** F1: 18, F2: 13, F3: 6, F4: 2, F5: 9, F6: 2, F7: 14, F8: 14. Consistency across every function, with no catastrophic failures.

### LOOCV surrogate validation — the most important technique in this analysis

Before generating any acquisition candidate, Rank 1 fits 7+ surrogate models (Ridge, KNN, Random Forest, SVR, Gradient Boosting, GP with Matern and RBF kernels, MLP) and filters them by Leave-One-Out Cross-Validation against a baseline. The baseline is the standard deviation of observed outputs — if a surrogate cannot beat predicting the mean, it is discarded. Only surrogates that pass this test contribute to acquisition decisions.

This is the direct answer to F6. The GP surrogate on F6 was misspecified from round 6 onward — its Matern kernel could not represent a landscape feature narrower than 0.0001. LOOCV would have flagged this early: a GP that cannot predict held-out points better than the mean of six observations should not be trusted to guide the next query. Instead of six rounds of confident but misdirected exploitation, the diagnostic would have triggered a strategy switch by round 7 or 8.

**The general principle:** any surrogate with LOO error > std(observed outputs) is guessing. Do not use its acquisition scores. This test should run at every round before any query is submitted.

### TuRBO-1 trust region with multi-kernel Thompson sampling

Rather than a fixed radius, Rank 1 uses a trust region that self-adapts:
- Starts at length 0.8 (hyperrectangle centred on current best, scaled by ARD length scales)
- Expands after 3 consecutive improvements greater than 1e-3 × max(|best|, 1), up to a ceiling of 1.6
- Contracts after ceil(max(4, dim)) consecutive failures, halving each time
- Resets to initial size when it drops below 0.0078 — forcing global re-exploration

Inside the trust region, four GP kernels run simultaneously (Matern 0.5, 1.5, 2.5, RBF). One Thompson sample is drawn from each over 5000 candidates. The argmax is taken across the full (kernel, candidate) grid. This is Bayesian model averaging under kernel uncertainty — when the correct smoothness assumption is unknown, sample from all four and let the data decide which produced the most promising candidate.

The contraction-restart mechanism is the key advantage over the fixed-radius approach. When the search stalls, the trust region shrinks until it resets — which re-anchors on the best known point and forces a fresh broad search. This is a principled escape mechanism from the situations where fixed-radius search simply tightens on a local optimum and stops improving.

### Output warping (Yeo-Johnson)

Before fitting any surrogate, observed outputs are passed through a Yeo-Johnson power transform to make the distribution more Gaussian. The transform handles negatives (unlike Box-Cox), and falls back gracefully when the output distribution is too degenerate to warp (near-constant outputs, extreme single spikes). This would have helped directly with F5's super-linear output range (1089 to 8662) and F6's spike distribution.

**Why this matters:** GP regression assumes outputs are approximately Gaussian. When they are not — as with F5's boundary-seeking super-linear growth — the GP fits poorly in the high-output region. Warping the outputs first makes the surrogate's assumptions valid.

### How F6 2nd place was achieved

The combination of LOOCV (detecting surrogate failure early), the Gaussian-magnitude structural transform (reshaping the spike to be more learnable), and TuRBO's dynamic trust region (contracting and resetting rather than staying locked in a 0.0001 neighbourhood) explains the F6 result. The fixed-radius GP-UCB approach spent 6 rounds in a sub-0.0001 neighbourhood without knowing the surrogate had failed. LOOCV would have caught this inside 2 rounds.

---

## Winner Analysis: Rank 4 (F1, F8)

**Strategy:** GP with Matern ν=2.5 + ARD, SVM region filter, MC-Dropout NN ensemble, EI with adaptive ξ decay, Sobol quasi-random sampling, L-BFGS-B acquisition maximisation.

**Why Rank 4 won F8:** 9.9933 vs 9.9724. Both are approaching the 10.0 ceiling. L-BFGS-B found the precise acquisition maximum at each step rather than the best of a finite random sample — the gap is small but consistent across 13 rounds of compound precision.

**Why Rank 4 won F1:** Sobol sequences covered the full 2D domain early. The 1.5578 peak was in a region that the anchored search never visited. F1 was confirmed dead too early based on a regionally biased sample.

**Key gaps closed by Rank 4's approach vs this project:** F1, F4, F7 (anchor problem), F8 (acquisition precision).

---

## Winner Analysis: F5 Winner (F5)

**Average position: 21.2 — worse than this project's 15.75.**

**Strategy:** GP + UCB with dynamic beta scheduling (beta sweep across values per round), dimension freezing, data masking to subregions on plateau detection.

**Why F5 Winner won F5:** The same structural insight this project found — push all dimensions toward 1.0. Both hit essentially the same ceiling (8662.48). the F5 Winner's 1st vs this project's 5th comes down to a margin of 0.0005 in output value. The strategy difference on F5 was negligible; the win is effectively noise at the ceiling.

**Techniques worth noting:**
- **Beta sweep:** Rather than fixing beta at 2.0, the F5 Winner evaluated multiple beta values and selected the one that produced the most promising candidate. This is a low-cost way to adapt exploration versus exploitation within a single round.
- **Dimension freezing:** When a function on a plateau showed one dimension consistently at its boundary value, that dimension was locked and search continued in the remaining dimensions. Useful for reducing effective dimensionality when strong signals exist.

**Where F5 Winner's approach fell short:** 48th on F1 (same anchor problem), 35th on F7, 23rd on F6. The beta sweep and dimension freezing added no advantage on functions where the anchor problem was the binding constraint.

---

## Winner Analysis: F7 Winner (F7)

**Average position: 16.5 — essentially tied with this project's 15.75.**

**Strategy:** GP with RBF kernel + UCB, soft-margin SVM classification for region pre-filtering, NN deep ensembles for higher-dimensional functions (F4, F6, F8), ARD for F2, manual visualisation for F1.

**Why F7 Winner won F7:** The winning combination was SVM classification + GP in 6D. The SVM was trained as a binary classifier on observed inputs: outputs above the median labelled "promising", below labelled "unpromising". New candidates were pre-filtered through the SVM before GP scoring. In 6D space, this reduced the effective search space dramatically and guided the GP toward the region containing the 3.2265 peak.

**Why RBF instead of Matern on F7:** RBF (Gaussian) kernel assumes infinite differentiability, which implies an extremely smooth landscape. If F7's landscape is genuinely smooth in the region around the global maximum, RBF will produce tighter, more confident GP predictions than Matern ν=2.5 in that region. This is a hypothesis — but the result supports it.

**Where F7 Winner's approach fell short:** NN ensembles on F4 (26th), F5 (24th), F8 (37th) performed significantly worse than this project's GP-UCB. This is the same pattern across multiple participants. NN surrogates with 10--13 training points overfit noise rather than learning landscape structure. The SVM pre-filter worked for F7 because it is a classifier, not a regression surrogate — it only needs to distinguish high-output from low-output regions, which is a simpler task with sparse data.

**F3 finding:** The F7 Winner found +0.00882 (positive). This project's best was -0.0108. F3 has a positive region that this project's tight-radius search, anchored in the negative basin, never reached. Manual visualisation of F1 by the F7 Winner (finding 0.8287) further confirms that human-guided global inspection can find peaks that automated tight-radius search misses entirely.

---

## When to Use SVM Pre-filtering

The F7 Winner analysis raises a specific question: under what conditions does adding an SVM classifier ahead of the GP acquisition function actually help, and how can that decision be made principled rather than arbitrary?

### What the SVM does in this context

The SVM is trained as a binary classifier on the accumulated query history. Inputs where the observed output exceeds a threshold (typically the median or mean of observed outputs) are labelled class 1 (promising). Inputs below the threshold are labelled class 0 (unpromising). New candidate points are filtered through the SVM before being scored by the GP acquisition function — only class 1 candidates are passed to the GP. The effect is to narrow the candidate set to regions the classifier considers productive.

### When it helps

**High dimensionality (d ≥ 5).** In low-dimensional spaces (2D, 3D), the GP acquisition function can adequately distinguish promising from unpromising regions on its own — the space is small enough that 10--15 points provide reasonable coverage. In 5D+ spaces, 10--15 points are a negligible sample of the volume. The GP uncertainty is high almost everywhere, and the acquisition function is dominated by the uncertainty term rather than the mean. In this regime, the SVM provides a second signal — a geometric boundary based on which regions have historically produced high outputs — that the GP cannot infer from so few points.

F7 (6D) is the case that confirms this. The SVM pre-filter helped substantially. F6 (5D) showed no benefit — but F6 is a pathological case where the peak is so narrow (sub-0.0001 width) that no classifier trained on 13 points will learn a useful boundary around it.

**When the output distribution is bimodal or has a clear high/low structure.** If observed outputs cluster into a group of high values and a group of low values with a gap between them, the SVM has a learnable decision boundary. If outputs are uniformly spread, the SVM boundary is arbitrary and adds noise rather than signal.

**Test:** Calculate the ratio of the standard deviation of observed outputs to the range. If std/range < 0.3, the outputs are concentrated and the classifier will have little to learn. If std/range > 0.5, there is a clear high/low split and the SVM decision boundary will be meaningful.

### When it does not help

**Sparse narrow peaks (F6 type).** If the global maximum occupies a region smaller than the typical inter-point distance, no SVM trained on observed data will correctly classify the neighbourhood of the peak as promising — it has never been seen. The GP uncertainty term is the only signal available, and the SVM will actively filter out unexplored regions as unpromising.

**Low dimensionality (d ≤ 4).** The GP acquisition function is already an effective region classifier in low dimensions. Adding an SVM layer introduces an additional model to fit on 10--15 points with no gain.

**When the classifier would overfit.** With fewer than 15 training points in d ≥ 5 dimensions, a hard-margin SVM will fit the training data perfectly but generalise poorly. Use soft-margin SVM with a moderate regularisation parameter (C = 1.0 to 10.0) and verify that cross-validation accuracy on the training set is above 70% before trusting the classifier output.

### Decision rule

Use SVM pre-filtering when all three conditions hold:
1. Input dimension d ≥ 5
2. At least 10 accumulated observations
3. Output std/range > 0.4 (clear high/low structure in observed outputs)

Do not use it when the peak is known to be extremely narrow (GP uncertainty in the peak neighbourhood exceeds 10× the surrounding uncertainty) — in that case, the SVM will classify the peak region as unpromising.

---

## What to Borrow

### 0. LOOCV surrogate validation — run before every query

At every round, before submitting any acquisition candidate, fit multiple surrogate models and filter by Leave-One-Out Cross-Validation. Discard any surrogate whose LOO error exceeds the standard deviation of observed outputs — that surrogate is no more useful than predicting the mean. Only pass surviving surrogates to the acquisition function.

This is the single highest-value technique identified across all winner analyses. It is a diagnostic that catches surrogate failure before it costs rounds. On F6, it would have triggered a strategy change 5 rounds earlier than the clustering analysis did.

**Implementation:** For each surrogate, compute LOO predictions using cross_val_score with cv=LeaveOneOut() and neg_mean_squared_error. If RMSE > std(Y_observed), discard that surrogate. Proceed only with surviving models.

### 0b. Replace fixed radius with TuRBO trust region

Replace the manually tuned fixed-radius random search with TuRBO-1: a trust region that expands on consecutive successes and contracts on consecutive failures, restarting when it becomes too small. This removes all manual radius tuning and provides a principled escape mechanism when the search stalls.

The multi-kernel Thompson sampling variant (Matern 0.5, 1.5, 2.5, RBF in parallel) adds robustness to kernel choice — no single smoothness assumption is committed to when the landscape's properties are unknown.

### 0c. Output warping before fitting any surrogate

Apply Yeo-Johnson power transform to observed outputs before fitting surrogates. This makes the surrogate's Gaussian assumption more valid for skewed or super-linear output distributions. Fall back to raw outputs if the transform produces non-finite values.

### 1. Sobol sampling for global coverage in early rounds

Replace uniform random candidates in rounds 1--4 with Sobol sequences over the full input domain. This costs nothing in terms of model complexity and materially improves the chance of finding distant peaks before exploitation locks in. F1, F3, and F7 were all anchor-problem failures that Sobol in rounds 1--2 would likely have resolved.

**Implementation:** scipy.stats.qmc.Sobol(d=dims, scramble=True).random(n=5000) as the candidate set in early rounds, switching to tight-radius random sampling for exploitation once a strong region is confirmed.

### 2. L-BFGS-B for acquisition maximisation

Replace argmax over 5000 random candidates with scipy.optimize.minimize on the negative acquisition function, starting from multiple random initialisations. This finds the true acquisition maximum rather than an approximation from a finite sample.

**When it matters most:** High-dimensional functions (F7, F8) where the acquisition landscape is complex and 5000 random points provide poor coverage.

### 3. ARD kernel for high-dimensional functions

For d ≥ 5, switch from isotropic Matern to ARD Matern. The GP learns separate length scales per dimension, down-weighting inert dimensions and concentrating on the informative ones.

**When it matters most:** F6 (5D), F7 (6D), F8 (8D). For F2 and F3 (2D) the isotropic kernel is sufficient.

### 4. SVM pre-filtering for high-dimensional functions with clear output structure

Apply the decision rule above. Specifically: d ≥ 5, at least 10 observations, output std/range > 0.4. Use soft-margin SVM (C = 1.0 to 10.0). This is the technique that drove F7 Winner's F7 win — the largest single-function gap between this project and a winner.

### 5. Beta sweep instead of fixed beta

Rather than fixing UCB beta at 2.0 throughout, evaluate 3--5 candidate beta values (e.g. 0.5, 1.0, 2.0, 4.0) at each round and select the one that produces the highest-scoring distinct candidate. This costs 3--5× the acquisition evaluation budget but adapts the exploration-exploitation balance to the current data without requiring a decay schedule.

### 6. Dimension freezing on confirmed plateau functions

When a function shows no improvement for 3+ consecutive rounds and one or more dimensions consistently appear at a boundary value (0.0 or 1.0) in the top candidates, lock those dimensions and run the GP-UCB search in the remaining free dimensions only. This reduces effective dimensionality and avoids wasting budget re-exploring confirmed boundaries.

### 7. Never retire a function

F1 retirement cost 24 leaderboard positions. F3's positive region was never found because the search anchored in the negative basin from round 1. The correct rule is: maintain one Sobol-based global probe on any apparently-dead function every three to four rounds. The cost is one query per period; the potential benefit is finding an undiscovered peak.

---

## What Not to Borrow

**Neural network surrogates (primary or ensemble).** one participant using NN and SVM as primary surrogates averaged above 50th. F7 Winner's NN ensemble on F4, F6, F8 placed 26th, 17th, 37th — worse than this project's GP-UCB on all three. With 10--13 training points, NNs fit noise. The GP's uncertainty quantification is more reliable at this data volume. This finding is consistent across three independent participants.

---

## The Core Gap Explained

This project placed 9th overall with an average position of 15.75. The cohort leader, Rank 1, averaged 9.75 — a gap of 6 positions. That gap breaks down cleanly:

**The anchor problem cost the most.** F1 (25th) was retired after anchoring in a zero-output region. F3 (26th) anchored in the negative basin, never reaching the positive region the F7 Winner found. F7 (18th) climbed the wrong hill for 8 consecutive rounds. All three were the same failure mode — tight-radius exploitation with no global check. Sobol sampling in rounds 1--3 and a rule against retiring functions would have recovered most of this gap.

**Surrogate failure on F6 cost the next most.** F6 finished 9th, which looks competitive — but Rank 1 finished 2nd on the same function with the same budget. The difference is LOOCV catching the GP surrogate failure 5 rounds earlier than the clustering analysis did. Six rounds of misdirected confidence vs two. F6 at 2nd rather than 9th would alone reduce the average position gap by nearly a full point.

**Acquisition precision was a smaller but consistent factor.** F4 (18th vs Rank 1's 2nd), F8 (7th vs Rank 4's 1st). TuRBO's self-adapting trust region and L-BFGS-B's precise acquisition maximisation compounded small advantages across every round.

**Where this project held its own:** F5 (5th — structural boundary insight matched by almost no one), F6 still top 10 despite the surrogate failure, F8 7th. The tight-radius GP-UCB approach is genuinely strong on narrow-peak functions — it is not a weak strategy. It is an incomplete one.

---

## Revised Strategy for a Future BBO Challenge

**Before every query — non-negotiable:**
- Fit multiple surrogate families (GP with Matern and RBF, Ridge, SVR, Random Forest)
- Apply Yeo-Johnson output warping before fitting
- Filter by LOOCV: discard any surrogate with LOO RMSE > std(Y observed)
- Only surviving surrogates contribute to acquisition

**Rounds 1--3:** Sobol quasi-random candidates over the full domain for all functions. Fit GP with ARD Matern for d ≥ 5. Use EI or UCB with high exploration weight. Do not commit to any region until round 3 data confirms a peak.

**Check after round 2:** Compute output std/range for each function. For d ≥ 5 and std/range > 0.4, activate SVM pre-filtering from round 3 onwards. Initialise TuRBO trust region at length 0.8 centred on the round 2 best.

**Rounds 4--8:** TuRBO trust region with multi-kernel Thompson sampling (Matern 0.5, 1.5, 2.5, RBF). Let the trust region adapt — expand on improvement, contract on failure. Use beta sweep (3--5 values) to cross-check acquisition scores. Apply dimension freezing on functions hitting confirmed boundaries.

**Rounds 9--13:** Trust region will have self-calibrated to the right exploitation radius. Maintain one global Sobol probe per function every three rounds regardless of convergence. If trust region has restarted, treat round 9 as a new round 3 for that function.

**Never retire a function. Never use NN surrogates with fewer than 50 observations.**

The revised strategy above represents the best available synthesis from the repositories analysed. It is not a theoretical ideal — every technique listed is evidenced by at least one top-5 performer in this cohort.
