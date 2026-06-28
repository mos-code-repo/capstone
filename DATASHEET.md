# Datasheet: BBO Capstone Project Dataset

## Motivation

This dataset was created to support a black-box optimisation (BBO) challenge as part of the Imperial College London AI/ML postgraduate programme. The task is to maximise the output of eight unknown functions by submitting one query vector per function per round, with no access to gradients or function definitions. The dataset records the full history of query inputs and corresponding function outputs accumulated across thirteen rounds of iterative search. It supports the development, evaluation, and retrospective analysis of sequential optimisation strategies under extreme data scarcity.

---

## Composition

The dataset contains input-output pairs for eight functions, each with a different dimensionality:

| Function | Input dimension | All-time best output | Round achieved |
|----------|----------------|---------------------|----------------|
| F1 | 2 | ~0 (confirmed zero) | Retired R7 |
| F2 | 2 | 0.6947 | R11 |
| F3 | 3 | -0.0108 | R11 |
| F4 | 4 | 0.5962 | R13 |
| F5 | 4 | 8662.4825 | R9 |
| F6 | 5 | -0.1778 | R5 |
| F7 | 6 | 2.4330 | R13 |
| F8 | 8 | 9.9724 | R13 |

Each function has 14 rows of data: one initial evaluation (provided by the course portal) plus one submitted query per round across thirteen rounds. Total dataset size: 112 input-output pairs (8 functions x 14 evaluations).

**Format:** NumPy arrays (.npy) for initial data; rounds 1--13 embedded inline in the capstone notebook. All inputs are normalised to [0, 1]^d. Outputs are raw scalar values from the oracle.

**Gaps:** F1 output is effectively zero for all inputs evaluated -- no signal to exploit. F6 achieved its best result in R5 (-0.178) and never recovered that value across eight subsequent rounds; hierarchical clustering (Ward linkage, k=3) applied in R12 confirmed a sub-0.0001 spike in 5D; basin exploration in R12 returned -2.85; midpoint probe in R13 returned -0.669 -- confirming no second peak exists in the F6 landscape. The search space for high-dimensional functions (F7: 6D, F8: 8D) remains heavily undersampled -- 13 points in an 8-dimensional unit hypercube provides negligible global coverage.

---

## Collection Process

Queries were generated using a Gaussian Process Upper Confidence Bound (GP-UCB) local search strategy implemented in scikit-learn, with a Matern kernel ensemble (length scales 0.05 and 0.10, ν=2.5). For each function, 5000 candidate points were sampled uniformly within a tight radius of the current best known input. The candidate maximising the UCB acquisition function (β=2.0) was selected as the submitted query.

The strategy evolved across rounds:
- **R1:** Simple nudge — small perturbation of the initial best input
- **R2–R3:** Gaussian Process with UCB acquisition, ensemble GP with Latin Hypercube candidates
- **R4–R8:** Hybrid strategy combining GP-UCB and TensorFlow neural network gradient ascent, with function-specific radius and method selection based on accumulated round history
- **R9--R11:** Pure scikit-learn GP-UCB (TensorFlow dropped due to runtime instability); tight local exploitation anchored on confirmed all-time best inputs per function
- **R11 F6:** GP-UCB abandoned in favour of a dense random probe within +/-0.0001 of the R5 best, after observing that the Matern kernel (length scale 0.05) cannot represent a landscape that drops 0.12 over a distance of 0.003
- **R12 F6:** Hierarchical clustering (Ward linkage) applied to all 11 F6 input points to identify the most unexplored region of the 5D input space; probe submitted at [0.8538, 0.119, 0.035, 0.0172, 0.9531] -- 1.31 Euclidean units from all known F6 queries
- **R12 F2/F3/F4/F7/F8:** GP-UCB continued with tight radii anchored on R11 all-time bests
- **R13 F4/F7/F8:** GP-UCB exploitation anchored on R12 all-time bests; 3 new all-time bests achieved in the final round
- **R13 F6:** Midpoint probe at [0.337, 0.615, 0.681, 0.786, 0.365] between R5 best and R2 cluster as final landscape survey; returned -0.669, confirming no useful structure in that region

**Time frame:** Thirteen weeks, one round per week, following the Imperial College module schedule.

---

## Preprocessing and Uses

**Transformations applied:** All inputs are in the [0, 1]^d domain as specified by the course portal. No normalisation of outputs was applied for submission; outputs were normalised internally within the GP (normalize_y=True) for surrogate fitting only.

**Intended uses:**
- Retrospective analysis of sequential optimisation strategy performance
- Benchmarking of surrogate model quality (GP vs. NN) with extremely sparse training data
- Illustration of the curse of dimensionality in black-box optimisation
- Documentation of how acquisition function choice interacts with landscape properties (sharp peaks, boundary effects, flat regions)

**Inappropriate uses:**
- This dataset should not be used to train a general-purpose surrogate for the underlying functions — 10 points in up to 8 dimensions is insufficient for any reliable generalisation beyond the immediate neighbourhood of observed points
- Results should not be interpreted as evidence of global optima; all reported bests are strong local optima within the explored regions

---

## Distribution and Maintenance

**Availability:** This dataset and the associated notebook are maintained in a public GitHub repository at [https://github.com/mos-code-repo/capstone](https://github.com/mos-code-repo/capstone).

**Terms of use:** Created for academic coursework under the Imperial College London AI/ML programme. Dataset may be freely used for educational and research purposes with attribution.

**Maintenance:** Maintained by the project author. The dataset will be extended if additional rounds are submitted. No automated update pipeline exists — all entries are manually verified against the course portal output files before being added to the notebook.
