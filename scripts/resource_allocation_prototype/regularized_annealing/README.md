Regularized Annealing
=====================

This experiment augments the standard annealing training with purely data–driven regularizations (no synthetic scenario generator required) to reduce the tendency of single–scenario training to overfit label noise at the training contexts.

Implemented regularizations
---------------------------

1) kNN mean anchor
- For each training context x, we compute a local average μ̂(x) of the observed scenario vectors from its k nearest neighbors (in x–space).
- During training we add a small penalty λ_anchor · ||ξ̂(x) − μ̂(x)||² to the decision–focused loss.
- Effect: discourages idiosyncratic predicted scenarios tied to noisy single–sample labels, nudging toward the local conditional mean inferred from the dataset itself.

2) Consistency under input perturbation
- For each x in a batch, create x+ε with small Gaussian noise (σ_x). Predict ξ̂(x+ε) and penalize λ_cons · ||ξ̂(x+ε) − ξ̂(x)||².
- Effect: enforces local Lipschitz behavior of the predictor and reduces sensitivity to tiny covariate changes.

3) L2 weight decay
- Add λ_l2 · ∑||θ||² over parameters to the batch loss.
- Effect: prevents overly sharp solutions, complements the anchor and consistency penalties.

4) Dropout in the network
- A small dropout (default p=0.1) is inserted after each hidden layer in a locally defined network (does not modify library code).
- Effect: improves robustness, combats co–adaptation in hidden units.

5) Validation and checkpointing
- Hold out a small validation split from the training contexts and track the decision–focused validation loss.
- Save the best model across epochs.

What is NOT used here
----------------------
- No multi–scenario augmentation per training context is performed. The regularizations only use the observed (x, ξ) pairs.

Additional test: SAA benchmark on training contexts
---------------------------------------------------
- After training, we run the SAA benchmark on:
  - A random subset of 30 training contexts, and
  - All out–of–sample (OOS) contexts.
- This mirrors the discrepancy observed earlier and helps verify whether the regularizations reduce the large gap on training contexts when evaluated against richer SAA objectives.

How to run
----------

From the repository root:

```
julia scripts/resource_allocation_prototype/regularized_annealing/regularized_annealing.jl
```

Key hyperparameters (inside the script)
---------------------------------------
- λ_anchor = 1e-3, λ_cons = 1e-3, λ_l2 = 1e-4, σ_x = 0.01, k_neighbors = 10, dropout p = 0.1
- batchsize = 10, epochs = 40, step_size = 1e-3

Notes
-----
- The SAA benchmark that compares train–subset vs OOS uses the same generator parameters as elsewhere in the repository purely for evaluation (not for training). This isolates the effect of the regularizations on the previously observed train–context performance gap.

