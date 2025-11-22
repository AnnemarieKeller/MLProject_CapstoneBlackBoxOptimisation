 Rule of thumb for candidate generation
Dimension	Recommended method
1–3D	Random or Sobol/LHS (optional)
4–6D	Sobol or LHS strongly preferred
7D+	Sobol/LHS preferred; may need more points or adaptive candidate refinement
Why? Because even a few random points can cover most of the space.

Example: 2D function, 50–100 random points already fill the area decently.

Random sampling starts to leave gaps — you could miss important regions.

Better methods:

Latin Hypercube Sampling (LHS) → spreads points evenly along each dimension.

Sobol sequences → quasi-random, low-discrepancy, evenly cover space.

Example: 6D function, 500 points with LHS or Sobol gives much better coverage than 500 pure random points.

hiigh dimensions (7D+)

Even LHS/Sobol will start struggling — coverage becomes sparse.

You need more points or smarter candidate selection.

BO is slower, but using LHS/Sobol is still better than pure random.

High-dimensional BO: If input_dim > 8–10, pure Sobol may leave gaps in the space. That’s why some implementations mix in a few random “local perturbations” around current best points. It’s optional but can help exploration.

Lower dimensions (<4): Sobol or LHS are nice but random sampling is usually fine because the space is small.



You’re trying to do black-box optimization (BBO) with Bayesian Optimization) on an unknown function and you have candidate inputs like the ones you posted. You’re asking about whether to use NN surrogate or Random Forest (RF).

Problem with the inputs you posted

Looking closely at your array:

Duplicate points:

[0.950533   0.022066   0.982521   0.086132   0.814387]


appears three times. This is problematic for NN or GP surrogates: repeated inputs can confuse the model and artificially reduce uncertainty, especially if outputs are slightly different.

Scaling: Inputs are between 0 and 1 — good for NN.

Sparsity in some regions: Some regions of the 5D space may be missing samples (e.g., corners, middles), which affects surrogate accuracy.

So, your first step is to ensure candidate points cover the whole space (LHS or Sobol works better than pure random) and remove duplicates.

Choosing the surrogate

Neural Network (NN):

Pros: flexible, can handle high dimensions.

Cons: doesn’t give proper uncertainty easily, which is critical for Bayesian Optimization.

EI / UCB / Thompson sampling all rely on uncertainty; without it, BO degenerates into a heuristic optimizer.

Random Forest (RF):

Pros:

Gives pseudo-uncertainty via variance across trees (good for UCB).

Handles noisy, multi-modal functions well.

Cons: Not smooth — predictions can be piecewise constant, may not extrapolate well.

GP (Gaussian Process):

Pros: gold standard for BO, provides mean + variance.

Cons: scales poorly with large datasets and high dimensions (>8D).

Rule of thumb for unknown functions:

Dimension	Small (<5D)	Medium (5-8D)	High (>8D)
GP	✅ best	✅ maybe, slow	❌ impractical
RF	✅ good	✅ good	✅ good, fast
NN	⚠️ risky	⚠️ risky	✅ possible if using uncertainty methods (Bayesian NN)