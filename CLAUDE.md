# TNO Compiler

## Goal

A quantum circuit compiler that produces weighted ensembles of circuits approximating a target unitary to within a diamond-norm error budget, using stochastic tensor renormalization group (TRG) methods. See SPEC.md for the full specification.

## Roadmap

### Stage 1: Basic MPO compiler + test harness (current)

Build a deterministic MPO-based compiler: given a target MPO V, optimize 2-qubit gates of a 1D brickwall circuit U to maximize |Tr(V†U)|². Uses standard MPO contraction for gate environments and Riemannian ADAM on U(4) for optimization. Limited to small/medium circuits (MPO bond dimension grows exponentially), but establishes:
- The compilation loop (environments → gradient → Riemannian update)
- The testing methodology (verify compilation via Qiskit exact simulation)
- The brickwall circuit and MPO representations

### Stage 2: Kalloor ensemble pipeline

Wrap the basic compiler in the ensemble framework from Kalloor et al. 2025:
- Run multiple compilations (different initializations / seeds)
- Compute pairwise overlaps Tr(U_i†U_j) and target overlaps Tr(V†U_i)
- Phase-align, build Gram matrix, PSD-repair, solve convex QP for optimal weights
- Certify via diamond-norm bound: ‖Ē − V‖◇ ≤ 2δ_ens + R²

We do this before replacing the contraction engine because the ensemble pipeline is engine-agnostic. Validating the QP, certification, and overlap estimation in an exactly verifiable setting means we can trust these components when we later swap in stochastic TRG.

### Stage 3: Stochastic TRG contraction engine

Replace deterministic MPO contraction with Ferris sampling + hierarchical TRG (Ferris 2015). This is what breaks through the scaling wall:
- The closed TN Tr(V†U) for a layered target on 1D hardware is a 2D strip
- MPO methods hit a wall because bond dim grows as 4^(min(n,D))
- Stochastic TRG coarse-grains the full 2D network in log(N) levels
- For gapped targets, the sampling dimension d_k saturates → polynomial cost
- The contraction is unbiased (exact in expectation) with controllable variance

The Kalloor pipeline from Stage 2 stays unchanged -- we just feed it circuit candidates produced by the new engine. The overlap estimations also switch to stochastic TRG.

### Stage 4: ∂TRG backward pass

Replace the deterministic MPO environment computation with reverse-mode AD through the stochastic TRG hierarchy (Chen et al. 2019):
- Same cost as the forward pass
- Produces unbiased 4×4 gate environments
- The stochasticity that makes contraction unbiased also makes the optimization trajectory wander, naturally producing ensemble diversity for the Kalloor QP

### Stage 5: Full integration

The stochastic TRG compiler feeds the Kalloor ensemble pipeline end-to-end:
- Trajectory snapshots from the stochastic optimization serve as ensemble members
- Pairwise overlaps estimated by forward-pass-only stochastic TRG
- Confidence intervals from TNMC replicas feed into the certification bound
- Optional cost-aware compilation via per-gate entangler budget masks

## Where we are now

Beginning of Stage 1. Building the basic MPO compiler and its test harness.

## Code guidelines

- **quimb** for tensor network operations (MPO, MPS, contraction, compression)
- **qiskit** for test verification (exact unitaries, process fidelity, random Haar unitaries)
- **JAX** for numerics and optimization
- **hypothesis** to parametrize tests across random circuits
- **uv** for package management
- Keep code minimalistic. Use existing libraries. Only write custom code for what no library provides.
- Reference for Riemannian optimization: https://github.com/INMLe/rqcopt-mpo

## Git discipline

- Commits should be clean, scoped, focused, and frequent.
- Each commit does one thing and its message explains why.
- The history should read as a clear narrative of how the project was built.
- Use branches when the development structure calls for it (features, stages, experiments).
