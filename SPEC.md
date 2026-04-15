# Quantum Circuit Compilation via Stochastic Tensor Renormalization

**Status:** This document supersedes all prior versions.

---

## 1. Problem

**Given:**
- A target unitary $V \in U(2^n)$ on $n$ qubits, specified as a
  tensor network operator (TNO)
- A circuit ansatz $U(\theta)$ on the same qubits: a sequence of
  2-qubit gates $G_1, \ldots, G_{|\mathcal{G}|}$ on a hardware graph,
  at depth $D$
- An error budget $\delta > 0$

**Produce:**
- A weighted ensemble $\{(p_i, U_i)\}$ of circuits such that
  $\|\sum_i p_i U_i \rho U_i^\dagger - V\rho V^\dagger\|_\diamond
  \leq \delta$

**Runtime:** Each shot, sample circuit $i$ with probability $p_i$,
execute, measure.

---

## 2. The Closed Tensor Network

The overlap between target and circuit is

$$\mathrm{Tr}(V^\dagger U) = \text{contraction of a closed tensor network}$$

The structure of this closed TN depends on how $V$ is specified.

**Layered targets (primary case).** If $V$ is itself a product of
2-qubit gate layers (e.g., a Trotterized Hamiltonian propagator), the
closed TN is a planar strip of width $n$ and depth $D_{\mathrm{tot}}
= B + D$ (target depth + circuit depth), with $N_0 \approx
nD_{\mathrm{tot}}/2$ tensors of bond dimension $\chi_0 = 2$. This is
the geometry assumed in the cost analysis of §4.

**General TNO targets.** If $V$ is a TNO with virtual bond dimension
$\chi_V$ on a graph $\Lambda_V$ that differs from the hardware graph,
the closed TN has a combined graph structure inherited from both $V$
and $U$. The local bond dimensions are inherited from $\chi_V$ and
$\chi_0 = 2$, and the tensor count and topology depend on the
specific TNO. The algorithm applies unchanged; only the cost formulas
must be adjusted for the specific closed TN geometry.

In either case, the closed TN is at least 2D: the spatial dimension
of the hardware crossed with the circuit depth. For 2D hardware it is
quasi-3D.

The algorithm handles the 2D case natively. This covers all circuits
on 1D and quasi-1D hardware at any depth, which is the most
practically relevant regime on current and near-term devices.

---

## 3. The Algorithm

### One Optimization Sweep

The compiler repeats the following:

**Forward pass (stochastic TRG).**

Apply tensor renormalization group coarse-graining to the closed TN
$\mathcal{N} = \mathrm{Tr}(V^\dagger U)$. At each TRG level $k$:

1. Block $2 \times 2$ patches of tensors. The intermediate bond
   dimension is $d_k^2$ (product of two incoming legs).
2. Compute the full SVD of the $d_k^2 \times d_k^2$ matrix across
   each cut.
3. Instead of keeping the top $d_{k+1}$ singular values
   (deterministic truncation), **Ferris-sample** a size-$d_{k+1}$
   subset of the $d_k^2$ singular values. The joint probability of
   selecting subset $\mathbf{i} = \{i_1, \ldots, i_{d_{k+1}}\}$ is
   proportional to $\prod_j S_{i_j}^2$ (Fisher's noncentral
   multivariate hypergeometric distribution). Each retained value is
   rescaled by $1/r(i)$ where $r(i)$ is the marginal inclusion
   probability, computed from the MPS form of the sampling
   distribution (Ferris 2015, Appendix). This rescaling ensures the
   projector identity.
4. Store the coarse-grained tensors and the projector choices.

After $K = \frac{1}{2}\log_2 N_0$ levels, $O(1)$ tensors remain and
are contracted exactly to a scalar $\hat{z}(\omega)$.

The Ferris projector identity guarantees

$$\mathbb{E}_\omega[W_L(\omega) W_R(\omega)^\dagger] = I$$

at every bond, for any choice of sampling basis and probabilities,
so long as every singular value has nonzero inclusion probability
(Ferris 2015, Eq. 5). Multilinearity of the contraction and
independence across bonds then gives
$\mathbb{E}[\hat{z}(\omega)] = \mathrm{Tr}(V^\dagger U) / 2^n$
exactly.

**Backward pass (∂TRG).**

Reverse the TRG hierarchy. Starting from the root, propagate
environments downward: at each level, contract the top environment
with stored sibling tensors and stored (stochastic) projectors from
the forward pass.

At level 0: for each circuit gate $\ell$, the result is a stochastic
gate environment $E_\ell(\omega) \in \mathbb{C}^{4 \times 4}$.

The backward pass is reverse-mode automatic differentiation through
the TRG computational graph (Chen et al. 2019, Supplemental §E). Its
cost equals the forward pass cost.

Because $G_\ell$ enters the closed TN multilinearly, the environment
$E_\ell(\omega)$ is the TNMC contraction of the open network
$\mathcal{N} \setminus G_\ell$ — the closed TN with $G_\ell$ removed
and its legs left open. By the Ferris identity applied to this
sub-network:

$$\mathbb{E}_\omega[E_\ell(\omega)] = E_\ell^{\mathrm{exact}}$$

The projectors $\omega$ were computed from spectra of the full
network (including $G_\ell$), so they are suboptimally
importance-sampled for $\mathcal{N} \setminus G_\ell$. This affects
variance, not expectation.

**Variance reduction.** Average $P$ independent forward-backward
passes (different Ferris random seeds, same circuit):
$\bar{E}_\ell = (1/P) \sum_{p=1}^P E_\ell(\omega_p)$. Each pass
is independent and embarrassingly parallel.

**Gate update.** For each gate $\ell$, update:

$$G_\ell^{\mathrm{new}} = \arg\max_{G \in \mathcal{A}} \mathrm{Re}\,\mathrm{Tr}(\bar{E}_\ell^\dagger G)$$

For unrestricted $\mathcal{A} = U(4)$: polar decomposition of
$\bar{E}_\ell$. Closed-form, $O(1)$ per gate.

For restricted $\mathcal{A}_{\leq b}$ (at most $b$ native
entanglers): search over strata by alternating maximization. Still
$O(1)$ per gate.

### Ensemble Collection

**From stochastic trajectory.** The stochasticity of the Ferris
projectors makes the optimization trajectory wander. After
convergence to the vicinity of a local minimum, each sweep produces
a slightly different circuit due to the different stochastic
environments. These are approximate samples from the neighborhood of
the optimum. Whether the resulting diversity is sufficient for the
Kalloor QP is an empirical question (see §9).

Collect circuit snapshots every $\tau_{\mathrm{auto}}$ sweeps
(where $\tau_{\mathrm{auto}}$ is the autocorrelation time of the
trajectory, measured empirically).

**From independent compilations (optional).** For additional
diversity — especially cost diversity — run $M_{\mathrm{ind}}$
independent compilations from different initializations, different
Ferris seeds, or different per-gate budget masks. Each converges to
a potentially different local minimum. This supplements the
trajectory snapshots.

After $M$ total circuit candidates (from either or both sources):

**Overlap estimation.** For each pair $(i,j)$, the overlap
$\mathrm{Tr}(U_i^\dagger U_j) / 2^n$ is a 2D closed TN contraction
— estimated by forward-pass-only stochastic TRG. Same for the target
overlaps $|\mathrm{Tr}(V^\dagger U_i)| / 2^n$. Total: $O(M^2)$
independent stochastic contractions, all parallel.

**Phase alignment.** Phase-align each $U_i$ against $V$: set
$\phi_i^* = -\arg(\hat{z}_i)$ and replace $U_i \leftarrow
e^{i\phi_i^*} U_i$. The phase-aligned target overlap is
$\hat{c}_i = |\hat{z}_i|$. The pairwise overlaps must also be
rotated to the same gauge:

$$\hat{G}_{ij}^{(\mathrm{align})} = \mathrm{Re}\!\left(e^{i(\phi_j^* - \phi_i^*)} \frac{\mathrm{Tr}(U_i^\dagger U_j)}{2^n}\right)$$

If this rotation is omitted, the Gram matrix and the target overlaps
refer to different representatives of the channel-equivalent
unitaries, and the QP and certification are inconsistent.

**PSD repair (for optimization only).** The phase-aligned Gram matrix
$\hat{G}^{(\mathrm{align})}$ may fail to be PSD due to TNMC noise.
Symmetrize and project to nearest PSD (clip negative eigenvalues) to
obtain $\tilde{G}$. Use $\tilde{G}$ in the QP (which requires
convexity). Do NOT use $\tilde{G}$ for certification — certify from
the raw phase-consistent overlaps plus confidence margins (see below).

**Kalloor QP.** Solve:

$$p^* = \arg\min_{\substack{p \geq 0,\; \sum_i p_i = 1}} J_\lambda(p) = p^\top \tilde{G} p - 2\hat{c}^\top p + \lambda C^\top p$$

where $C_i$ is the entangler count of circuit $i$ and $\lambda \geq 0$
trades accuracy against hardware cost.

For unrestricted compilation ($\mathcal{A} = U(4)$, all gates free):
every snapshot has the same gate structure, so $C_i = C$ for all $i$
and the cost term is constant. Set $\lambda = 0$; the QP reduces to
a pure accuracy optimization.

For cost-aware compilation: augment the trajectory snapshots with
circuits compiled under different per-gate entangler budgets. Run
$M_{\mathrm{mask}}$ independent compilations with randomly sampled
budget masks $b_{i,\ell} \in \{0, 1, 2, 3\}$ per gate position,
where each gate is restricted to $\mathcal{A}_{\leq b_{i,\ell}}$.
These produce circuits with genuinely different entangler counts
$C_i = \sum_\ell b_{i,\ell}$. The QP then selects the ensemble that
best trades accuracy against hardware cost. This is a supplementary
mechanism, not the core algorithm.

**Certification.** Certification uses the raw phase-consistent
overlaps $\hat{G}^{(\mathrm{align})}$ and $\hat{c}$, NOT the
PSD-repaired $\tilde{G}$.

*Support pruning.* Before certifying, discard any snapshot $i$ with
$\hat{c}_i < c_{\min}$ (individual accuracy too poor for the
pointwise bound). Re-solve the QP over the remaining support.

*Confidence bookkeeping.* Each overlap estimate $\hat{G}_{ij}$ and
$\hat{c}_i$ has a standard error from $P_{\mathrm{ov}}$ independent
TNMC replicas. With failure probability budget $\eta$, allocate
$\eta / (M^2 + M)$ per estimate via a union bound. Compute one-sided
confidence intervals for each overlap, yielding a confidence margin
$\epsilon_{\mathrm{conf}}$.

*Mean condition.* Evaluate the certification objective using raw
phase-consistent overlaps:

$$q(p^*) = (p^*)^\top \hat{G}^{(\mathrm{align})} p^* - 2\hat{c}^\top p^* + 1$$

$$\delta_{\mathrm{ens}} = \sqrt{2^n \cdot \max(q(p^*), 0)} + \epsilon_{\mathrm{conf}}$$

*Pointwise condition.*

$$R = \max_{i:\, p_i^* > 0} \sqrt{2 \cdot 2^n(1 - \hat{c}_i)} + \epsilon_{\mathrm{conf}}$$

*Diamond-norm bound* (Campbell 2017, Hastings 2017):

$$\|\bar{\mathcal{E}} - \mathcal{V}\|_\diamond \leq 2\delta_{\mathrm{ens}} + R^2$$

This holds with probability $\geq 1 - \eta$ over the TNMC estimation
noise.

---

## 4. Cost

### Per-sweep cost

The closed TN has $N_0 \approx n D_{\mathrm{tot}} / 2$ tensors. TRG
produces $K = \frac{1}{2}\log_2 N_0$ levels. At level $k$:

- Blocking operations: $N_0 / 4^{k+1}$
- SVD per operation: $O(d_k^6)$ (Levin-Nave TRG)
- Level cost: $(N_0 / 4^{k+1}) \cdot O(d_k^6)$

Total forward pass: $\sum_{k=0}^K (N_0 / 4^{k+1}) \cdot d_k^6$.
Backward pass: same. Gate updates: $O(N_0)$. One sweep with $P$
replicas: $P \times 2 \sum_k (N_0 / 4^{k+1}) \cdot d_k^6$.

### What controls $d_k$

At TRG level $k$, each tensor represents a $2^k \times 2^k$ block
of the original closed TN. The Ferris sampling dimension $d_{k+1}$
at that level must be large enough to capture the significant part
of the Schmidt spectrum across the level-$k$ cut.

For gapped targets with correlation length $\xi$: once $2^k > \xi$,
the spectrum saturates. $d_k \to d^*$ independent of system size.
Total cost: $O(N_0 \cdot (d^*)^6) = O(n D \cdot (d^*)^6)$.

For critical targets: $d_k$ grows with $k$. The top level dominates.
At the top, each tensor represents a block of $\sqrt{N_0} \times
\sqrt{N_0}$ sites. The required $d_K$ depends on the entanglement
across a cut of this block. For 1D area-law systems:
$d_K \sim \sqrt{N_0} \cdot \mathrm{polylog}(N_0)$. Top-level cost:
$O(d_K^6)$. Total: polynomial in $\sqrt{N_0} = \sqrt{nD}$.

### Variance

The variance of the Ferris estimator across all levels is:

$$\log(\mathrm{rel.\, var.}) \approx \sum_{k=0}^K \frac{N_0}{4^{k+1}} \cdot e^{-\alpha_k d_{k+1}}$$

where $\alpha_k$ is the spectral decay rate at level $k$. Choosing
$d_{k+1}$ to make each term $O(1/K)$ controls the total variance.
This is the SAME $d_k$ schedule that controls the contraction cost.
Variance and cost are not separate concerns — they are the same
bottleneck.

### Concrete estimate

$n = 80$ qubits, 1D brickwall, depth $D = 40$, gapped target with
$\xi = 5$, saturation dimension $d^* \approx 50$.

- Forward + backward pass: $\sim 10^{11}$ FLOPS
- A100 GPU: $\sim 10$ ms per sweep
- 100 sweeps to convergence: $\sim 1$ second
- 50 ensemble members at 500 sweeps: $\sim 10$ seconds
- $O(M^2)$ overlaps: $\sim 2$ seconds
- Total compilation: $\sim 12$ seconds

### Comparison

| Method | Bottleneck | Cost for $n = 80$, $D = 40$ |
|---|---|---|
| MPO compilation (sweep along $n$) | Bond dim $4^D = 4^{40}$ | Impossible |
| MPO compilation (sweep along $D$) | Bond dim $4^n = 4^{80}$ | Impossible |
| VQE-style parameter shift | Barren plateaus | Exponentially slow |
| This method (stochastic TRG) | $d^*$ from entanglement | Seconds |

---

## 5. Expected: No Barren Plateaus (*hypothesis, not proved*)

The ∂TRG backward pass gives each gate a DIRECT $4 \times 4$
environment matrix — the partial derivative of the overlap with
respect to that gate's tensor. This is not a gradient estimated by
parameter shifts through an exponentially flat landscape. It is a
concrete matrix extracted from the stored TRG hierarchy by
backpropagation.

The signal-to-noise ratio of the environment is controlled by the
TNMC variance, which is controlled by $d_k$ at each TRG level. For
adequate $d_k$, the environment is expected to be an accurate estimate
of the true gradient direction. The noise comes from the stochastic
projectors, not from an exponentially small signal.

This argument is structural, not a theorem. It has not been verified
that the environment signal remains polynomially large for all
targets at all system sizes. Ferris (2015) explicitly warns that
real-time unitary dynamics may be limited by the sign problem, which
would degrade the environment signal even with adequate $d_k$. The
claim must be validated empirically for each target class.

---

## 6. Expected: Global Compilation (*hypothesis, conditioned on cost estimates*)

If the cost estimates in §4 hold — specifically, if $d_k$ saturates
at a manageable $d^*$ for the target of interest — then the
contraction is efficient for the full $n \times D$ closed TN without
block decomposition. In this regime, the backward pass gives each
gate an environment conditioned on every other gate in the circuit.
This would enable the compiler to discover global compressions —
cooperative rotations across distant gates — that are invisible to
any block-decomposition approach.

The error accounting would simplify: no composition penalty from
$K$ blocks. The compiler would directly optimize
$\mathrm{Re}\,\mathrm{Tr}(V^\dagger U)$ for the entire circuit,
end to end.

Block decomposition remains available as a fallback if the full
closed TN is too expensive to contract globally. In that case, the
algorithm reduces to the block-wise compiler described in prior
versions of this spec, with the triangle inequality for error
composition across blocks.

---

## 7. Correctness

**Lemma A (Ferris unbiasedness).** *Rigorous.* For any closed TN
and any TNMC sampling dimension $d \geq 1$:
$\mathbb{E}[\hat{z}(\omega)] = \mathrm{contract}(\mathcal{N})$.
Source: Ferris 2015, Eq. 5.

**Lemma B (environment unbiasedness).** *Supported.* For each gate
$\ell$: $\mathbb{E}[E_\ell(\omega)] = E_\ell^{\mathrm{exact}}$.
Follows from Lemma A applied to the open sub-network
$\mathcal{N} \setminus G_\ell$. The Ferris identity holds for any
network the projectors are applied to, including one different from
the network they were computed for.

**Lemma C (sweep convergence).** *Empirical.* The deterministic polar
sweep converges rapidly from Trotter initialization (Lin 2021,
McKeever-Lubasch 2023, Gibbs-Cincio 2025). The stochastic version
adds unbiased noise controlled by $P$. Convergence is expected but
not proved for finite $P$. Asymptotic consistency holds as
$P \to \infty$.

**Lemma D (Kalloor bound).** *Rigorous.* For unitary target $V$ and
unitary ensemble members $U_i$: if $\|\sum_i p_i U_i - V\|_{op}
\leq \delta_{\mathrm{ens}}$ and $\|U_i - V\|_{op} \leq R$ for all
$i$ with $p_i > 0$, then $\|\bar{\mathcal{E}} - \mathcal{V}\|_\diamond
\leq 2\delta_{\mathrm{ens}} + R^2$. Source: Campbell 2017, Hastings
2017, Kalloor et al. 2025 Lemma 1.

---

## 8. Applicability

**Native regime:** Any circuit whose closed TN is 2D. This includes
all circuits on 1D hardware at any depth, and constant-depth circuits
on 2D hardware.

**Favorable targets:** Gapped Hamiltonians, shallow circuits, any
target where the entanglement across cuts of the closed TN is
bounded. The Ferris sampling dimension $d_k$ saturates and the cost
is polynomial.

**Challenging targets:** Critical systems (large $\xi$), deep circuits
on 2D hardware (quasi-3D closed TN). Cost grows with entanglement.
For 2D closed TNs, the cost is polynomial in $\sqrt{N_0}$ (empirical,
supported by the variance analysis in §4 but not proved as a theorem).
For quasi-3D closed TNs, the surface-area law worsens the scaling;
see prior discussion on 3D TRG approaches.

**Hard limit:** The target must be a unitary expressible as a TNO.
The ensemble channel is mixed-unitary (unital) and cannot approximate
non-unital maps. Non-unitary targets require ancilla dilations.

---

## 9. Components

| Component | What it does | Source |
|---|---|---|
| Stochastic TRG | Unbiased contraction of 2D closed TN via hierarchical Ferris sampling | Ferris 2015 + any TRG variant |
| ∂TRG backward pass | Gate environments via reverse-mode AD through stored stochastic hierarchy | Chen et al. 2019 (= Xie et al. 2008 SRG) |
| Polar decomposition | Gate update from environment | Textbook |
| Kalloor ensemble QP | Optimal weights + diamond-norm certification from stochastic trajectory | Kalloor et al. 2025, Campbell 2017, Hastings 2017 |

**What is new:** The stochastic TRG forward pass composed with the
∂TRG backward pass yields unbiased gate environments for circuit
compilation. The same stochasticity that makes the contraction
unbiased is expected to make the optimization trajectory explore
a neighborhood of the local optimum, producing circuit diversity
for the Kalloor QP. Whether this diversity is sufficient (i.e.,
whether the autocorrelation time is short enough) is an empirical
question to be validated in the prototype.

---

## References

1. A. J. Ferris, "Unbiased Monte Carlo for the age of tensor
   networks," arXiv:1507.00767 (2015).
2. B.-B. Chen et al., "Automatic Differentiation for Second
   Renormalization of Tensor Networks," arXiv:1912.02780 (2019).
3. J. Kalloor et al., "Application scale quantum circuit compilation
   with controlled error," arXiv:2510.18000 (2025).
4. E. T. Campbell, "Shorter gate sequences for quantum computing by
   mixing unitaries," PRA 95, 042306 (2017).
5. M. B. Hastings, "Turning gate synthesis errors into incoherent
   errors," QIC 17, 488 (2017).
6. J. Gibbs and Ł. Cincio, "Deep circuit compression for quantum
   dynamics via tensor networks," Quantum 9, 1789 (2025).
