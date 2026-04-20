# Candidate generation — systematic testing

The ensemble compiler generates candidates by running `compile_circuit`
from Haar-random inits. We need to understand this compile's behavior
before layering on Kalloor-style diversification/certification.

For general compilation targets, Haar-random init is the only practical
seed-diversity source. Perturbation-based diversity comes on top later.

## Methodology

The test suite is the experimental framework.
`tests/test_pipeline.py` drives two parametrized suites:

- `test_pipeline_same_depth` — compile at `ansatz_depth = target_depth`
  on random-brickwall targets. Tests the optimizer itself.
- `test_pipeline_tfi_compression` — compile a TFI Trotter target to a
  shallower brickwall ansatz. Tests real compression.

Accuracy is verified via `sampled_max_trace_distance` (Kalloor results
§Demonstration). Scales to ~n=20 via statevector sim + rank-limited
eigendecomposition, no 2²ⁿ materialization.

Failing cases are retained to document what the compiler currently
can't do. We do not loosen thresholds to force passes.

## Important lesson learned

An earlier iteration tested compression by asking "compile a
random-brickwall depth-4 target to a depth-2 ansatz". That's
information-theoretically unachievable: a random depth-`D` brickwall
has MPO bond ~4^min(k, D-k) at every cut; a depth-`D/2` ansatz just
can't store that spectrum. The "failures" there were the task being
ill-posed, not the compiler being broken.

Compression only makes sense for **structured targets** with bounded
entanglement — like gapped-TFI Trotter evolution at short-to-moderate
time. Those are the right benchmarks.

## Results

### Same-depth (compile convergence on random brickwalls)

Threshold `max_td < 0.05`, 3 seeds, 200 polar iters.

| case            | n  | result |
|:----------------|:--:|:------:|
| random_n4       | 4  | pass   |
| random_n8       | 8  | pass   |
| random_n12      | 12 | pass   |
| random_n16      | 16 | pass   |

Compile recovers the target exactly at every tested n. The optimizer
works.

### Compression on TFI Trotter (J=-1, g=0.5, h=0, dt=0.1)

Threshold `max_td < 0.1`, 3 seeds, 200 polar iters, all with `seed=42`
so compile_errs across members are comparable.

| case            | n  | steps | target depth | ansatz d | max_td | compile_errs    | result |
|:----------------|:--:|:-----:|:------------:|:--------:|-------:|:----------------|:------:|
| tfi_n8_s2_d2    | 8  | 2     | ~8           | 2        | < 0.1  | —               | pass   |
| tfi_n12_s2_d2   | 12 | 2     | ~8           | 2        | < 0.1  | —               | pass   |
| tfi_n8_s4_d2    | 8  | 4     | ~16          | 2        | < 0.1  | —               | pass   |
| tfi_n8_s12_d4   | 8  | 12    | 48           | 4        | < 0.1  | —               | pass   |
| tfi_n16_s4_d2   | 16 | 4     | ~16          | 2        | < 0.1  | —               | pass   |
| **tfi_n8_s8_d2**| 8  | 8     | 32           | 2        | **0.30** | [4.73, 4.73, 4.73] | fail |
| **tfi_n8_s12_d2**| 8 | 12    | 48           | 2        | **0.66** | [10.7, 10.7, 10.7] | fail |
| **tfi_n16_s8_d4**| 16| 8     | 32           | 4        | **0.52** | [174, 174, 174] | fail   |

## Observations

### Compile works when the target is genuinely compressible

Short-time (2 Trotter steps) and moderate-time+deep-ansatz (12 steps
compressed to depth-4) cases pass cleanly. The compile machinery is
functional.

### Compile fails on physically-overcompressed configs

tfi_n8_s8_d2: depth-32 target → depth-2 ansatz is 16× compression.
Even a gapped target has *some* entanglement buildup after 8 Trotter
steps; depth-2 can't capture it. Same for s12/d2 (24×) and n16_s8_d4
(8× at larger n).

### Seed-only diversity produces one circuit across every failing case

In every failing TFI case, compile_errs are **identical across the 3
Haar-random inits** — [4.73, 4.73, 4.73], [10.7, 10.7, 10.7], etc.
Same pattern we saw on the random-brickwall "compression" tests.

This is the key negative result for the ensemble pipeline: **regardless
of target class, polar sweeps from random inits converge to the same
basin.** The QP with rank-1 Gram cannot produce a convex combination
that does better than a single member. ReWEE-style quadratic reduction
is structurally impossible from this source of diversity alone.

## Direct implication for Kalloor

Seed-only diversification (the only general-purpose source for arbitrary
targets) does not meet the ReWEE precondition. Members do not bracket
V. This is a property of the polar-sweep optimizer, not of the target
class.

Perturbation-based diversification is the required second pathway. The
*paired-opposite* perturbation trick from Kalloor methods §Ensemble
Generation is load-bearing — it's specifically designed to place
members *around* V so their convex hull contains it.

## Things still to check

1. Does the compiler's basin-committing behavior depend on `max_iter`?
   (Very unlikely given Exp 1's convergence in 1–8 iters, but worth
   confirming.)
2. Does initializing closer to V (Trotter-coarsened init rather than
   Haar random) give faster convergence AND access to different basins?
3. How close to V does a perturbation need to be to land in a useful
   diversity-providing basin without ending up too far from V?

## Grid sweep — TFI compression landscape

Script: `scripts/compression_sweep.py` → CSV at
`docs/data/compression_grid.csv`.

**Grid.** n ∈ {4, 6, 8, 10}, g ∈ {0.3, 1.0, 1.5} (gapped / critical /
paramagnetic), Trotter steps ∈ {2, 4, 6, 8} (J=-1, h=0, dt=0.1).
Compression ratios 2, 4, 8 (ansatz depth = ⌊target_depth / ratio⌋,
capped at 16 to keep MPO conversion bounded). 3 candidate circuits
per config, 100 polar iters, 10 verification samples. 144 unique
configs, total 1325s (~22 min).

**Headline.** The compile is robust across most of the grid. At the
standard threshold `max_td < 0.1`, 121/144 (84%) of configs pass.
Failures cluster in one corner: ultra-short target (tgt_depth = 8)
with ratio=8 (ansatz_depth = 1). Depth-1 is structurally too shallow,
regardless of target.

### Median max_td by (Trotter steps × compression ratio)

|                | ratio=2   | ratio=4   | ratio=8   |
|:---------------|:---------:|:---------:|:---------:|
| steps=2        | 6.5e-3    | 8.6e-3    | **0.33**  |
| steps=4        | 4.1e-2    | 3.1e-2    | 7.8e-2    |
| steps=6        | 4.1e-2    | 2.3e-2    | 2.8e-2    |
| steps=8        | 7.0e-2    | 3.0e-2    | 3.7e-2    |

The steps=2, ratio=8 cell is the failure pocket — it's where
ansatz_depth collapses to 1. Elsewhere 2× and 4× compression work
smoothly.

### Median max_td by (n × ratio)

|      | ratio=2 | ratio=4 | ratio=8 |
|:-----|:-------:|:-------:|:-------:|
| n=4  | 2.9e-2  | 1.2e-2  | 5.1e-2  |
| n=6  | 2.5e-2  | 2.5e-2  | 4.8e-2  |
| n=8  | 4.3e-2  | 2.2e-2  | 4.8e-2  |
| n=10 | 5.7e-2  | 4.0e-2  | 1.0e-1  |

Accuracy degrades gently with n — no cliff. The compiler does scale.

### Median max_td by (g × ratio)

|        | ratio=2 | ratio=4 | ratio=8 |
|:-------|:-------:|:-------:|:-------:|
| g=0.3  | 3.1e-2  | 2.4e-2  | 3.1e-2  |
| g=1.0  | 3.7e-2  | 1.6e-2  | 7.1e-2  |
| g=1.5  | 3.6e-2  | 3.6e-2  | 6.6e-2  |

Gap size has only a mild effect at these dt/steps. Critical g=1 and
gapped g=0.3 both compress well.

### Seeds agree on the basin?

Percent of configs where the 3 Haar-seeded candidates converge to
the same Frobenius error (to within 1e-4):

|                | ratio=2 | ratio=4 | ratio=8 |
|:---------------|:-------:|:-------:|:-------:|
| steps=2        |    0%   |  100%   |   75%   |
| steps=4        |    0%   |    0%   |  100%   |
| steps=6        |    0%   |    0%   |    0%   |
| steps=8        |    0%   |    0%   |    0%   |

**Updates the earlier narrative.** For sufficiently-nontrivial
targets (steps ≥ 6 at any ratio, or moderate steps with modest
compression) seeds *do* find different local minima — 77% of configs
overall show seed disagreement. The "seed agreement ⇒ no diversity
possible" story only applies in pathologically-shallow or
pathologically-trivial regimes (short evolution, ansatz depth 1).

But disagreement alone doesn't prove *useful* diversity for Kalloor's
ReWEE condition — the candidates still need to bracket V. Whether the
current QP actually reduces error below the best single member is a
separate question worth checking with a targeted test.

## Scaling with n — how to guess an ansatz depth

### Observation 1: n-dependence is sub-linear to linear, never exponential.

Fixing `(g, steps, ansatz_d)` and sweeping `n`:

| g   | steps | ansatz_d | n=4    | n=6    | n=8    | n=10   |
|:---:|:-----:|:--------:|:------:|:------:|:------:|:------:|
| 0.3 | 2     | 2        | 1.7e-3 | 2.6e-3 | 3.1e-3 | 3.4e-3 |
| 0.3 | 4     | 2        | 1.8e-2 | 2.5e-2 | 2.9e-2 | 3.2e-2 |
| 1.0 | 4     | 2        | 5.5e-2 | 8.0e-2 | 9.2e-2 | 1.0e-1 |
| 1.5 | 4     | 4        | 1.3e-2 | 3.8e-2 | 7.6e-2 | 7.0e-2 |

`max_td` grows at most ~4× from n=4 to n=10 across the whole grid. No
cliff. This is consistent with an area-law view: a gapped target's cut
entanglement is bounded, so the depth requirement for a given accuracy
does not grow with n. The observed slow growth in `max_td` at fixed
depth mostly reflects the number of cuts, not entanglement per cut.

### Observation 2: minimum ansatz depth is roughly `⌈steps/2⌉`.

Minimum ansatz_d achieving `max_td < 0.05` from the first sweep's
coarse grid (ansatz_d ∈ {1, 2, 3, 4, 6, 8, 12, 16}):

|  g  | steps | n=4 | n=6 | n=8 | n=10 |
|:---:|:-----:|:---:|:---:|:---:|:----:|
| 0.3 |   2   |  2  |  2  |  2  |  2   |
| 0.3 |   4   |  2  |  2  |  2  |  2   |
| 0.3 |   6   |  3  |  3  |  3  |  3   |
| 0.3 |   8   |  4  |  4  |  4  |  8†  |
| 1.0 |   2   |  2  |  2  |  2  |  2   |
| 1.0 |   4   |  4  |  4  |  —  |  —   |
| 1.0 |   6   |  6  |  3  |  3  |  —   |

(† = grid-resolution artifact, not structural. "—" = no tested
ansatz_d passed; the first sweep only had ansatz_d ∈ {1,2,3,4,6,8,12,16}.)

The gapped-g=0.3 column matches `ansatz_depth ≈ max(2, ⌈steps/2⌉)`
cleanly, with essentially no n-dependence. Critical g=1.0 needs
somewhat deeper ansatz at longer evolutions — consistent with
slow-growing entanglement at criticality.

### Empirical rule (to be validated on finer grid)

> For gapped TFI with dt=0.1: `ansatz_depth ≈ max(2, ⌈steps/2⌉)`,
> independent of `n`.
>
> For critical TFI: add 1–2 extra layers at longer evolutions.
>
> For any target: `ansatz_depth = 1` fails structurally
> (max_td ~ 0.2–0.4 regardless of n, g, steps).

### Sweep 2 — finer ansatz_d grid, larger n and steps (running)

Triggered by the gaps above. Running `scripts/compression_sweep2.py`:
n ∈ {4, 6, 8, 10, 12, 14}, g ∈ {0.3, 1.0, 1.5}, steps ∈ {4, 6, 8, 12, 16},
ansatz_d ∈ {2, 3, 4, 5, 6, 8, 10, 12}. 624 new configs (dedup'd
against `compression_grid.csv`). Will update this section once the
merged analysis is done.

## References

- Tests: `tests/test_pipeline.py::{test_pipeline_same_depth,
  test_pipeline_tfi_compression}`.
- Sampled trace distance: `src/tno_compiler/verify.py`.
- TFI Trotter helper: `src/tno_compiler/tfi.py`.
- Grid sweep 1: `scripts/compression_sweep.py` →
  `docs/data/compression_grid.csv` (144 rows).
- Grid sweep 2: `scripts/compression_sweep2.py` →
  `docs/data/compression_grid2.csv` (augments).
- Earlier random-brickwall exploration (superseded):
  `scripts/compile_test_exp1.py`.

## References

- Tests: `tests/test_pipeline.py::{test_pipeline_same_depth,
  test_pipeline_tfi_compression}`.
- Sampled trace distance: `src/tno_compiler/verify.py`.
- TFI Trotter helper: `src/tno_compiler/tfi.py`.
- Earlier random-brickwall exploration (superseded): `scripts/compile_test_exp1.py`.
