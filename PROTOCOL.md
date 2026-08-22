# Reproducibility Protocol

The concrete parameters behind every number the paper reports: data generators,
teacher constants, model configuration, and training. It is the record the
paper's availability statement points at, and it accompanies the code in this
repository: the executor implementation, the trajectory-recording code for the
three teachers, and the evaluation harness.

## Data

The community family uses the LFR benchmark (Lancichinetti et al., 2008) with
mean degree 4.5, maximum degree 10, mixing parameter 0.1, and community sizes
4–12, generated under seed 12345. The scale-free family is Barabási–Albert with
m = 3. The size-extrapolation corpus builds each family sparsely at
N = 10³–10⁶: a √N × √N grid, a random-geometric graph with radius set for
average degree 4, Barabási–Albert m = 3, and Erdős–Rényi p = 4/N with a
Hamiltonian-chain backbone for connectivity.

The training split is 5000 LFR graphs at N = 20–50, seeded 4000/500/500.

## Teachers

The three teachers share one state-space convention: grid initialization in
[-1, 1]², ideal length k = √(A/N) with A = 1, initial temperature
t₀ = max(0.1 × max coord-span, 10⁻⁴), linear cooling with step t₀/(T+1) to
floor 10⁻⁶, and a displacement clip of 0.01.

- **Fruchterman–Reingold.** The FR step mirrors networkx `spring_layout`
  exactly, verified to a maximum deviation below 10⁻⁴.
- **ForceAtlas2-like.** Keeps Gephi's defaults k_r = 2.0, k_g = 1.0 with linear
  attraction, but substitutes the shared cooling schedule for Gephi's adaptive
  global-speed control.
- **Kamada–Kawai-objective.** Executed as steepest descent on its own stress
  objective, Σ_{i<j} w_ij (‖p_i − p_j‖ − d_ij)² with d_ij = k · dist(i, j) and
  w_ij = 1/d_ij², stepped under the shared schedule — a substitution of solution
  method rather than of the reference implementation.

Both substitutions remove state that depends on the iteration history, which is
what makes (G, X_t, τ_t) a complete description of the update. The manuscript
states this in §V-A and returns to it in §VI.

## Model

The equivariant executor of Satorras et al. (2021): a linear node encoder into
width 64; two message-passing layers of the same shape over E (attraction) and
the rebuilt geometric kNN (repulsion) with k_geo = 10 and the brute-force
backend; a combine layer on the concatenation of the node, attraction, and
repulsion states; and a two-layer MLP readout φ_θ whose input dimension is
2·64 + 1 + 2 (distance, two membership flags) + 1 for τ_t + 64 for the learned
teacher embedding c, plus 1 for the stride input in the compressed variant.

The structural encoder uses LapPE width 10 through a sign-invariant network and,
at training scale, RRWP width 16; at N ≥ 10⁵ only the LapPE term is kept.

Parameter counts measured from the checkpoints — the fidelity rows are not
ordered by capacity:

| Model | Parameters |
|---|---|
| Φ-layouter | 76,289 (76,417 multi-teacher) |
| Unaligned MPNN | 67,970 |
| Triplet-GMPNN | 117,314 |
| G-ForgetNet | 104,770 |

## Training

One graph per optimizer step with all recorded steps summed into the loss; Adam
at η = 10⁻³, weight decay 10⁻⁵, gradient clipping at 1, and 50 epochs (~215 s
per epoch) on one RTX 4090 over the seed-42 80/10/10 split.

The multi-teacher model interleaves the three teachers' training splits by a
seed-42 shuffle. The compressed model samples a stride uniformly from
{1, 2, 4, 8} per graph per epoch.

Trajectories are teacher-forced during training and evaluated by autoregressive
rollout. The extended-rollout temperature clamps to the schedule floor, the same
10⁻⁶ used inside the window.

Fidelity is reported as mean ± standard deviation over three training seeds
(42, 7, 2024); Triplet-GMPNN is over two (42, 7).

## Rollout classification

Extended rollouts are classified by mean per-step node displacement over their
second half:

- **converged** — below half the first-half mean, or below 10⁻³;
- **diverged** — positions exceed 10⁴ or become non-finite;
- **oscillating** — otherwise.
