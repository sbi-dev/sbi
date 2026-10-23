# Conditional Mean Flows (Arruda, Hasenauer, Radev 2026): paper summary

**Paper:** Jonas Arruda, Jan Hasenauer, Stefan T. Radev. *Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference.* arXiv:2610.21657 (v1, 15 Oct 2026). Univ. Bonn and RPI.

**Code:** MIT, `https://anonymous.4open.science/r/MeanFlow-AISTAST`. The page needs JavaScript; the file API works: `https://anonymous.4open.science/api/repo/MeanFlow-AISTAST/files/?path=<dir>` lists a directory, `.../file/<path>` returns a file. Python 3.14, JAX (CUDA 12), Keras 3 on JAX, BayesFlow from its `dev` branch, PyMC 5.28.5, nutpie 0.16.11.

## The method

Flow matching gives flexible density estimators, but its density needs an ODE solve with the divergence integral (hundreds of network evaluations). That is too slow inside MCMC for NLE. Normalizing flows evaluate in one pass but restrict the architecture. Consistency models sample in few steps but have no density.

A conditional mean flow predicts the **average velocity** of the flow-matching ODE between two times, with $t=0$ the data and $t=1$ the noise:

$$u_\psi(z,s,t,x)\approx\frac{1}{t-s}\int_s^t v(z_\tau,\tau,x)\,d\tau,\qquad \Phi_\psi(z;s,t,x)=z+(t-s)\,u_\psi(z,s,t,x).$$

One network evaluation is a finite-time flow map.

**Training** (paper Eq. 5 to 8, Table S2):

- MeanFlow identity $v(z_s,s,x)=u(z_s,s,t,x)-(t-s)\frac{d}{ds}u(z_s,s,t,x)$, with the total derivative from one Jacobian-vector product with tangent $(\mathrm{sg}[\hat v],1,0)$.
- $\hat v$ comes from an auxiliary velocity head that shares all but the last 2 blocks with $u$.
- Loss $w\|V-(\epsilon-z_0)\|^2+\tfrac1d\|\hat v-(\epsilon-z_0)\|^2$ with adaptive weight $w=\mathrm{sg}[(\|V-(\epsilon-z_0)\|^2+0.1)^{-0.25}]$.
- Time pairs from a power law ($s=u^{1.5}$), 70% of them collapsed to $t=s$.
- Cycle loss (weight 2.5): map $s\to t\to s$ and penalize the round-trip error, with the condition swapped for another sample's in 60% of the batch. In the paper's code the cycle draws **its own** time pair without the collapse to $t=s$.

**Exact density** (Eq. 9): on an EDM schedule $0=t_0<\dots<t_S=1$ ($\rho=7$, $\sigma_{\min}=10^{-3}$),
$\log q^{(S)}(z\mid x)=\log\mathcal N(z^{(S)};0,I)+\sum_{i=1}^S\log|\det\nabla_zG_i(z^{(i-1)})|$, with each full Jacobian from autodiff. Cost $O(d^3)$ per step.

**Sampling** runs the maps backwards. It matches the density only if the maps are invertible; the cycle loss encourages that but does not guarantee it.

**Theory.** Proposition 1: every population minimizer of the unweighted loss recovers the true average velocity, so Eq. 9 is exact for every schedule. Proposition 2: a masked transformer trained with masked losses gives exact marginal densities for any subset of target coordinates (missing data).

**Delayed-acceptance MCMC.** A coarse level (the mean flow at $S_c$ steps) proposes through a subchain of 5 random-walk Metropolis steps; a fine level (the same network's velocity head, 100 fixed Tsit5 steps with the exact divergence) accepts. Gradient-free.

## The paper's experiments and results

All experiments use 10 independent datasets and check calibration with TARP. Two moons is the only NPE experiment; the other three are NLE, where the learned likelihood enters MCMC.

### 1. Two moons: sample and density quality (NPE, Section 4.1)

- Setup: 2-D parameters, bimodal crescent posterior, 50,176 simulations, 5,000 draws per dataset against 32,000 exact reference draws.
- Compared: spline coupling flow, consistency model, flow matching (S Euler steps or adaptive Tsit5), mean flow at S = 1 to 32 with the schedule tuned per S on held-out observations.
- Sample quality (C2ST): mean flow about 0.52 at every S, 0.523 at S=1; normalizing flow 0.543; the consistency model needs S ≥ 4; flow matching needs ≥ 32 Euler steps. Mean flow S=1 samples 23x faster than flow matching with 32 Euler steps and 140x faster than adaptive.
- Density quality (Hellinger on a 512×512 grid): mean flow 0.322 at S=1, 0.169 at S=32; normalizing flow 0.383; the mean flow's own velocity head with the adaptive solver is best at 0.087. Mean-flow mass 1.066 at S=1 and 1.003 at S=32; Euler flow-matching densities have mass up to $10^{40}$.

### 2. Drift-diffusion model: composed likelihood in MCMC (NLE, Section 4.2)

- Setup: Wiener first-passage model, 3 parameters, one trial likelihood on the signed response time, composed over N = 4, 16, 64, 256 trials; NUTS; 50,176 simulations.
- Compared: mean flow at S = 1, 2, 4, 8 against flow matching with 200 Euler steps. **No normalizing-flow baseline.**
- Results: at S=1 the parameter error and the posterior contraction match the 200-step baseline at every N (NRMSE 0.070 vs 0.067 at N=256), at 182 to 216x faster MCMC. More steps do not help: all MCMC convergence failures (8.5% of runs) have S ≥ 4. Every method becomes overconfident at large N (95% coverage 0.60 for S=1 and 0.70 for the baseline at N=256).

### 3. Epidemic renewal model with missing data (NLE, Section 4.3)

- Setup: 128-dimensional parameters, 123 observed time points with 30% missing at random, 8,192 simulations.
- Compared: mean flow with a masked transformer that marginalizes missing points, against a normalizing flow that gets the value −1 for missing entries.
- Results: the mean flow has lower parameter error (0.187 vs 0.199) and lower predictive error (0.060 vs 0.125) and is calibrated (the flow is not). The normalizing flow finishes **22x sooner** (1,624 s vs 36,146 s).

### 4. g-and-k: flow steps against posterior draws (NLE, Section 4.4)

- Setup: bivariate g-and-k with a Gaussian copula, parameters (g, k, ρ), 50 observations per dataset, 100,000 simulations. Every arm requests the same budget of draws times network evaluations per density (8,000 draws per chain at one evaluation), warmup 25% of the request.
- Compared: mean flow at S = 1, 2, 4, 8; a coupling flow; delayed-acceptance MCMC; flow matching with 200 Euler steps outside the budget.
- Results (C2ST): mean flow 0.577 to 0.604 (best 0.577 at S=4); coupling flow 0.679, worse than every mean-flow setting and about 4x slower in wall time than mean flow S=2; flow matching 0.614 and about 50x slower; delayed acceptance about 0.59, 26 to 37x faster than flow matching but slower than NUTS on the mean flow. Under a fixed budget more draws beat more steps; they recommend S=2.

### The paper's stated limitations

- Density cost grows cubically with the dimension; NUTS needs second derivatives of the log-determinant.
- The cycle loss does not guarantee invertibility, so densities may not be exactly normalized.
- Coverage drops for every method at large trial counts.

## How the paper relates to sbi

- **Speed claims are against flow matching.** sbi's NLE uses one-pass flows (default MAF, `sbi/inference/trainers/nle/nle_base.py:107`), so the 200x does not carry over.
- **Baselines.** The paper's normalizing flow is a spline coupling flow (depth 5, widths 256, 256 on g-and-k) trained 200 epochs with early stopping, against 1,000 epochs and a 200-trial Optuna search for the mean flow. sbi's `nsf` is also a spline coupling flow (rational-quadratic coupling with alternating masks), with LU-linear layers and residual conditioners; it is not autoregressive.
- **Fit.** A mean flow is a `ConditionalDensityEstimator`: NLE and NPE train it through `.loss()` and evaluate it through `.log_prob()` without trainer or potential changes. No sbi estimator today offers a flow-matching-style NLE density.
- **Missing data and delayed acceptance** are separate claims; sbi has no masked-transformer marginal estimator and no delayed-acceptance MCMC.
