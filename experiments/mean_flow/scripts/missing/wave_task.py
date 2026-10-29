"""A tractable stand-in for the paper's renewal task: an epidemic-like wave observed
with missing values, with an exact likelihood for every missingness pattern.

Parameters theta ~ N(0, I_4) map to growth rate r = 0.3 exp(0.3 theta_1), peak time
tau = 16 + 4 theta_2, log peak height h = 3 + 0.5 theta_3 and decay rate
d = 0.15 exp(0.3 theta_4). The log-incidence curve on t = 1..32 is

    m_t = h + log 2 - logaddexp(-r (t - tau), d (t - tau)),

observed with stationary Gaussian AR(1) noise (correlation 0.8, sd 0.15). 30% of the
coordinates are missing completely at random, at least one kept (the paper's rule). The
observed coordinates S are N(m_S, Sigma_SS), so the likelihood of any pattern is exact.
Reference posteriors come from adaptive importance sampling (a Gaussian proposal refit
four times from the prior), resampled to equal weights.

Plain NumPy, importable from the paper's environment and from sbi's.
"""

import numpy as np

T = 32
NUM_PARAMS = 4
RHO = 0.8
SIGMA = 0.15
MISSING_RATE = 0.3
MISSING_FILL = -1.0
TIME = np.arange(1, T + 1, dtype=float)
COVARIANCE = SIGMA**2 * RHO ** np.abs(TIME[:, None] - TIME[None, :])
CHOLESKY = np.linalg.cholesky(COVARIANCE)


def curve(theta: np.ndarray) -> np.ndarray:
    """Mean log-incidence, shape `(..., T)`."""
    theta = np.asarray(theta, dtype=float)
    r = 0.3 * np.exp(0.3 * theta[..., 0:1])
    tau = 16 + 4 * theta[..., 1:2]
    h = 3 + 0.5 * theta[..., 2:3]
    d = 0.15 * np.exp(0.3 * theta[..., 3:4])
    u = TIME - tau
    return h + np.log(2.0) - np.logaddexp(-r * u, d * u)


def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Complete observations, shape `(n, T)`."""
    theta = np.atleast_2d(theta)
    noise = rng.standard_normal((theta.shape[0], T)) @ CHOLESKY.T
    return curve(theta) + noise


def present_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    """Missing completely at random, `True` where seen, at least one kept per row."""
    drawn = rng.random((int(n), T))
    present = drawn >= MISSING_RATE
    return present | (drawn >= drawn.max(axis=-1, keepdims=True))


def log_likelihood(theta: np.ndarray, x: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Exact log-likelihood of the present coordinates of one observation `x`, at each
    row of `theta`. Shape `(m,)`."""
    present = np.asarray(present, dtype=bool)
    cov = COVARIANCE[np.ix_(present, present)]
    chol = np.linalg.cholesky(cov)
    residual = np.asarray(x, dtype=float)[present] - curve(theta)[..., present]
    white = np.linalg.solve(chol, residual.T).T
    return (
        -0.5 * np.sum(white**2, axis=-1)
        - np.sum(np.log(np.diag(chol)))
        - 0.5 * present.sum() * np.log(2 * np.pi)
    )


def reference_posterior(
    x: np.ndarray,
    present: np.ndarray,
    n: int,
    rng: np.random.Generator,
    proposals: int = 400_000,
    rounds: int = 4,
) -> tuple[np.ndarray, float]:
    """`n` equally weighted posterior draws and the importance-sampling ESS of the last
    round."""
    mean, cov = np.zeros(NUM_PARAMS), np.eye(NUM_PARAMS)
    for _ in range(rounds):
        draws = rng.multivariate_normal(mean, cov, size=proposals)
        chol = np.linalg.cholesky(cov)
        white = np.linalg.solve(chol, (draws - mean).T).T
        log_proposal = -0.5 * np.sum(white**2, -1) - np.sum(np.log(np.diag(chol)))
        log_prior = -0.5 * np.sum(draws**2, -1)
        log_weight = log_likelihood(draws, x, present) + log_prior - log_proposal
        weight = np.exp(log_weight - log_weight.max())
        weight /= weight.sum()
        mean = weight @ draws
        centered = draws - mean
        cov = 2.0 * (weight[:, None] * centered).T @ centered
        cov += 1e-8 * np.eye(NUM_PARAMS)
    ess = float(1.0 / np.sum(weight**2))
    picks = rng.choice(proposals, size=n, replace=True, p=weight)
    return draws[picks], ess


def datasets(num: int, seed: int = 30_000) -> list[dict]:
    """Evaluation datasets: truth, complete observation and pattern, fixed by seed."""
    out = []
    for index in range(num):
        rng = np.random.default_rng(seed + index)
        theta = rng.standard_normal(NUM_PARAMS)
        x = simulate(theta, rng)[0]
        out.append({"theta": theta, "x": x, "present": present_mask(1, rng)[0]})
    return out
