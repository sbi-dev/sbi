from torch import ones, zeros, eye
from torch.distributions import MultivariateNormal
from sbi.inference import SNPE_C
from sbi.simulators.linear_gaussian import linear_gaussian


#%% parameters

num_dim = 2

x_o = zeros(1, num_dim)
num_samples = 1000

prior = MultivariateNormal(loc=zeros(num_dim), covariance_matrix=eye(num_dim))

likelihood_shift = -1.0 * ones(num_dim)
likelihood_cov = 0.3 * eye(num_dim)
simulator = lambda theta: linear_gaussian(theta, likelihood_shift, likelihood_cov)

infer = SNPE_C(
    simulator=simulator,
    density_estimator=None,  # Use default MAF.
    prior=prior,
    z_score_x=True,
    simulation_batch_size=10,
    retrain_from_scratch_each_round=False,
    discard_prior_samples=False,
    show_progressbar=False,
    sample_with_mcmc=False,
)

posterior = infer(num_rounds=1, num_simulations_per_round=2000)  # type: ignore
posterior.freeze(x_o=x_o)
samples = posterior.sample(num_samples, x=x_o)


# %%
