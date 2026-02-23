# %%
import torch
from torch.distributions import MultivariateNormal, Normal

from sbi.neural_nets.estimators.tabpfn_flow import TabPFNFlow
from tests.mini_sbibm.two_moons import TwoMoons
import matplotlib.pyplot as plt

torch.manual_seed(0)

two_moons_task = TwoMoons()
prior = two_moons_task.get_prior()
simulator = two_moons_task.get_simulator()

theta_samples = prior.sample((48000,))
sims = simulator(theta_samples)

print(theta_samples.shape)
print(sims.shape)

obs1 = two_moons_task.get_observation(1)
obs2 = two_moons_task.get_observation(2)
obs3 = two_moons_task.get_observation(3)

stack_obs = torch.cat([obs1, obs2, obs3], dim=0)
print(stack_obs.shape)


# %%
two_moons_estimator = TabPFNFlow(
    input_shape=torch.Size([2]),
    condition_shape=torch.Size([2]),
)

two_moons_estimator.set_context(theta_samples, sims)


# %%
samples = two_moons_estimator.sample((1000,), stack_obs)
print(samples.shape)

# %%
plt.scatter(samples[:, 0, 0], samples[:, 0, 1])
plt.scatter(samples[:, 1, 0], samples[:, 1, 1])
plt.scatter(samples[:, 2, 0], samples[:, 2, 1])
plt.show()

# %%

grid_size = 50
theta_1 = torch.linspace(-1.0, 1.0, grid_size)
theta_2 = torch.linspace(-1.0, 1.0, grid_size)
grid_1, grid_2 = torch.meshgrid(theta_1, theta_2, indexing="ij")
grid_points = torch.stack([grid_1.reshape(-1), grid_2.reshape(-1)], dim=-1)

# Evaluate all 3 observations in one call.
grid_input = grid_points.unsqueeze(1).expand(-1, stack_obs.shape[0], -1)
print(grid_input.shape)
grid_prob = two_moons_estimator.log_prob(grid_input, stack_obs).exp()

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

for obs_idx, ax in enumerate(axes):
    heatmap = grid_prob[:, obs_idx].reshape(grid_size, grid_size).cpu()
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[-1, 1, -1, 1],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(f"Estimated p(theta | x_{obs_idx + 1})")
    ax.set_xlabel("theta[0]")
    ax.set_ylabel("theta[1]")
    fig.colorbar(image, ax=ax)

plt.show()


# %%
from sbi.inference.trainers.npe.npe_pfn import NPE_PFN
from sbi.neural_nets import posterior_nn

estimator = posterior_nn("tabpfn", z_score_theta="none", z_score_x="none")
# Given: parameters theta and corresponding simulations x
inference = NPE_PFN(prior=prior, density_estimator=estimator)
inference.append_simulations(theta_samples, sims)
posterior = inference.build_posterior(
    sample_with="direct",
    # filtered_direct_sampling_parameters={"filter_size": 1000}
    direct_sampling_parameters={"max_sampling_batch_size": 1000},
)

# %%
f_samples = posterior.sample((1000,), x=obs1)

# %%
plt.scatter(f_samples[:, 0], f_samples[:, 1])
plt.show()

# %%
print(posterior)
