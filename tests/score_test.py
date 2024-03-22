# %%
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

# adapted from https://anonymous.4open.science/r/diffusions-for-sbi-7675
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from sbi.neural_nets.vf_estimators.score_estimator import (
    VEScoreEstimator,
    VPScoreEstimator,
)


def get_vpdiff_uniform_score(a, b):
    # score of diffused prior: grad_t log prior_t (theta_t)
    #
    # prior_t = int p_{t|0}(theta_t|theta) p(theta)dtheta
    #         = uniform_cst * int_[a,b] p_{t|0}(theta_t|theta) dtheta
    # where p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
    #
    # ---> prior_t: uniform_cst * f_1(theta_t_1) * f_2(theta_t_2)
    # ---> grad log prior_t: (f_1_prime / f_1, f_2_prime / f_2)
    norm = torch.distributions.Normal(
        loc=torch.zeros((1,), device=a.device),
        scale=torch.ones((1,), device=a.device),
        validate_args=False,
    )

    def vpdiff_uniform_score(theta, alpha_t):
        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
        # with _t = theta_0 * scaling_t
        scaling_t = alpha_t**0.5
        sigma_t = 1.0 - alpha_t

        # N(theta_t|mu_t, sigma^2_t) = N(mu_t|theta_t, sigma^2_t)
        # int N(theta_t|mu_t, sigma^2_t) dtheta = int N(mu_t|theta_t, sigma^2_t) dmu_t / scaling_t
        # theta in [a, b] -> mu_t in [a, b] * scaling_t
        f = (
            norm.cdf((b * scaling_t - theta) / sigma_t)
            - norm.cdf((a * scaling_t - theta) / sigma_t)
        ) / scaling_t
        f_prime = (
            -1
            / sigma_t
            * (
                torch.exp(norm.log_prob((b * scaling_t - theta) / sigma_t))
                - torch.exp(norm.log_prob((a * scaling_t - theta) / sigma_t))
            )
            / scaling_t
        )

        # score of diffused prior: grad_t log prior_t (theta_t)
        prior_score_t = f_prime / (f + 1e-6)

        return prior_score_t

    return vpdiff_uniform_score


def get_vpdiff_gaussian_score(mean, cov):
    # score of diffused prior: grad_t log prior_t (theta_t)
    # for Gaussian prior p(theta) = N(theta | mean, cov)

    def vpdiff_gaussian_score(theta, alpha_t):
        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t I)
        # with mu_t = theta * scaling_t
        scaling_t = alpha_t**0.5
        sigma_t = 1.0 - alpha_t

        # from Bishop 2006 (2.115)
        # p_t(theta_t) = int p_{t|0}(theta_t|theta) p(theta)dtheta
        # = N(theta_t | scaling_t * mean, sigma^2_t I + scaling_t^2 * cov)
        loc = scaling_t * mean
        covariance_matrix = (
            sigma_t**2 * torch.eye(theta.shape[-1], device=mean.device)
            + scaling_t**2 * cov
        )

        # grad_theta_t log N(theta_t | loc, cov) = - cov^{-1} * (theta_t - loc)
        prior_score_t = -(theta - loc) @ torch.linalg.inv(covariance_matrix)
        return prior_score_t

    return vpdiff_gaussian_score


class Gaussian_MixtGaussian_mD:
    def __init__(self, dim, rho_min=0.6, rho_max=1.4, device="cpu") -> None:
        """
        Prior: mD Gaussian: theta ~ N(means, diag(scales)).
        Simulator: mD Gaussian: x ~ N(theta, rho * I_m).
        SBI task: infer theta from x.
        """
        self.dim = dim
        self.simulator_base_std = (
            torch.linspace(rho_min, rho_max, dim).to(device) ** 0.5
        )
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim).to(device),
            covariance_matrix=torch.eye(dim).to(device),
            validate_args=False,
        )
        self.device = device

    def simulator(self, theta):
        cp_dist = MultivariateNormal(
            loc=theta[None].repeat(2, 1),
            scale_tril=torch.stack(
                (
                    torch.diag(self.simulator_base_std * (2.25) ** 0.5),
                    torch.diag(self.simulator_base_std / 3),
                ),
                dim=0,
            ),
            validate_args=False,
        )
        samples_x = torch.distributions.MixtureSameFamily(
            component_distribution=cp_dist,
            mixture_distribution=torch.distributions.Categorical(
                probs=torch.ones(2).to(self.device) / 2,
                validate_args=False,
            ),
        ).sample((1,))[0]
        return samples_x

    def true_posterior(self, x_obs):
        equivalent_post_diag_cov_1 = 1 / (1 / (2.25 * self.simulator_base_std**2) + 1)
        equivalent_post_diag_cov_2 = 1 / (9 / self.simulator_base_std**2 + 1)
        log_weights = (
            torch.distributions.Normal(
                loc=torch.zeros_like(x_obs),
                scale=(2.25**0.5) * self.simulator_base_std + 1,
                validate_args=False,
            )
            .log_prob(x_obs)
            .sum(dim=-1),
            torch.distributions.Normal(
                loc=torch.zeros_like(x_obs),
                scale=(1 / 3) * self.simulator_base_std + 1,
                validate_args=False,
            )
            .log_prob(x_obs)
            .sum(dim=-1),
        )

        base_comp = MultivariateNormal(
            loc=torch.stack(
                (
                    equivalent_post_diag_cov_1
                    * x_obs
                    / ((self.simulator_base_std**2) * 2.25),
                    equivalent_post_diag_cov_2
                    * x_obs
                    / ((self.simulator_base_std**2) * (1 / 9)),
                ),
                dim=0,
            ),
            scale_tril=torch.stack(
                (
                    torch.diag(equivalent_post_diag_cov_1**0.5),
                    torch.diag(equivalent_post_diag_cov_2**0.5),
                ),
                dim=0,
            ),
            validate_args=False,
        )
        return torch.distributions.MixtureSameFamily(
            component_distribution=base_comp,
            mixture_distribution=torch.distributions.Categorical(
                logits=torch.stack(log_weights),
                validate_args=False,
            ),
            validate_args=False,
        )

    def diffused_posterior_mean_std(self, x_obs, mean_t, std_t):
        # This will only support one observation at a time, and also only one diffusion time
        # This is because the score computation is computed based on a torch distribution
        posterior_0 = self.true_posterior(x_obs)
        cov = posterior_0.component_distribution.covariance_matrix
        return torch.distributions.MixtureSameFamily(
            component_distribution=MultivariateNormal(
                loc=posterior_0.component_distribution.loc * mean_t,
                covariance_matrix=cov * mean_t**2
                + std_t * torch.eye(cov.shape[-1])[None].to(cov.device),
            ),
            mixture_distribution=posterior_0.mixture_distribution,
        )


# TODO implement one for each SDE type instead of commenting in and out
# class MockVPScoreEstimator(VPScoreEstimator):
class MockVPScoreEstimator(VEScoreEstimator):
    def __init__(
        self,
        # beta_min: float = 0.1,
        # beta_max: float = 20,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
    ) -> None:
        # super().__init__(None, (1, 1), "identity", beta_min, beta_max)
        super().__init__(None, (1, 1), "identity", sigma_min, sigma_max)
        # TODO make the dimensionality an argument
        self.gaussian_mix = Gaussian_MixtGaussian_mD(1)

    def forward(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        mean_val = self.mean_fn(torch.ones_like(input), times)
        std_val = self.std_fn(times)

        gm = self.gaussian_mix.diffused_posterior_mean_std(
            condition, mean_val[0].item(), std_val[0].item()
        )

        # necessary?
        theta = input.clone().detach().requires_grad_(True)

        log_prob_theta = gm.log_prob(theta)

        grad_log_prob_theta = torch.autograd.grad(
            log_prob_theta,
            theta,
            grad_outputs=torch.ones_like(log_prob_theta),
            create_graph=True,
        )[0]

        return grad_log_prob_theta.detach()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test = MockVPScoreEstimator()

    different_times = []
    for jdx, j in enumerate(torch.linspace(0.01, 0.9, 100)):
        hmm = test.forward(
            torch.linspace(-5, 5, 100).unsqueeze(1),
            torch.tensor([0.0]),
            torch.tensor([j]),
        )
        different_times.append(hmm)

    plt.plot(torch.cat(different_times, dim=1).detach().T, color="black", alpha=0.1)
