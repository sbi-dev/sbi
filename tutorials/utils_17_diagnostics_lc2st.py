from typing import Callable

import torch


class GaussianMixture:
    def __init__(self, dim=2, prior_bound=10) -> None:
        """Gaussian Mixture: Pytorch implementation of the `sbibm` implementation at
        https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/tasks/gaussian_mixture/task.py

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
        """
        self.dim_parameters = dim
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }
        self.prior_dist = torch.distributions.Independent(
            torch.distributions.Uniform(
                low=self.prior_params["low"], high=self.prior_params["high"]
            ),
            1,
        )

        self.simulator_base_std = torch.tensor([1, 0.1])

    def get_simulator(self) -> Callable:
        """Get function returning samples from simulator given parameters

        Return:
            Simulator callable
        """

        def simulator(parameters):
            mixing_dist = torch.distributions.Categorical(probs=torch.ones(2) / 2)
            samples = []
            for theta in parameters:
                component_dist = torch.distributions.MultivariateNormal(
                    loc=theta[None].repeat(2, 1),
                    covariance_matrix=torch.eye(self.dim_parameters)
                    * self.simulator_base_std,
                )
                mixture = torch.distributions.MixtureSameFamily(
                    mixture_distribution=mixing_dist,
                    component_distribution=component_dist,
                )
                samples.append(mixture.sample((1,))[0])

            return torch.stack(samples)

        return simulator
