from typing import Union

import torch
from torch import Tensor
from torch.distributions import Binomial, Distribution, InverseGamma

from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils.torchutils import atleast_2d


class BinomialGammaPotential(BasePotential):
    """Binomial-Gamma potential for mixed data."""

    def __init__(
        self,
        prior: Distribution,
        x_o: Tensor,
        concentration_scaling: Union[Tensor, float] = 1.0,
        device="cpu",
    ):
        super().__init__(prior, x_o, device)

        # concentration_scaling needs to be a float or match the batch size
        if isinstance(concentration_scaling, Tensor):
            num_trials = x_o.shape[0]
            assert concentration_scaling.shape[0] == num_trials

            # Reshape to match convention (batch_size, num_trials, *event_shape)
            concentration_scaling = concentration_scaling.reshape(1, num_trials, -1)

        self.concentration_scaling = concentration_scaling

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        theta = atleast_2d(theta)

        with torch.set_grad_enabled(track_gradients):
            iid_ll = self.iid_likelihood(theta)

        return iid_ll + self.prior.log_prob(theta)

    def iid_likelihood(self, theta: Tensor) -> Tensor:
        batch_size = theta.shape[0]
        num_trials = self.x_o.shape[0]
        theta = theta.reshape(batch_size, 1, -1)
        beta, rhos = theta[:, :, :1], theta[:, :, 1:]

        # vectorized
        logprob_choices = torch.stack(
            [Binomial(probs=rho).log_prob(self.x_o[:, 1:]) for rho in rhos],
        )

        logprob_rts = InverseGamma(
            concentration=self.concentration_scaling * torch.ones_like(beta),
            rate=beta,
        ).log_prob(self.x_o[:, :1].reshape(1, num_trials, -1))

        joint_likelihood = torch.sum(logprob_choices, dim=-1) + logprob_rts.squeeze()

        assert joint_likelihood.shape == torch.Size([theta.shape[0], self.x_o.shape[0]])
        return joint_likelihood.sum(1)
