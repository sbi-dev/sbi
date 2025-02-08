from typing import Optional

import numpy as np
import torch
import torch.distributions as dist
from torch import Tensor

from sbi.neural_nets.estimators import ConditionalDensityEstimator


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array):
    return torch.tensor(np.array(array))


class TruncatedNormal(dist.TransformedDistribution):
    def __init__(self, loc, scale, low, high):
        """
        Implements a truncated normal using SigmoidTransform.

        Args:
            loc (Tensor): Mean of the base normal distribution.
            scale (Tensor): Standard deviation of the base normal.
            low (Tensor): Lower bound for truncation.
            high (Tensor): Upper bound for truncation.
        """
        # Ensure low and high match loc in shape
        low, high = torch.broadcast_tensors(low, high)
        low, high = low.expand_as(loc), high.expand_as(loc)

        base_dist = dist.Normal(loc, scale)
        sigmoid = dist.transforms.SigmoidTransform()
        scale_transform = dist.AffineTransform(loc=low, scale=(high - low))

        super().__init__(base_dist, [sigmoid, scale_transform])
        self.low = low
        self.high = high

    def sample(self, sample_shape=torch.Size()):
        """Samples from the truncated normal distribution."""
        return torch.as_tensor(super().sample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Computes log probability correctly accounting for truncation."""
        # Ensure value, low, and high have the same shape
        value, low, high = torch.broadcast_tensors(value, self.low, self.high)

        log_base = self.base_dist.log_prob(value)
        log_cdf_low = self.base_dist.cdf(low).log()
        log_cdf_high = (1 - self.base_dist.cdf(high)).log()

        return log_base - torch.log1p(-torch.exp(log_cdf_low + log_cdf_high))


class HierarchicalPrior(dist.Distribution):
    def __init__(
        self,
        mu_prior,
        sigma_prior,
        lower_bounds,
        upper_bounds,
        num_subjects,
        validate_args=None,
    ):
        """
        Hierarchical prior as a PyTorch distribution.

        Args:
            mu_prior (torch.distributions.Distribution): Prior for group means, shape (D,).
            sigma_prior (torch.distributions.Distribution): Prior for group stds, shape (D,).
            lower_bounds (torch.Tensor): Lower bounds for subject-level parameters, shape (D,).
            upper_bounds (torch.Tensor): Upper bounds for subject-level parameters, shape (D,).
            num_subjects (int): Number of subjects (M).
        """
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.num_subjects = num_subjects
        self.num_parameters = lower_bounds.shape[0]

        super().__init__(event_shape=self.event_shape, validate_args=validate_args)

    @property
    def event_shape(self):
        """Return the event shape as a property."""
        return torch.Size([(self.num_subjects + 2) * self.num_parameters])  # (M+2) * D

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """
        Samples from the hierarchical prior.

        Returns:
            theta_full_flat (torch.Tensor): Flattened sampled parameters of shape (sample_shape, (M+2) * D).
        """
        # Sample group-level parameters
        mu = self.mu_prior.sample(sample_shape)  # Shape (sample_shape, D)
        sigma = self.sigma_prior.sample(sample_shape)  # Shape (sample_shape, D)

        # Expand for broadcasting
        mu = mu.unsqueeze(-2)  # (sample_shape, 1, D)
        sigma = sigma.unsqueeze(-2)  # (sample_shape, 1, D)

        # Sample subject-level parameters
        truncated_normal = TruncatedNormal(
            mu.expand(*sample_shape, self.num_subjects, -1),
            sigma.expand(*sample_shape, self.num_subjects, -1),
            self.lower_bounds,
            self.upper_bounds,
        )
        theta = truncated_normal.sample()  # Shape (sample_shape, M, D)

        # Concatenate mu, sigma, and theta
        theta_full = torch.cat(
            (mu, sigma, theta), dim=-2
        )  # Shape (sample_shape, M+2, D)

        # Flatten for MCMC sampling
        return theta_full.reshape(*sample_shape, -1)  # Shape (sample_shape, (M+2) * D)

    def log_prob(self, theta_full_flat: Tensor) -> Tensor:
        """
        Computes log-probability of hierarchical prior.

        Args:
            theta_full_flat (torch.Tensor): Flattened hierarchical parameters, shape (batch_size, (M+2) * D).

        Returns:
            log_prob (torch.Tensor): Log probability, shape (batch_size,).
        """
        batch_size = theta_full_flat.shape[0]
        theta_full = theta_full_flat.view(
            batch_size, self.num_subjects + 2, -1
        )  # Reshape to (batch, M+2, D)

        mu = theta_full[:, 0, :]  # Shape (batch_size, D)
        sigma = theta_full[:, 1, :]  # Shape (batch_size, D)
        theta = theta_full[:, 2:, :]  # Shape (batch_size, M, D)

        # Log prob of group-level parameters (mu, sigma)
        log_p_mu = self.mu_prior.log_prob(mu).sum(dim=-1)  # Shape (batch_size,)
        log_p_sigma = self.sigma_prior.log_prob(sigma).sum(
            dim=-1
        )  # Shape (batch_size,)

        # Log prob of subject-level parameters
        truncated_normal = TruncatedNormal(
            mu.unsqueeze(-2), sigma.unsqueeze(-2), self.lower_bounds, self.upper_bounds
        )
        log_p_theta = truncated_normal.log_prob(theta).sum(
            dim=(-2, -1)
        )  # Sum over M and D

        return log_p_mu + log_p_sigma + log_p_theta  # Shape (batch_size,)


class HierarchicalPotentialFunction:
    def __init__(
        self, x_o, likelihood: ConditionalDensityEstimator, hierarchical_prior
    ):
        """
        Potential function for hierarchical Bayesian inference.

        Args:
            x_o (torch.Tensor): Observed data of shape (M, num_trials, 2).
            likelihood (torch.nn.Module): Likelihood function with .log_prob().
            hierarchical_prior (HierarchicalPrior): Hierarchical prior instance.
        """
        self.x_o = x_o  # Observed data, shape (M, N, 2)
        self.likelihood = likelihood  # SBI likelihood (torch module)
        self.hierarchical_prior = hierarchical_prior  # Hierarchical prior
        self.num_subjects = hierarchical_prior.num_subjects
        self.num_parameters = hierarchical_prior.lower_bounds.shape[
            0
        ]  # Dimensionality of parameters

    def __call__(self, theta: Tensor, x_o: Optional[Tensor] = None) -> Tensor:
        """
        Compute the unnormalized log-posterior (negative potential).

        Args:
            theta_full_flat (torch.Tensor): Flattened hierarchical parameters, shape (batch_size, (M+2) * D).

        Returns:
            torch.Tensor: Negative potential value (scalar).
        """
        batch_size = theta.shape[0]
        x_o: Tensor = x_o if x_o is not None else self.x_o

        # Unflatten theta_full
        theta_full = theta.view(
            batch_size, self.num_subjects + 2, self.num_parameters
        )  # Shape (batch, M+2, D)

        # Compute log prior
        log_p = self.hierarchical_prior.log_prob(theta)

        # Extract subject-level parameters
        theta_subject = theta_full[:, 2:, :]  # (batch, M, D)

        # Compute log likelihood
        log_likelihood = torch.zeros(batch_size, device=theta_subject.device)

        for m in range(self.num_subjects):
            theta_m = theta_subject[:, m, :]  # (batch_size, D)
            x_m = x_o[m].unsqueeze(1)  # (N, 1, x_event_dim)

            # Expand x_m to match batch size
            x_m = x_m.expand(
                -1, batch_size, *([-1] * (x_m.dim() - 2))
            )  # (N, batch_size, x_event_dim)

            # Compute log likelihood and sum over trials
            log_likelihood += self.likelihood.log_prob(x_m, theta_m).sum(
                dim=0
            )  # Sum over trials (N)

        return log_p + log_likelihood  # Shape (batch_size,)

    def _check_x_o(self, x_o):
        """Check if observed data has correct shape."""
        assert x_o.shape[2:] == self.x_o.shape[2:], (
            "Observed data has incorrect event shape."
        )


class PyroConditionalDensityEstimator():
    def __init__(self, density_estimator: ConditionalDensityEstimator):
        self.density_estimator = density_estimator

    def log_prob(self, input: np.ndarray, condition: np.ndarray) -> np.ndarray:
        input_torch = numpy_to_torch(input)
        condition_torch = numpy_to_torch(condition)
        log_probs_torch = self.density_estimator.log_prob(input_torch, condition_torch)
        return torch_to_numpy(log_probs_torch)

    def sample(self, sample_shape: tuple, condition: np.ndarray) -> np.ndarray:
        condition_torch = numpy_to_torch(condition)
        samples_torch = self.density_estimator.sample(torch.Size(sample_shape), condition_torch)
        return torch_to_numpy(samples_torch)
