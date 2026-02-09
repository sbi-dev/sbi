# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Mixture of Gaussians (MoG) dataclass and operations.

This module provides a lightweight container for MoG parameters with methods
for evaluation (log_prob), sampling, and conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class MoG:
    """Mixture of Gaussians parameters and operations.

    Represents an unconditional MoG distribution with fixed parameters.
    This class is used as the output of MultivariateGaussianMDN and for
    SNPE-A proposal correction.

    The MoG represents the distribution:
        p(x) = sum_k w_k * N(x; mu_k, Sigma_k)

    where w_k are mixture weights (derived from logits), mu_k are means,
    and Sigma_k are covariance matrices (inverse of precisions).

    Shape conventions:
        logits: (batch_size, num_components)
        means: (batch_size, num_components, dim)
        precisions: (batch_size, num_components, dim, dim)
        precision_factors: (batch_size, num_components, dim, dim)
            Upper triangular matrices A such that precision = A^T @ A

    Example:
        >>> mog = MoG(logits, means, precisions, precision_factors)
        >>> samples = mog.sample(torch.Size([1000]))  # (1000, batch_size, dim)
        >>> log_p = mog.log_prob(theta)  # (batch_size,) or broadcasted
    """

    logits: Tensor
    means: Tensor
    precisions: Tensor
    precision_factors: Optional[Tensor] = None

    # Small constant for numerical stability in Cholesky decomposition
    _CHOLESKY_EPSILON: float = 1e-6

    def __post_init__(self) -> None:
        """Validate tensor shapes, check for NaN/Inf, and compute precision_factors."""
        # Validate tensor dimensions
        if self.logits.dim() != 2:
            raise ValueError(
                f"logits must be 2D (batch_size, num_components), "
                f"got {self.logits.dim()}D"
            )
        if self.means.dim() != 3:
            raise ValueError(
                f"means must be 3D (batch_size, num_components, dim), "
                f"got {self.means.dim()}D"
            )
        if self.precisions.dim() != 4:
            raise ValueError(
                f"precisions must be 4D (batch_size, num_components, dim, dim), "
                f"got {self.precisions.dim()}D"
            )

        batch_size, num_components = self.logits.shape
        if self.means.shape[:2] != (batch_size, num_components):
            raise ValueError(
                f"means shape {self.means.shape} incompatible with "
                f"logits shape {self.logits.shape}"
            )
        if self.precisions.shape[:2] != (batch_size, num_components):
            raise ValueError(
                f"precisions shape {self.precisions.shape} incompatible with "
                f"logits shape {self.logits.shape}"
            )

        # Validate precision matrices are square
        dim = self.means.shape[2]
        if self.precisions.shape[2:] != (dim, dim):
            raise ValueError(
                f"precisions must be square matrices of size ({dim}, {dim}), "
                f"got {self.precisions.shape[2:]}"
            )

        # Validate no NaN or Inf values in input tensors
        if not torch.all(torch.isfinite(self.logits)):
            raise ValueError("logits contains NaN or Inf values")
        if not torch.all(torch.isfinite(self.means)):
            raise ValueError("means contains NaN or Inf values")
        if not torch.all(torch.isfinite(self.precisions)):
            raise ValueError("precisions contains NaN or Inf values")

        # Compute precision_factors if not provided
        # precision = A^T @ A where A is upper triangular (precision_factors)
        if self.precision_factors is None:
            # Add small epsilon to diagonal for numerical stability
            eye = torch.eye(
                dim, device=self.precisions.device, dtype=self.precisions.dtype
            )
            precisions_stabilized = self.precisions + self._CHOLESKY_EPSILON * eye

            try:
                # Use Cholesky: precision = L @ L^T, so A = L^T (upper triangular)
                L = torch.linalg.cholesky(precisions_stabilized)
            except torch.linalg.LinAlgError as e:
                raise ValueError(
                    "Failed to compute Cholesky decomposition of precision matrix. "
                    "This indicates the precision matrix is not positive definite. "
                    "Check that your MoG parameters are valid. "
                    f"Original error: {e}"
                ) from e

            # Note: need to use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'precision_factors', L.transpose(-2, -1))
        else:
            if self.precision_factors.shape != self.precisions.shape:
                raise ValueError(
                    f"precision_factors shape {self.precision_factors.shape} must "
                    f"match precisions shape {self.precisions.shape}"
                )
            if not torch.all(torch.isfinite(self.precision_factors)):
                raise ValueError("precision_factors contains NaN or Inf values")

    @property
    def num_components(self) -> int:
        """Number of mixture components."""
        return self.logits.shape[1]

    @property
    def dim(self) -> int:
        """Dimension of the distribution."""
        return self.means.shape[2]

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape of the distribution."""
        return torch.Size([self.logits.shape[0]])

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.logits.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensors."""
        return self.logits.dtype

    @property
    def weights(self) -> Tensor:
        """Normalized mixture weights (softmax of logits).

        Returns:
            Tensor of shape (batch_size, num_components) summing to 1.
        """
        return F.softmax(self.logits, dim=-1)

    @property
    def log_weights(self) -> Tensor:
        """Log of normalized mixture weights.

        Returns:
            Tensor of shape (batch_size, num_components).
        """
        return self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)

    def log_prob(self, inputs: Tensor) -> Tensor:
        """Evaluate log probability of inputs under the MoG.

        Computes log p(x) = log sum_k w_k * N(x; mu_k, Sigma_k)

        Uses the log-sum-exp trick for numerical stability.

        Args:
            inputs: Points at which to evaluate, shape (batch_size, dim) or
                (sample_size, batch_size, dim).

        Returns:
            Log probabilities. Shape depends on input:
                - (batch_size, dim) -> (batch_size,)
                - (sample_size, batch_size, dim) -> (sample_size, batch_size)
        """
        # Handle different input shapes
        if inputs.dim() == 2:
            # Shape: (batch_size, dim) -> add sample dimension
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Expand inputs for broadcasting with mixture components
        # inputs: (sample_size, batch_size, 1, dim)
        inputs_expanded = inputs.unsqueeze(2)

        # means: (1, batch_size, num_components, dim)
        means_expanded = self.means.unsqueeze(0)

        # Compute (x - mu) for each component
        # Shape: (sample_size, batch_size, num_components, dim)
        diff = inputs_expanded - means_expanded

        # Compute quadratic form: (x - mu)^T @ precision @ (x - mu)
        # precisions: (1, batch_size, num_components, dim, dim)
        precisions_expanded = self.precisions.unsqueeze(0)

        # diff_col: (sample_size, batch_size, num_components, dim, 1)
        diff_col = diff.unsqueeze(-1)

        # quadratic: (sample_size, batch_size, num_components)
        quadratic = (
            torch.matmul(
                torch.matmul(diff_col.transpose(-2, -1), precisions_expanded), diff_col
            )
            .squeeze(-1)
            .squeeze(-1)
        )

        # Compute log determinant of precision (= -log det of covariance)
        # Use sum of log diagonal of precision factors if available
        if self.precision_factors is not None:
            # sumlogdiag: (batch_size, num_components)
            sumlogdiag = torch.sum(
                torch.log(torch.diagonal(self.precision_factors, dim1=-2, dim2=-1)),
                dim=-1,
            )
        else:
            # Fall back to computing from precision matrices
            # log|precision| = log|A^T A| = 2 * log|A| = 2 * sum(log(diag(A)))
            # For general precision, use slogdet
            _, logabsdet = torch.linalg.slogdet(self.precisions)
            sumlogdiag = 0.5 * logabsdet

        # Compute log probability for each component
        # log N(x; mu, Sigma) = -0.5 * d * log(2*pi) + 0.5 * log|precision|
        #                      - 0.5 * (x-mu)^T @ precision @ (x-mu)
        log_norm = -0.5 * self.dim * np.log(2 * np.pi)

        # sumlogdiag: (1, batch_size, num_components)
        sumlogdiag_expanded = sumlogdiag.unsqueeze(0)

        # log_component_probs: (sample_size, batch_size, num_components)
        log_component_probs = log_norm + sumlogdiag_expanded - 0.5 * quadratic

        # Add log weights and sum over components using logsumexp
        # log_weights: (1, batch_size, num_components)
        log_weights_expanded = self.log_weights.unsqueeze(0)

        # log_prob: (sample_size, batch_size)
        log_prob = torch.logsumexp(log_weights_expanded + log_component_probs, dim=-1)

        if squeeze_output:
            log_prob = log_prob.squeeze(0)

        return log_prob

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample from the MoG.

        Args:
            sample_shape: Shape prefix for samples. Default is empty (single sample
                per batch element).

        Returns:
            Samples of shape (*sample_shape, batch_size, dim).
        """
        num_samples = int(torch.Size(sample_shape).numel()) if sample_shape else 1
        batch_size = self.logits.shape[0]

        # Normalize logits to get mixture coefficients
        coefficients = self.weights  # (batch_size, num_components)

        # Sample component indices for each sample
        # choices: (batch_size, num_samples)
        choices = torch.multinomial(
            coefficients, num_samples=num_samples, replacement=True
        )

        # Create batch indices for advanced indexing
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        batch_idx = batch_idx.expand(batch_size, num_samples)

        # Select means and precision factors for chosen components
        # chosen_means: (batch_size, num_samples, dim)
        chosen_means = self.means[batch_idx, choices, :]
        # chosen_precfs: (batch_size, num_samples, dim, dim)
        # Note: precision_factors is guaranteed to be set by __post_init__
        assert self.precision_factors is not None  # For type checker
        chosen_precfs = self.precision_factors[batch_idx, choices, :, :]

        # Sample from standard normal and transform
        # To get samples from N(mu, Sigma) where Sigma^{-1} = A^T A,
        # we compute: mu + A^{-1} @ z, where z ~ N(0, I)
        z = torch.randn(
            batch_size, num_samples, self.dim, 1, device=self.device, dtype=self.dtype
        )

        # Solve A @ x = z for x (i.e., x = A^{-1} @ z)
        # chosen_precfs is upper triangular
        zero_mean_samples = torch.linalg.solve_triangular(
            chosen_precfs, z, upper=True
        ).squeeze(-1)

        # Add means
        samples = chosen_means + zero_mean_samples

        # Reshape to requested sample_shape
        # Currently: (batch_size, num_samples, dim)
        # Target: (*sample_shape, batch_size, dim)
        samples = samples.transpose(0, 1)  # (num_samples, batch_size, dim)

        if sample_shape:
            samples = samples.reshape(*sample_shape, batch_size, self.dim)
        else:
            samples = samples.squeeze(0)  # (batch_size, dim)

        return samples

    def condition(
        self,
        condition: Tensor,
        dims_to_sample: List[int],
    ) -> MoG:
        """Compute conditional distribution p(X_free | X_fixed = condition).

        Given a MoG over all dimensions, returns a new MoG over only the
        free dimensions (specified by dims_to_sample), conditioned on the
        remaining dimensions being fixed to the values in condition.

        Uses the standard Gaussian conditioning formulas applied to each
        mixture component, with re-weighted mixture coefficients.

        Args:
            condition: Values for ALL dimensions, shape (batch_size, dim).
                Values at dims_to_sample are ignored.
            dims_to_sample: Indices of dimensions to keep free (not conditioned on).

        Returns:
            New MoG over the free dimensions only.
        """
        batch_size, num_components, full_dim = self.means.shape

        # Create mask for free vs fixed dimensions
        mask = torch.zeros(full_dim, dtype=torch.bool, device=self.device)
        mask[dims_to_sample] = True

        # Extract values for fixed dimensions
        y = condition[:, ~mask]  # (batch_size, num_fixed)

        # Extract means for free and fixed dimensions
        mu_x = self.means[:, :, mask]  # (batch_size, num_components, num_free)
        mu_y = self.means[:, :, ~mask]  # (batch_size, num_components, num_fixed)

        # Extract precision submatrices
        # precfs_xx: precision factors for free dimensions
        # Note: precision_factors is guaranteed to be set by __post_init__
        assert self.precision_factors is not None  # For type checker
        precfs = self.precision_factors

        precfs_xx = precfs[:, :, mask, :][:, :, :, mask]
        precfs_yy = precfs[:, :, ~mask, :][:, :, :, ~mask]

        # Compute precision matrices
        precs_xx = torch.matmul(precfs_xx.transpose(-2, -1), precfs_xx)
        precs_yy = torch.matmul(precfs_yy.transpose(-2, -1), precfs_yy)

        # Full precision and cross terms
        precs = torch.matmul(precfs.transpose(-2, -1), precfs)
        precs_xy = precs[:, :, mask, :][:, :, :, ~mask]

        # Conditional mean: mu_x|y = mu_x - precs_xx^{-1} @ precs_xy @ (y - mu_y)
        # y: (batch_size, num_fixed) -> (batch_size, num_components, num_fixed, 1)
        y_expanded = y.unsqueeze(1).unsqueeze(-1).expand(-1, num_components, -1, -1)
        mu_y_col = mu_y.unsqueeze(-1)  # (batch_size, num_components, num_fixed, 1)
        diff_y = y_expanded - mu_y_col

        # Compute precs_xx_inv @ precs_xy @ diff_y using solve for numerical stability
        # solve(A, B) computes A^{-1} @ B more stably than inv(A) @ B
        rhs = torch.matmul(precs_xy, diff_y)
        adjustment = torch.linalg.solve(precs_xx, rhs)
        cond_means = mu_x - adjustment.squeeze(-1)

        # Conditional precision factors are just precfs_xx (precision doesn't change)
        cond_precfs = precfs_xx
        cond_precs = precs_xx

        # Update mixture weights using marginal likelihood of y:
        # p(X|Y=y) = p(Y=y, X) / p(Y=y) => weights ~ p(y | component)
        diags_yy = torch.diagonal(precfs_yy, dim1=-2, dim2=-1)
        sumlogdiag_yy = torch.sum(torch.log(diags_yy), dim=-1)

        # Compute log p(y | component k) for each component k
        # This is N(y; mu_y_k, Sigma_yy_k) - need per-component log probs
        log_prob_y_per_component = self._log_prob_gaussian_per_component(
            y, mu_y, precs_yy, sumlogdiag_yy
        )

        # New (unnormalized) log weights: log(w_k * p(y|k)) = log(w_k) + log(p(y|k))
        new_log_weights = self.logits + log_prob_y_per_component
        # Normalize
        new_logits = new_log_weights - torch.logsumexp(
            new_log_weights, dim=-1, keepdim=True
        )

        return MoG(
            logits=new_logits,
            means=cond_means,
            precisions=cond_precs,
            precision_factors=cond_precfs,
        )

    @staticmethod
    def _log_prob_gaussian_per_component(
        inputs: Tensor,
        means: Tensor,
        precisions: Tensor,
        sumlogdiag: Tensor,
    ) -> Tensor:
        """Compute log probability per component (without mixture weighting).

        Computes log N(inputs; means_k, Sigma_k) for each component k.

        Args:
            inputs: (batch_size, dim)
            means: (batch_size, num_components, dim)
            precisions: (batch_size, num_components, dim, dim)
            sumlogdiag: (batch_size, num_components)

        Returns:
            Log probabilities per component (batch_size, num_components)
        """
        _, num_components, dim = means.shape

        # inputs: (batch_size, 1, dim)
        inputs_expanded = inputs.unsqueeze(1)

        # diff: (batch_size, num_components, dim)
        diff = inputs_expanded - means

        # quadratic form
        diff_col = diff.unsqueeze(-1)  # (batch_size, num_components, dim, 1)
        quad = (
            torch.matmul(torch.matmul(diff_col.transpose(-2, -1), precisions), diff_col)
            .squeeze(-1)
            .squeeze(-1)
        )  # (batch_size, num_components)

        # log probability per component (no mixture weights)
        log_norm = -0.5 * dim * np.log(2 * np.pi)
        log_component_probs = log_norm + sumlogdiag - 0.5 * quad

        return log_component_probs  # (batch_size, num_components)

    def to(self, device: torch.device) -> MoG:
        """Move all tensors to the specified device.

        Args:
            device: Target device.

        Returns:
            New MoG with tensors on the specified device.
        """
        return MoG(
            logits=self.logits.to(device),
            means=self.means.to(device),
            precisions=self.precisions.to(device),
            precision_factors=(
                self.precision_factors.to(device)
                if self.precision_factors is not None
                else None
            ),
        )

    def detach(self) -> MoG:
        """Detach all tensors from the computation graph.

        Returns:
            New MoG with detached tensors.
        """
        return MoG(
            logits=self.logits.detach(),
            means=self.means.detach(),
            precisions=self.precisions.detach(),
            precision_factors=(
                self.precision_factors.detach()
                if self.precision_factors is not None
                else None
            ),
        )

    @classmethod
    def from_gaussian(
        cls,
        mean: Tensor,
        covariance: Tensor,
    ) -> MoG:
        """Create a single-component MoG from Gaussian parameters.

        Useful for representing Gaussian priors as MoG for SNPE-A correction.

        Args:
            mean: Mean vector, shape (dim,) or (batch_size, dim).
            covariance: Covariance matrix, shape (dim, dim) or (batch_size, dim, dim).

        Returns:
            MoG with a single component.
        """
        # Ensure batch dimension
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)  # (1, dim)
        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0)  # (1, dim, dim)

        batch_size = mean.shape[0]

        # Add component dimension
        means = mean.unsqueeze(1)  # (batch_size, 1, dim)

        # Compute precision from covariance
        precision = torch.linalg.inv(covariance)  # (batch_size, dim, dim)
        precisions = precision.unsqueeze(1)  # (batch_size, 1, dim, dim)

        # Compute precision factor via Cholesky
        # precision = L @ L^T where L is lower triangular
        # We want upper triangular A such that precision = A^T @ A
        # So A = L^T
        L = torch.linalg.cholesky(precision)  # (batch_size, dim, dim)
        precision_factor = L.transpose(-2, -1)  # Upper triangular
        precision_factors = precision_factor.unsqueeze(1)  # (batch_size, 1, dim, dim)

        # Single component means logits = 0 (weight = 1 after softmax)
        logits = torch.zeros(batch_size, 1, device=mean.device, dtype=mean.dtype)

        return cls(
            logits=logits,
            means=means,
            precisions=precisions,
            precision_factors=precision_factors,
        )
