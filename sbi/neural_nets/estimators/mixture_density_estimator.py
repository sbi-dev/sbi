# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Mixture Density Estimator for simulation-based inference.

This module provides:
- MultivariateGaussianMDN: Neural network that outputs MoG parameters
- MixtureDensityEstimator: ConditionalDensityEstimator wrapper for sbi integration

The MDN implementation is adapted from pyknos (https://github.com/sbi-dev/pyknos),
original implementation by Conor M. Durkan et al., licensed under Apache 2.0.
Based on: C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)
"""

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.mog import MoG
from sbi.utils.torchutils import BoxUniform


class MultivariateGaussianMDN(nn.Module):
    """Neural network that outputs Mixture of Gaussians parameters.

    Maps a context vector to MoG parameters (logits, means, precision_factors).
    Uses full (non-diagonal) covariance matrices parameterized via precision factors.

    Covariance parameterization:
        precision = precision_factor^T @ precision_factor
        where precision_factor is upper triangular with positive diagonal
        (enforced via softplus on diagonal elements).

    The network architecture consists of:
        1. A hidden network that maps context to hidden features
        2. Output layers for logits, means, and precision factors

    Example:
        >>> mdn = MultivariateGaussianMDN(
        ...     features=2,
        ...     context_features=10,
        ...     hidden_features=50,
        ...     num_components=5,
        ... )
        >>> context = torch.randn(32, 10)  # batch of 32 contexts
        >>> mog = mdn(context)  # returns MoG with batch_size=32
        >>> samples = mog.sample(torch.Size([100]))  # (100, 32, 2)
    """

    def __init__(
        self,
        features: int,
        context_features: int,
        hidden_features: int,
        num_components: int,
        hidden_net: Optional[nn.Module] = None,
        num_hidden_layers: int = 2,
        epsilon: float = 1e-4,
        custom_initialization: bool = False,
    ):
        """Initialize the MDN.

        Args:
            features: Dimension of the output distribution (parameter space).
            context_features: Dimension of the context/condition (observation space).
            hidden_features: Number of hidden units in the hidden network.
            num_components: Number of mixture components.
            hidden_net: Optional custom hidden network. If None, a default MLP
                with ReLU activations and dropout is created.
            num_hidden_layers: Number of hidden layers in default hidden network.
                Ignored if hidden_net is provided.
            epsilon: Small constant added to precision diagonal for numerical stability.
            custom_initialization: If True, initialize mixture coefficients to be
                approximately uniform and covariances to be approximately identity.
        """
        super().__init__()

        # Validate arguments
        if features < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        if context_features < 1:
            raise ValueError(f"context_features must be >= 1, got {context_features}")
        if hidden_features < 1:
            raise ValueError(f"hidden_features must be >= 1, got {hidden_features}")
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}")
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        self._features = features
        self._context_features = context_features
        self._hidden_features = hidden_features
        self._num_components = num_components
        self._epsilon = epsilon

        # Number of upper triangular elements (excluding diagonal)
        self._num_upper_params = (features * (features - 1)) // 2

        # Indices for filling precision factor matrix
        self._row_ix, self._column_ix = np.triu_indices(features, k=1)
        self._diag_ix = range(features)

        # Hidden network
        if hidden_net is not None:
            self._hidden_net = hidden_net
            # Verify hidden_features matches
            with torch.no_grad():
                test_output = self._hidden_net(torch.randn(1, context_features))
                inferred_features = test_output.shape[-1]
            if inferred_features != hidden_features:
                raise ValueError(
                    f"hidden_net output dimension ({inferred_features}) does not "
                    f"match hidden_features ({hidden_features})"
                )
        else:
            # Default hidden network: MLP with ReLU and dropout
            layers = []
            in_features = context_features
            for _ in range(num_hidden_layers):
                layers.extend([
                    nn.Linear(in_features, hidden_features),
                    nn.ReLU(),
                    nn.Dropout(p=0.0),  # Can be adjusted
                ])
                in_features = hidden_features
            self._hidden_net = nn.Sequential(*layers)

        # Output layers for MoG parameters
        self._logits_layer = nn.Linear(hidden_features, num_components)
        self._means_layer = nn.Linear(hidden_features, num_components * features)
        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )
        # Upper triangular elements (only needed for features > 1)
        if self._num_upper_params > 0:
            self._upper_layer = nn.Linear(
                hidden_features, num_components * self._num_upper_params
            )
        else:
            self._upper_layer = None

        # Initialize weights
        if custom_initialization:
            self._initialize()

    def forward(self, context: Tensor) -> MoG:
        """Compute MoG parameters for given context.

        Args:
            context: Conditioning context, shape (batch_size, context_features).

        Returns:
            MoG object containing the mixture parameters.
        """
        return self.get_mixture_components(context)

    def get_mixture_components(self, context: Tensor) -> MoG:
        """Compute MoG parameters for given context.

        Args:
            context: Conditioning context, shape (batch_size, context_features).

        Returns:
            MoG object with:
                - logits: (batch_size, num_components)
                - means: (batch_size, num_components, features)
                - precisions: (batch_size, num_components, features, features)
                - precision_factors: (batch_size, num_components, features, features)
        """
        batch_size = context.shape[0]

        # Pass through hidden network
        h = self._hidden_net(context)

        # Compute logits (unconstrained)
        logits = self._logits_layer(h)  # (batch_size, num_components)

        # Compute means (unconstrained)
        means = self._means_layer(h).view(
            batch_size, self._num_components, self._features
        )

        # Compute precision factors
        # Diagonal elements must be positive (enforced via softplus)
        unconstrained_diagonal = self._unconstrained_diagonal_layer(h).view(
            batch_size, self._num_components, self._features
        )
        diagonal = F.softplus(unconstrained_diagonal)

        # Build precision factor matrices (upper triangular)
        precision_factors = torch.zeros(
            batch_size,
            self._num_components,
            self._features,
            self._features,
            device=context.device,
            dtype=context.dtype,
        )

        # Fill diagonal
        precision_factors[..., self._diag_ix, self._diag_ix] = diagonal

        # Fill upper triangular (only for features > 1)
        if self._upper_layer is not None:
            upper = self._upper_layer(h).view(
                batch_size, self._num_components, self._num_upper_params
            )
            precision_factors[..., self._row_ix, self._column_ix] = upper

        # Compute precision matrices: precision = A^T @ A
        precisions = torch.matmul(
            precision_factors.transpose(-2, -1), precision_factors
        )

        # Add epsilon to diagonal for numerical stability
        precisions[..., torch.arange(self._features), torch.arange(self._features)] += (
            self._epsilon
        )

        return MoG(
            logits=logits,
            means=means,
            precisions=precisions,
            precision_factors=precision_factors,
        )

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Compute log probability of inputs given context.

        Convenience method that extracts MoG and evaluates log_prob.

        Args:
            inputs: Points to evaluate, shape (batch_size, features).
            context: Conditioning context, shape (batch_size, context_features).

        Returns:
            Log probabilities, shape (batch_size,).
        """
        mog = self.get_mixture_components(context)
        return mog.log_prob(inputs)

    def sample(self, sample_shape: torch.Size, context: Tensor) -> Tensor:
        """Sample from the MoG given context.

        Args:
            sample_shape: Shape prefix for samples (PyTorch convention).
            context: Conditioning context, shape (batch_size, context_features).

        Returns:
            Samples, shape (*sample_shape, batch_size, features).
        """
        mog = self.get_mixture_components(context)
        return mog.sample(sample_shape)

    def _initialize(self) -> None:
        """Initialize MDN for approximately uniform mixture and identity covariance.

        This initialization helps with training stability by starting with
        a reasonable prior over the output space.
        """
        # Initialize logits layer to give approximately uniform weights
        nn.init.normal_(self._logits_layer.weight, mean=0.0, std=self._epsilon)
        nn.init.normal_(self._logits_layer.bias, mean=0.0, std=self._epsilon)

        # Initialize diagonal layer to give approximately identity covariance
        # softplus(x) ≈ 1 when x ≈ log(e - 1) ≈ 0.54
        nn.init.normal_(
            self._unconstrained_diagonal_layer.weight, mean=0.0, std=self._epsilon
        )
        softplus_one = torch.log(torch.exp(torch.tensor(1.0 - self._epsilon)) - 1)
        nn.init.constant_(self._unconstrained_diagonal_layer.bias, softplus_one.item())

        # Initialize upper triangular layer to zeros (diagonal covariance initially)
        if self._upper_layer is not None:
            nn.init.normal_(self._upper_layer.weight, mean=0.0, std=self._epsilon)
            nn.init.zeros_(self._upper_layer.bias)

    @property
    def features(self) -> int:
        """Dimension of the output distribution."""
        return self._features

    @property
    def context_features(self) -> int:
        """Dimension of the context."""
        return self._context_features

    @property
    def num_components(self) -> int:
        """Number of mixture components."""
        return self._num_components

    @property
    def hidden_features(self) -> int:
        """Dimension of hidden representation."""
        return self._hidden_features


class MixtureDensityEstimator(ConditionalDensityEstimator):
    """MDN-based conditional density estimator.

    Wraps a MultivariateGaussianMDN and provides the standard
    ConditionalDensityEstimator interface for use with sbi's training and
    inference pipelines.

    The estimator models the conditional distribution p(input | condition) as a
    Mixture of Gaussians, where the mixture parameters are predicted by a neural
    network given the condition.

    Example:
        >>> # Create MDN with 2D input, 5D condition, 10 mixture components
        >>> mdn_net = MultivariateGaussianMDN(
        ...     features=2,
        ...     context_features=5,
        ...     hidden_features=50,
        ...     num_components=10,
        ... )
        >>> estimator = MixtureDensityEstimator(
        ...     net=mdn_net,
        ...     input_shape=torch.Size([2]),
        ...     condition_shape=torch.Size([5]),
        ... )
        >>> # Use for density estimation
        >>> condition = torch.randn(32, 5)
        >>> samples = estimator.sample(torch.Size([100]), condition)  # (100, 32, 2)
        >>> log_prob = estimator.log_prob(samples, condition)  # (100, 32)
    """

    # Small constant for numerical stability in matrix operations
    _STABILITY_EPSILON: float = 1e-6

    def __init__(
        self,
        net: MultivariateGaussianMDN,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        transform_input: Optional[Tensor] = None,
    ) -> None:
        """Initialize the mixture density estimator.

        Args:
            net: MultivariateGaussianMDN that maps conditions to MoG parameters.
            input_shape: Shape of the input (parameter space), typically (dim,).
            condition_shape: Shape of the condition (observation space),
                typically (dim,).
            embedding_net: Optional network to embed the condition before passing
                to the MDN. If provided, condition is first transformed by this
                network. The output dimension must match net.context_features.
            transform_input: Optional tensor of shape (2, input_dim) containing
                [shift, scale] for z-score transformation of inputs. If provided,
                inputs are transformed as: z = (x - shift) / scale before density
                evaluation, and samples are inverse transformed as:
                x = z * scale + shift.
                This is used for z-scoring inputs to improve numerical stability.
        """
        super().__init__(net, input_shape, condition_shape)
        self._embedding_net = (
            embedding_net if embedding_net is not None else nn.Identity()
        )

        # Validate that embedding_net output matches MDN input if embedding is provided
        if embedding_net is not None:
            with torch.no_grad():
                test_input = torch.randn(1, *condition_shape)
                test_output = self._embedding_net(test_input)
                embedded_dim = test_output.shape[-1]
            if embedded_dim != net.context_features:
                raise ValueError(
                    f"embedding_net output dimension ({embedded_dim}) does not match "
                    f"MDN context_features ({net.context_features})"
                )

        # Store z-score transform parameters as buffers (not trained, moved with model)
        if transform_input is not None:
            if transform_input.shape[0] != 2:
                raise ValueError(
                    f"transform_input must have shape (2, input_dim), "
                    f"got {transform_input.shape}"
                )
            scale = transform_input[1]
            if not torch.all(scale > 0):
                raise ValueError(
                    "transform_input scale (second row) must be strictly positive, "
                    f"got values in range [{scale.min().item():.6f}, "
                    f"{scale.max().item():.6f}]"
                )
            self.register_buffer("_transform_shift", transform_input[0])
            self.register_buffer("_transform_scale", scale)
        else:
            self.register_buffer("_transform_shift", None)
            self.register_buffer("_transform_scale", None)

        # SNPE-A correction state (None means no correction applied)
        self._proposal_mog: Optional[MoG] = None
        self._prior_mog: Optional[MoG] = None  # None for uniform priors
        self._prior_is_uniform: bool = False

    @property
    def embedding_net(self) -> nn.Module:
        """Return the embedding network."""
        return self._embedding_net

    @property
    def has_input_transform(self) -> bool:
        """Whether input z-score transform is enabled."""
        return self._transform_shift is not None

    def _transform_input(self, input: Tensor) -> Tensor:
        """Apply z-score transform to input: z = (x - shift) / scale."""
        if not self.has_input_transform:
            return input
        return (input - self._transform_shift) / self._transform_scale

    def _inverse_transform_input(self, z: Tensor) -> Tensor:
        """Apply inverse z-score transform: x = z * scale + shift."""
        if not self.has_input_transform:
            return z
        return z * self._transform_scale + self._transform_shift

    def _log_det_jacobian_forward(self, input: Tensor) -> Tensor:
        """Compute log determinant of Jacobian for the forward z-score transform.

        For the forward affine transform z = (x - shift) / scale:
            - The Jacobian matrix is: dz/dx = diag(1/scale)
            - The determinant is: |det(dz/dx)| = prod(1/scale)
            - The log determinant is: log|det(dz/dx)| = -sum(log(scale))

        Change of Variables Formula:
            When we have a density p_z(z) and want p_x(x), we use:
            p_x(x) = p_z(z(x)) * |det(dz/dx)|

            In log space:
            log p_x(x) = log p_z(z(x)) + log|det(dz/dx)|

            Since log|det(dz/dx)| = -sum(log(scale)), we have:
            log p_x(x) = log p_z(z) - sum(log(scale))

        This method returns log|det(dz/dx)| = -sum(log(scale)), which should
        be ADDED to log p_z(z) to get log p_x(x).

        Args:
            input: Input tensor, used to determine device and dtype.

        Returns:
            Log determinant of forward Jacobian (scalar), on the same device as input.
        """
        if not self.has_input_transform:
            return torch.zeros(1, device=input.device, dtype=input.dtype).squeeze()
        return -torch.log(self._transform_scale).sum()

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Compute log probability of inputs given conditions.

        If SNPE-A correction is applied, returns corrected log probability.
        If input transform is set, inputs are z-scored before density evaluation
        and the log-det-jacobian is subtracted from the log probability.

        Args:
            input: Inputs to evaluate, shape (sample_dim, batch_dim, *input_shape)
                or (batch_dim, *input_shape).
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Log probabilities. Shape (sample_dim, batch_dim) if input has sample_dim,
            otherwise (batch_dim,).
        """
        self._check_condition_shape(condition)
        self._check_input_shape(input)

        # Handle input with or without sample dimension
        has_sample_dim = input.dim() > len(self.input_shape) + 1
        if not has_sample_dim:
            input = input.unsqueeze(0)

        # Apply z-score transform to input if enabled
        transformed_input = self._transform_input(input)

        # Get MoG (corrected if correction is applied)
        mog = self._get_mog_internal(condition)

        # MoG.log_prob handles (sample_dim, batch_dim, dim) input
        # Change of variables: log p(x) = log p(z) + log|det(dz/dx)|
        # where z = (x - shift) / scale and log|det(dz/dx)| = -sum(log(scale))
        log_probs = mog.log_prob(transformed_input)  # (sample_dim, batch_dim)
        log_probs = log_probs + self._log_det_jacobian_forward(input)

        if not has_sample_dim:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Compute training loss (negative log probability).

        Args:
            input: Inputs, shape (batch_dim, *input_shape).
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Loss per batch element, shape (batch_dim,).
        """
        # Add sample dimension, compute log_prob, remove sample dimension
        log_prob = self.log_prob(input.unsqueeze(0), condition)
        return -log_prob.squeeze(0)

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        """Sample from the conditional distribution.

        If SNPE-A correction is applied, samples from the corrected distribution.
        If input transform is set, samples are inverse transformed (un-z-scored).

        Args:
            sample_shape: Shape prefix for samples.
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Samples, shape (*sample_shape, batch_dim, *input_shape).
        """
        self._check_condition_shape(condition)

        # Get MoG (corrected if correction is applied)
        mog = self._get_mog_internal(condition)

        # MoG.sample returns (*sample_shape, batch_dim, dim) - matches sbi convention
        samples = mog.sample(sample_shape)

        # Apply inverse transform to get samples in original space
        samples = self._inverse_transform_input(samples)

        return samples

    def get_mog(self, condition: Tensor) -> MoG:
        """Extract MoG parameters for a given condition.

        If SNPE-A correction is applied, returns the corrected MoG.
        Use `get_uncorrected_mog()` to get the raw density estimator output.

        Args:
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            MoG object containing mixture parameters.
        """
        return self._get_mog_internal(condition)

    def get_uncorrected_mog(self, condition: Tensor) -> MoG:
        """Extract raw MoG parameters (without SNPE-A correction).

        Args:
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            MoG object containing raw density estimator output.
        """
        self._check_condition_shape(condition)
        embedded_condition = self._embedding_net(condition)
        return self.net.get_mixture_components(embedded_condition)

    def _get_mog_internal(self, condition: Tensor) -> MoG:
        """Get MoG, applying correction if set.

        Args:
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            MoG object (corrected if correction is applied).
        """
        self._check_condition_shape(condition)
        embedded_condition = self._embedding_net(condition)
        density_mog = self.net.get_mixture_components(embedded_condition)

        if not self.correction_applied:
            return density_mog
        else:
            return self._compute_corrected_mog(density_mog)

    # =========================================================================
    # SNPE-A Correction Methods
    # =========================================================================

    @property
    def correction_applied(self) -> bool:
        """Whether SNPE-A correction is currently applied."""
        # Only check proposal - prior_mog can be None for uniform priors
        return self._proposal_mog is not None

    def apply_correction(
        self,
        proposal_mog: MoG,
        prior: Union[MultivariateNormal, BoxUniform],
    ) -> None:
        """Apply SNPE-A posterior correction.

        After calling this, log_prob(), sample(), and get_mog() return
        corrected values. Implements Appendix C of Papamakarios et al. 2016.

        The correction computes the true posterior from the proposal posterior:
            posterior = density_estimator * prior / proposal

        For uniform (BoxUniform) priors, the prior term is omitted from the
        correction formula, as uniform distributions have zero precision
        (infinite covariance) within their bounds.

        Args:
            proposal_mog: MoG from previous round's posterior (the proposal
                distribution used to generate training data).
            prior: Prior distribution. Must be MultivariateNormal or BoxUniform.

        Raises:
            ValueError: If prior is not a supported type.
        """
        # Handle different prior types
        if isinstance(prior, BoxUniform):
            # For uniform priors, the correction formula simplifies:
            # posterior = density * prior / proposal
            # Since uniform has zero precision, prior term is omitted
            self._prior_mog = None
            self._prior_is_uniform = True
        elif isinstance(prior, MultivariateNormal):
            # Note: covariance_matrix is a Tensor but PyTorch stubs type it as property
            cov = prior.covariance_matrix
            assert isinstance(cov, Tensor)  # For type checker
            self._prior_mog = MoG.from_gaussian(prior.mean, cov).detach()
            self._prior_is_uniform = False
        else:
            raise ValueError(
                f"prior must be MultivariateNormal or BoxUniform, "
                f"got {type(prior).__name__}"
            )

        # Store detached copy of proposal
        self._proposal_mog = proposal_mog.detach()

    def clear_correction(self) -> None:
        """Remove SNPE-A correction.

        After calling this, log_prob(), sample(), and get_mog() return
        uncorrected density estimator output.
        """
        self._proposal_mog = None
        self._prior_mog = None
        self._prior_is_uniform = False

    def _compute_corrected_mog(self, density_mog: MoG) -> MoG:
        """Compute corrected MoG using stored proposal and prior.

        Implements the analytical correction from Appendix C of
        Papamakarios et al. 2016 (SNPE-A paper).

        The posterior is computed as:
            posterior = density_estimator * prior / proposal

        Since all are MoGs, this can be done in closed form.
        If proposal has L components and density has K components,
        the posterior has L*K components.

        For uniform priors, the prior term is omitted (zero precision).

        Args:
            density_mog: MoG from the density estimator for current observation.

        Returns:
            Corrected MoG representing the posterior.

        Raises:
            RuntimeError: If correction is not applied (no proposal set).
            ValueError: If posterior precision is not positive definite.
        """
        if self._proposal_mog is None:
            raise RuntimeError(
                "Cannot compute corrected MoG: no correction applied. "
                "Call apply_correction() first."
            )

        proposal = self._proposal_mog
        prior = self._prior_mog  # Can be None for uniform priors

        # Get the number of components
        num_comps_proposal = proposal.num_components
        num_comps_density = density_mog.num_components

        # Compute posterior precisions (Eq. 23)
        # prec_post = prec_density - prec_proposal + prec_prior
        # For uniform priors, prec_prior = 0
        prec_proposal_rep = proposal.precisions.repeat_interleave(
            num_comps_density, dim=1
        )
        prec_density_rep = density_mog.precisions.repeat(1, num_comps_proposal, 1, 1)

        prec_post = prec_density_rep - prec_proposal_rep

        # Add prior precision term only for Gaussian priors
        if prior is not None:
            prec_prior_rep = prior.precisions.repeat(
                1, num_comps_proposal * num_comps_density, 1, 1
            )
            prec_post = prec_post + prec_prior_rep

        # Add small epsilon to diagonal for numerical stability
        dim = prec_post.shape[-1]
        eye = torch.eye(dim, device=prec_post.device, dtype=prec_post.dtype)
        prec_post_stabilized = prec_post + self._STABILITY_EPSILON * eye

        # Check positive definiteness using Cholesky (more efficient than eigenvalues)
        self._check_precision_positive_definite(prec_post_stabilized, "posterior")

        # Compute posterior covariances using solve for numerical stability
        # Instead of cov = inv(prec), we use cov = solve(prec, I) which is more stable
        batch_shape = prec_post_stabilized.shape[:-2]
        eye_expanded = eye.expand(*batch_shape, dim, dim)
        cov_post = torch.linalg.solve(prec_post_stabilized, eye_expanded)

        # Compute posterior means (Eq. 24)
        # mean_post = cov_post @ (prec_density @ mean_density
        #                         - prec_proposal @ mean_proposal
        #                         + prec_prior @ mean_prior)
        # For uniform priors, prec_prior @ mean_prior = 0
        prec_mean_proposal = self._batched_mv(proposal.precisions, proposal.means)
        prec_mean_density = self._batched_mv(density_mog.precisions, density_mog.means)

        prec_mean_proposal_rep = prec_mean_proposal.repeat_interleave(
            num_comps_density, dim=1
        )
        prec_mean_density_rep = prec_mean_density.repeat(1, num_comps_proposal, 1)

        summed_prec_mean = prec_mean_density_rep - prec_mean_proposal_rep

        # Add prior mean term only for Gaussian priors
        if prior is not None:
            prec_mean_prior = self._batched_mv(prior.precisions, prior.means)
            prec_mean_prior_rep = prec_mean_prior.repeat(
                1, num_comps_proposal * num_comps_density, 1
            )
            summed_prec_mean = summed_prec_mean + prec_mean_prior_rep

        mean_post = self._batched_mv(cov_post, summed_prec_mean)

        # Compute posterior logits (Eqs. 25-26)
        logits_post = self._compute_posterior_logits(
            mean_post,
            prec_post,
            cov_post,
            proposal.logits,
            proposal.means,
            proposal.precisions,
            density_mog.logits,
            density_mog.means,
            density_mog.precisions,
            num_comps_proposal,
            num_comps_density,
        )

        # Compute precision factors from stabilized precisions using Cholesky
        try:
            precf_post = torch.linalg.cholesky(prec_post_stabilized, upper=True)
        except torch.linalg.LinAlgError as e:
            raise ValueError(
                "Failed to compute Cholesky decomposition during SNPE-A correction. "
                "This indicates numerical instability in the posterior precision "
                f"matrix. Original error: {e}"
            ) from e

        return MoG(
            logits=logits_post,
            means=mean_post,
            precisions=prec_post_stabilized,
            precision_factors=precf_post,
        )

    def _compute_posterior_logits(
        self,
        mean_post: Tensor,
        prec_post: Tensor,
        cov_post: Tensor,
        logits_proposal: Tensor,
        mean_proposal: Tensor,
        prec_proposal: Tensor,
        logits_density: Tensor,
        mean_density: Tensor,
        prec_density: Tensor,
        num_comps_proposal: int,
        num_comps_density: int,
    ) -> Tensor:
        """Compute posterior logits using Eqs. 25-26 from SNPE-A paper.

        The logits are computed as:
            logits_post = logits_density - logits_proposal
                          + 0.5 * (logdet(cov_post) + logdet(cov_proposal)
                                   - logdet(cov_density))
                          - 0.5 * (m_d^T P_d m_d - m_p^T P_p m_p
                                   - m_post^T P_post m_post)

        Uses torch.linalg.slogdet for numerical stability.
        """
        # Compute logit factors
        logits_proposal_rep = logits_proposal.repeat_interleave(
            num_comps_density, dim=1
        )
        logits_density_rep = logits_density.repeat(1, num_comps_proposal)
        logit_factors = logits_density_rep - logits_proposal_rep

        # Compute log-determinant terms using slogdet for numerical stability
        # slogdet returns (sign, logabsdet) - for positive definite matrices sign=1
        _, logdet_cov_post = torch.linalg.slogdet(cov_post)
        # logdet(cov) = -logdet(prec) for inverse relationship
        _, logdet_prec_proposal = torch.linalg.slogdet(prec_proposal)
        _, logdet_prec_density = torch.linalg.slogdet(prec_density)
        logdet_cov_proposal = -logdet_prec_proposal
        logdet_cov_density = -logdet_prec_density

        logdet_cov_proposal_rep = logdet_cov_proposal.repeat_interleave(
            num_comps_density, dim=1
        )
        logdet_cov_density_rep = logdet_cov_density.repeat(1, num_comps_proposal)

        log_sqrt_det_ratio = 0.5 * (
            logdet_cov_post + logdet_cov_proposal_rep - logdet_cov_density_rep
        )

        # Compute quadratic form terms (m^T P m)
        exponent_proposal = self._batched_vmv(prec_proposal, mean_proposal)
        exponent_density = self._batched_vmv(prec_density, mean_density)
        exponent_post = self._batched_vmv(prec_post, mean_post)

        exponent_proposal_rep = exponent_proposal.repeat_interleave(
            num_comps_density, dim=1
        )
        exponent_density_rep = exponent_density.repeat(1, num_comps_proposal)

        exponent = -0.5 * (exponent_density_rep - exponent_proposal_rep - exponent_post)

        return logit_factors + log_sqrt_det_ratio + exponent

    @staticmethod
    def _batched_mv(matrix: Tensor, vector: Tensor) -> Tensor:
        """Batched matrix-vector product with component dimension.

        Args:
            matrix: Shape (batch, num_components, dim, dim).
            vector: Shape (batch, num_components, dim).

        Returns:
            Product of shape (batch, num_components, dim).
        """
        return torch.einsum("bcij,bcj->bci", matrix, vector)

    @staticmethod
    def _batched_vmv(matrix: Tensor, vector: Tensor) -> Tensor:
        """Batched vector-matrix-vector product (quadratic form).

        Args:
            matrix: Shape (batch, num_components, dim, dim).
            vector: Shape (batch, num_components, dim).

        Returns:
            Quadratic form v^T M v of shape (batch, num_components).
        """
        mv = torch.einsum("bcij,bcj->bci", matrix, vector)
        return torch.einsum("bci,bci->bc", vector, mv)

    @staticmethod
    def _check_precision_positive_definite(prec: Tensor, name: str) -> None:
        """Check that precision matrices are positive definite.

        Uses Cholesky decomposition for efficient checking. This is O(n^3)
        compared to O(n^3) for eigendecomposition, but with a smaller constant
        factor in practice.

        Args:
            prec: Precision matrices of shape (batch, num_components, dim, dim).
            name: Name for error message.

        Raises:
            ValueError: If any precision matrix is not positive definite.
        """
        try:
            # Cholesky will raise LinAlgError if not positive definite
            torch.linalg.cholesky(prec)
        except torch.linalg.LinAlgError as e:
            raise ValueError(
                f"Precision matrix of {name} is not positive definite. "
                "This is a known issue with SNPE-A when the proposal and density "
                "estimator don't align well. Try different hyperparameters."
            ) from e
