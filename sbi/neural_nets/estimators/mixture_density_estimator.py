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

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.mog import MoG


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
        precisions[
            ..., torch.arange(self._features), torch.arange(self._features)
        ] += self._epsilon

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

    Wraps a MultivariateGaussianMDN and provides the standard ConditionalDensityEstimator
    interface for use with sbi's training and inference pipelines.

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

    def __init__(
        self,
        net: MultivariateGaussianMDN,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the mixture density estimator.

        Args:
            net: MultivariateGaussianMDN that maps conditions to MoG parameters.
            input_shape: Shape of the input (parameter space), typically (dim,).
            condition_shape: Shape of the condition (observation space), typically (dim,).
            embedding_net: Optional network to embed the condition before passing
                to the MDN. If provided, condition is first transformed by this
                network. The output dimension must match net.context_features.
        """
        super().__init__(net, input_shape, condition_shape)
        self._embedding_net = embedding_net if embedding_net is not None else nn.Identity()

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

    @property
    def embedding_net(self) -> nn.Module:
        """Return the embedding network."""
        return self._embedding_net

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Compute log probability of inputs given conditions.

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

        # Embed condition
        embedded_condition = self._embedding_net(condition)

        # Handle input with or without sample dimension
        has_sample_dim = input.dim() > len(self.input_shape) + 1
        if not has_sample_dim:
            input = input.unsqueeze(0)

        sample_dim = input.shape[0]
        batch_dim = input.shape[1]

        # Get MoG parameters
        mog = self.net.get_mixture_components(embedded_condition)

        # MoG.log_prob handles (sample_dim, batch_dim, dim) input
        log_probs = mog.log_prob(input)  # (sample_dim, batch_dim)

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

        Args:
            sample_shape: Shape prefix for samples.
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            Samples, shape (*sample_shape, batch_dim, *input_shape).
        """
        self._check_condition_shape(condition)

        # Embed condition
        embedded_condition = self._embedding_net(condition)

        # Get MoG parameters
        mog = self.net.get_mixture_components(embedded_condition)

        # MoG.sample returns (*sample_shape, batch_dim, dim) - matches sbi convention
        samples = mog.sample(sample_shape)

        return samples

    def get_mog(self, condition: Tensor) -> MoG:
        """Extract MoG parameters for a given condition.

        Useful for accessing the raw mixture parameters, e.g., for
        SNPE-A correction or analysis.

        Args:
            condition: Conditions, shape (batch_dim, *condition_shape).

        Returns:
            MoG object containing mixture parameters.
        """
        self._check_condition_shape(condition)
        embedded_condition = self._embedding_net(condition)
        return self.net.get_mixture_components(embedded_condition)
