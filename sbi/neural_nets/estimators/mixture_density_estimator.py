# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Mixture Density Estimator for simulation-based inference.

This module provides:
- MultivariateGaussianMDN: Neural network that outputs MoG parameters
- MixtureDensityEstimator: ConditionalDensityEstimator wrapper (TODO: Phase 2)

The MDN implementation is adapted from pyknos (https://github.com/sbi-dev/pyknos),
original implementation by Conor M. Durkan et al., licensed under Apache 2.0.
Based on: C. M. Bishop, "Mixture Density Networks", NCRG Report (1994)
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
            Samples, shape (batch_size, *sample_shape, features).

        Note:
            Output shape differs from MoG.sample() which returns (*sample_shape,
            batch_size, dim). MDN places batch_size first since it represents
            conditioning on different observations.
        """
        mog = self.get_mixture_components(context)

        # MoG.sample returns (*sample_shape, batch_size, features)
        samples = mog.sample(sample_shape)

        # Move batch dimension to front: (*sample_shape, batch_size, features)
        # -> (batch_size, *sample_shape, features)
        num_sample_dims = len(sample_shape)

        # Permute: move the batch_size dim (which is after sample_shape dims) to front
        # From: (s1, s2, ..., batch_size, features) to (batch_size, s1, s2, ..., features)
        dims = list(range(samples.dim()))
        batch_dim_idx = num_sample_dims
        new_dims = [batch_dim_idx] + dims[:batch_dim_idx] + dims[batch_dim_idx + 1:]
        samples = samples.permute(*new_dims)

        return samples

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
