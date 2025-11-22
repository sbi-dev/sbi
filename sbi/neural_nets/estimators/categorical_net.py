# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, List, Optional, cast

import torch
from nflows.utils import torchutils
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.utils.nn_utils import MADEWrapper as MADE


class CategoricalMADE(MADE):
    """Conditional density (mass) estimation for a n-dim categorical random variable.

    Takes as input parameters theta and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        num_categories: Tensor,
        num_hidden_features: int,
        categorical_values: Optional[List[Tensor]] = None,
        num_context_features: Optional[int] = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        random_mask: bool = False,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        epsilon: float = 1e-2,
        embedding_net: nn.Module = nn.Identity(),
    ):
        """Initialize the neural net.

        Args:
            num_categories: number of categories for each variable. len(categories)
                defines the number of input units, i.e., dimensionality of the features.
                max(categories) defines the number of output units, i.e., the largest
                number of categories. Can handle mutliple variables with differing
                numbers of choices.
            num_hidden_features: number of hidden units per layer.
            categorical_values: list of unique values for each variable.
            num_context_features: number of context features.
            num_blocks: number of masked blocks.
            use_residual_blocks: whether to use residual blocks.
            random_mask: whether to use a random mask.
            activation: activation function. default is ReLU.
            dropout_probability: dropout probability. default is 0.0.
            use_batch_norm: whether to use batch normalization.
            embedding_net: emebedding net for input.
        """
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")

        self.num_variables = len(num_categories)
        self.max_num_categories = int(torch.max(num_categories))

        super().__init__(
            features=self.num_variables,
            hidden_features=num_hidden_features,
            context_features=num_context_features,
            num_blocks=num_blocks,
            output_multiplier=self.max_num_categories,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        # Store the original num_categories for each variable as a buffer
        self.num_categories_per_var = num_categories.clone()

        # Determine if mapping is needed (only when categorical_values is provided)
        self._needs_mapping = categorical_values is not None

        # Store unique values for mapping (initialized on first use)
        if self._needs_mapping:
            self.categorical_values_per_var = cast(List[Tensor], categorical_values)
        else:
            # assume that all values are 0 to num_categories
            self.categorical_values_per_var: List[Tensor] = [
                torch.arange(int(num_categories[i])) for i in range(self.num_variables)
            ]

        mask = torch.zeros(self.num_variables, self.max_num_categories)
        for i, c in enumerate(num_categories):
            mask[i, :c] = 1
        self.register_buffer("mask", mask)

        self.embedding_net = embedding_net
        self.hidden_features = num_hidden_features
        self.epsilon = epsilon
        self.context_features = num_context_features

        # Initialize and register lookup buffers only if mapping is needed
        if self._needs_mapping:
            self._init_lookup_buffers()

    def _init_lookup_buffers(self) -> None:
        """Initialize and register lookup tensors as buffers."""
        # Precompute and register per-variable lookup tables for value/index mapping.
        dtype = (
            self.categorical_values_per_var[0].dtype
            if len(self.categorical_values_per_var) > 0
            else torch.long
        )
        values_lookup = torch.zeros(
            self.num_variables, self.max_num_categories, dtype=dtype
        )
        for i, categorical_vals in enumerate(self.categorical_values_per_var):
            sorted_vals = torch.sort(categorical_vals).values
            c = int(self.num_categories_per_var[i])
            values_lookup[i, :c] = sorted_vals.to(dtype)

        # Register buffers for proper device moves and state_dict handling.
        self.register_buffer("values_lookup", values_lookup)

    def _map_values_to_indices(self, input: Tensor) -> Tensor:
        """Map user categorical values to 0-based indices."""
        # Cast once to the lookup dtype to avoid per-iteration conversions
        input_for_search = input.to(self.values_lookup.dtype)
        mapped = input_for_search.clone()
        for i in range(self.num_variables):
            c = int(self.num_categories_per_var[i])
            unique_vals = self.values_lookup[i, :c]
            input_values = input_for_search[..., i].contiguous()
            indices = torch.searchsorted(unique_vals, input_values)

            # Check if clamping would change indices (indicating invalid values)
            indices_clamped = indices.clamp(0, c - 1)
            if not (indices == indices_clamped).all():
                invalid_mask = indices != indices_clamped
                raise ValueError(
                    f"Variable {i} contains values not seen during training: "
                    f"{input_values[invalid_mask].unique().tolist()}. "
                    f"Valid values are: {unique_vals.tolist()}"
                )

            mapped[..., i] = indices
        return mapped

    def _map_indices_to_values(self, indices: Tensor) -> Tensor:
        """Map 0-based indices back to original categorical values."""
        mapped = indices.clone()
        for i in range(self.num_variables):
            c = int(self.num_categories_per_var[i])
            lookup = self.values_lookup[i, :c]
            mapped[..., i] = lookup[indices[..., i].long()]
        return mapped

    def forward(self, input: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass of the categorical density estimator network to compute the
        conditional density at a given time.

        Args:
            input: Inputs datapoints of shape `(batch_size, *input_shape)`
            condition: Conditioning variable. `(batch_size, *condition_shape)`

        Returns:
            Predicted categorical logits. `(batch_size, *input_shape,
                num_categories)`
        """
        embedded_condition = self.embedding_net.forward(condition)
        out = super().forward(input, context=embedded_condition)
        # masks out logits i.e. for variables with num_categories < max(num_categories)
        return out.masked_fill(~self.mask.bool().flatten(), float("-inf"))

    def log_prob(self, input: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        r"""Return log-probability of samples.

        Evaluates `Categorical.log_prob`. The logits are given by the MADE.

        Args:
            input: Input datapoints of shape `(batch_size, *input_shape)`.
            condition: Conditioning variable. `(batch_size, *condition_shape)`.

        Returns:
            Log-probabilities of shape `(batch_size,)`.
        """
        if self._needs_mapping:
            mapped_input = self._map_values_to_indices(input)
        else:
            mapped_input = input

        outputs = self.forward(mapped_input, condition=condition)

        outputs = outputs.reshape(*mapped_input.shape, self.max_num_categories)
        log_prob = Categorical(logits=outputs).log_prob(mapped_input).sum(dim=-1)

        return log_prob

    def sample(
        self, sample_shape: torch.Size, context: Optional[Tensor] = None
    ) -> Tensor:
        """Sample from the conditional categorical distribution.

        Autoregressively samples from the conditional categorical distribution.
        Calls `Categorical.sample`. The logits are given by the MADE.

        Args:
            sample_shape: Shape of samples.
            context: Conditioning variable. `(batch_dim, *condition_shape)`.

        Returns:
            Samples of shape `(*sample_shape, batch_dim)`.
        """
        num_samples = int(torch.prod(torch.tensor(sample_shape)))

        # Prepare context
        if context is not None:
            batch_dim = context.shape[0]
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if batch_dim == 1:
                context = torchutils.repeat_rows(context, num_samples)
        else:
            context_dim = 0 if self.context_features is None else self.context_features
            context = torch.zeros(num_samples, context_dim)
            batch_dim = 1

        # Autoregressively sample from the conditional categorical distribution.
        # for i = 1, ..., num_variables:
        #   x_i ~ Categorical(logits=f_i(x_1, ..., x_{i-1}, c))
        with torch.no_grad():
            # Zeros is fine here because values will be masked out anyway.
            samples = torch.zeros(
                num_samples, batch_dim, self.num_variables, device=context.device
            )
            for i in range(self.num_variables):
                # generate outputs given all previously sampled variables.
                outputs = self.forward(samples, context)
                outputs = outputs.reshape(*samples.shape, self.max_num_categories)
                # Select logits and sample only for variable i
                samples[:, :, i] = Categorical(logits=outputs[:, :, i, :]).sample()

        # Map sampled indices back to original categorical values if needed
        if self._needs_mapping:
            samples = self._map_indices_to_values(samples)
        return samples.reshape(*sample_shape, batch_dim, self.num_variables)


class CategoricalMassEstimator(ConditionalDensityEstimator):
    """Conditional density (mass) estimation for a categorical random variable."""

    def __init__(
        self, net: CategoricalMADE, input_shape: torch.Size, condition_shape: torch.Size
    ) -> None:
        """Initialize the mass estimator.

        Args:
            net: CategoricalMADE.
            input_shape: Shape of the input data.
            condition_shape: Shape of the condition data
        """
        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )
        self.net = net
        self.num_categories = net.max_num_categories

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Return log-probability of samples under the categorical distribution.

        Args:
            input: Input datapoints of shape
                `(sample_dim, batch_dim, *event_shape_input)`.Must be a discrete
                indicator of class identity.
            condition: Conditions of shape `(batch_dim, *event_shape_condition)`.

        Returns:
            Log-probabilities of shape `(sample_dim, batch_dim)`.
        """
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        # The CatetoricalNet can actually handle torch shape conventions and
        # just returns log-probabilities of shape `(sample_dim, batch_dim)`.
        return self.net.log_prob(input, condition, **kwargs)

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        """Return samples from the conditional categorical distribution.

        Args:
            sample_shape: Shape of samples.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.

        Returns:
            Samples of shape `(*sample_shape, batch_dim_condition)`. Note that the
            `CategoricalMassEstimator` is defined to have `event_shape=()` and
            therefore `.sample()` does not return a trailing dimension for
            `event_shape`.
        """
        return self.net.sample(sample_shape, condition, **kwargs)

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs of shape `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)`.

        Returns:
            Loss of shape `(batch_dim,)`
        """

        return -self.log_prob(input.unsqueeze(0), condition)[0]
