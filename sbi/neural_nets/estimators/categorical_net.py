# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Optional

import torch
from nflows.nn.nde.made import MADE
from nflows.utils import torchutils
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator


class CategoricalMADE(MADE):
    """Conditional density (mass) estimation for a n-dim categorical random variable.

    Takes as input parameters theta and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        num_categories: Tensor,  # Tensor[int]
        hidden_features: int,
        context_features: Optional[int] = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        random_mask: bool = False,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        epsilon: float = 1e-2,
        custom_initialization: bool = True,
        embedding_net: Optional[nn.Module] = nn.Identity(),
    ):
        """Initialize the neural net.

        Args:
            num_categories: number of categories for each variable. len(categories)
                defines the number of input units, i.e., dimensionality of the features.
                max(categories) defines the number of output units, i.e., the largest
                number of categories.
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            embedding_net: emebedding net for input.
        """
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")

        self.num_variables = len(num_categories)
        self.num_categories = int(torch.max(num_categories))
        self.mask = torch.zeros(self.num_variables, self.num_categories)
        for i, c in enumerate(num_categories):
            self.mask[i, :c] = 1

        super().__init__(
            self.num_variables,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self.num_categories,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        self.embedding_net = embedding_net
        self.hidden_features = hidden_features
        self.epsilon = epsilon
        self.context_features = context_features

        if custom_initialization:
            self._initialize()

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass of the categorical density estimator network to compute the
        conditional density at a given time.

        Args:
            input: Original data, x0. (batch_size, *input_shape)
            condition: Conditioning variable. (batch_size, *condition_shape)

        Returns:
            Predicted categorical logits. (batch_size, *input_shape,
                num_categories)
        """
        embedded_context = self.embedding_net.forward(context)
        return super().forward(inputs, context=embedded_context)

    def compute_probs(self, outputs):
        ps = F.softmax(outputs, dim=-1) * self.mask
        ps = ps / ps.sum(dim=-1, keepdim=True)
        return ps

    def log_prob(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        r"""Return log-probability of samples.

        Args:
            input: Input datapoints of shape `(batch_size, *input_shape)`.
            context: Context of shape `(batch_size, *condition_shape)`.

        Returns:
            Log-probabilities of shape `(batch_size, num_variables, num_categories)`.
        """
        outputs = self.forward(inputs, context=context)
        outputs = outputs.reshape(*inputs.shape, self.num_categories)
        ps = self.compute_probs(outputs)

        # categorical log prob
        log_prob = torch.log(ps.gather(-1, inputs.unsqueeze(-1).long()))
        log_prob = log_prob.squeeze(-1).sum(dim=-1)

        return log_prob

    def sample(self, sample_shape, context=None):
        # Ensure sample_shape is a tuple
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample_shape = torch.Size(sample_shape)

        # Calculate total number of samples
        num_samples = int(torch.prod(torch.tensor(sample_shape)))

        # Prepare context
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            context = torchutils.repeat_rows(context, num_samples)
        else:
            context = torch.zeros(num_samples, self.context_features)

        with torch.no_grad():
            samples = torch.zeros(num_samples, self.num_variables)
            for i in range(self.num_variables):
                outputs = self.forward(samples, context)
                outputs = outputs.reshape(*samples.shape, self.num_categories)
                ps = self.compute_probs(outputs)
                samples[:, i] = Categorical(probs=ps[:, i]).sample()

        return samples.reshape(*sample_shape, self.num_variables)

    def _initialize(self):
        pass


class CategoricalMassEstimator(ConditionalDensityEstimator):
    """Conditional density (mass) estimation for a categorical random variable.

    The event_shape of this class is `()`.
    """

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
        self.num_categories = net.num_categories

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """Return log-probability of samples.

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
