# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import torch
from torch import Tensor, nn, distributions
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from nflows.nn.nde.made import MADE
from torch.nn import functional as F
from nflows.utils import torchutils
import numpy as np


class CategoricalMADE(MADE):
    def __init__(
        self,
        categories, # List[int] or Tensor[int]
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        epsilon=1e-2,
        custom_initialization=True,
        embedding_net: Optional[nn.Module] = nn.Identity(),
    ):

        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")

        self.num_variables = len(categories)
        self.num_categories = int(max(categories))
        self.categories = categories
        self.mask = torch.zeros(self.num_variables, self.num_categories)
        for i, c in enumerate(categories):
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

        if custom_initialization:
            self._initialize()

    def forward(self, inputs, context=None):
        embedded_inputs = self.embedding_net.forward(inputs)
        return super().forward(embedded_inputs, context=context)
    
    def compute_probs(self, outputs):
        ps = F.softmax(outputs, dim=-1)*self.mask
        ps = ps / ps.sum(dim=-1, keepdim=True)
        return ps
        
    # outputs (batch_size, num_variables, num_categories)
    def log_prob(self, inputs, context=None):
        outputs = self.forward(inputs, context=context)
        outputs = outputs.reshape(*inputs.shape, self.num_categories)
        ps = self.compute_probs(outputs)
        
        # categorical log prob
        log_prob = torch.log(ps.gather(-1, inputs.unsqueeze(-1).long()))
        log_prob = log_prob.sum(dim=-1)
       
        return log_prob

    def sample(self, sample_shape, context=None):
        # Ensure sample_shape is a tuple
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample_shape = torch.Size(sample_shape)
        
        # Calculate total number of samples
        num_samples = torch.prod(torch.tensor(sample_shape)).item()
        
        # Prepare context
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            context = torchutils.repeat_rows(context, num_samples)
        else:
            context = torch.zeros(num_samples, self.context_dim)
        
        with torch.no_grad():
            samples = torch.zeros(num_samples, self.num_variables)
            for variable in range(self.num_variables):
                outputs = self.forward(samples, context)
                outputs = outputs.reshape(num_samples, self.num_variables, self.num_categories)
                ps = self.compute_probs(outputs)
                samples[:, variable] = Categorical(probs=ps[:,variable]).sample()
        
        return samples.reshape(*sample_shape, self.num_variables)

    def _initialize(self):
        pass

class CategoricalNet(nn.Module):
    """Conditional density (mass) estimation for a categorical random variable.

    Takes as input parameters theta and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        num_input: int,
        num_categories: int,
        num_hidden: int = 20,
        num_layers: int = 2,
        embedding_net: Optional[nn.Module] = None,
    ):
        """Initialize the neural net.

        Args:
            num_input: number of input units, i.e., dimensionality of the features.
            num_categories: number of output units, i.e., number of categories.
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            embedding_net: emebedding net for input.
        """
        super().__init__()

        self.num_hidden = num_hidden
        self.num_input = num_input
        self.activation = Sigmoid()
        self.softmax = Softmax(dim=1)
        self.num_categories = num_categories

        # Maybe add embedding net in front.
        if embedding_net is not None:
            self.input_layer = nn.Sequential(
                embedding_net, nn.Linear(num_input, num_hidden)
            )
        else:
            self.input_layer = nn.Linear(num_input, num_hidden)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden, num_hidden))

        self.output_layer = nn.Linear(num_hidden, num_categories)

    def forward(self, condition: Tensor) -> Tensor:
        """Return categorical probability predicted from a batch of inputs.

        Args:
            condition: batch of context parameters for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """
        # forward path
        condition = self.activation(self.input_layer(condition))

        # iterate n hidden layers, input condition and calculate tanh activation
        for layer in self.hidden_layers:
            condition = self.activation(layer(condition))

        return self.softmax(self.output_layer(condition))

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        """Return categorical log probability of categories input, given condition.

        Args:
            input: categories to evaluate.
            condition: parameters.

        Returns:
            Tensor: log probs with shape (input.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(condition)
        # Squeeze the last dimension (event dim) because `Categorical` has
        # `event_shape=()` but our data usually has an event_shape of `(1,)`.
        return Categorical(probs=ps).log_prob(input.squeeze(dim=-1))

    def sample(self, sample_shape: torch.Size, condition: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.

        Args:
            sample_shape: number of samples to obtain.
            condition: batch of parameters for prediction.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(condition)
        return Categorical(probs=ps).sample(sample_shape=sample_shape)


class CategoricalMassEstimator(ConditionalDensityEstimator):
    """Conditional density (mass) estimation for a categorical random variable.

    The event_shape of this class is `()`.
    """

    def __init__(
        self, net: CategoricalNet, input_shape: torch.Size, condition_shape: torch.Size
    ) -> None:
        """Initialize the mass estimator.

        Args:
            net: CategoricalNet.
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
