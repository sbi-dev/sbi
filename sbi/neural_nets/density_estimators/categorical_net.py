from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax

from sbi.neural_nets.density_estimators import DensityEstimator


class CategoricalNet(nn.Module):
    """Class to perform conditional density (mass) estimation for a categorical RV.

    Takes as input parameters theta and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        num_input: int = 4,
        num_categories: int = 2,
        num_hidden: int = 20,
        num_layers: int = 2,
        embedding: Optional[nn.Module] = None,
    ):
        """Initialize the neural net.

        Args:
            num_input: number of input units, i.e., dimensionality of context.
            num_categories: number of output units, i.e., number of categories.
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            embedding: emebedding net for parameters, e.g., a z-scoring transform.
        """
        super().__init__()

        self.num_hidden = num_hidden
        self.num_input = num_input
        self.activation = Sigmoid()
        self.softmax = Softmax(dim=1)
        self.num_categories = num_categories

        # Maybe add z-score embedding for parameters.
        if embedding is not None:
            self.input_layer = nn.Sequential(
                embedding, nn.Linear(num_input, num_hidden)
            )
        else:
            self.input_layer = nn.Linear(num_input, num_hidden)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden, num_hidden))

        self.output_layer = nn.Linear(num_hidden, num_categories)

    def forward(self, context: Tensor) -> Tensor:
        """Return categorical probability predicted from a batch of inputs.

        Args:
            context: batch of context parameters for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """
        assert (
            context.dim() == 2
        ), f"context needs to have a batch dimension but its shape is {context.shape}."
        assert (
            context.shape[1] == self.num_input
        ), f"context dimensions must match num_input {self.num_input}"

        # forward path
        context = self.activation(self.input_layer(context))

        # iterate n hidden layers, input context and calculate tanh activation
        for layer in self.hidden_layers:
            context = self.activation(layer(context))

        return self.softmax(self.output_layer(context))

    def log_prob(self, input: Tensor, context: Tensor) -> Tensor:
        """Return categorical log probability of categories input, given context.

        Args:
            input: categories to evaluate.
            context: parameters.

        Returns:
            Tensor: log probs with shape (input.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(context)
        # Squeeze dim=1 because `Categorical` has `event_shape=()` but our data usually
        # has an event_shape of `(1,)`.
        return Categorical(probs=ps).log_prob(input.squeeze(dim=1))

    def sample(self, sample_shape: torch.Size, context: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.

        Args:
            sample_shape: number of samples to obtain.
            context: batch of parameters for prediction.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(context)
        return Categorical(probs=ps).sample(sample_shape=sample_shape)


class CategoricalMassEstimator(DensityEstimator):
    """Conditional density (mass) estimation for a categorical RV.

    The event_shape of this class is `()`.
    """

    def __init__(self, net: CategoricalNet) -> None:
        super().__init__(net=net, condition_shape=torch.Size([]))
        self.net = net
        self.num_categories = net.num_categories

    def log_prob(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        input_iid_dim = input.shape[0]
        input_batch_dim = input.shape[1]
        condition_batch_dim = condition.shape[0]
        condition_event_dims = len(condition.shape[1:])

        assert condition_batch_dim == input_batch_dim, (
            f"Batch shape of condition {condition_batch_dim} and input "
            f"{input_batch_dim} do not match."
        )

        # Nflows needs to have a single batch dimension for condition and input.
        input = input.reshape((input_batch_dim * input_iid_dim, -1))

        # Repeat the condition to match `input_batch_dim * input_iid_dim`.
        ones_for_event_dims = (1,) * condition_event_dims  # Tuple of 1s, e.g. (1, 1, 1)
        condition = condition.repeat(input_iid_dim, *ones_for_event_dims)

        return self.net.log_prob(input, condition, **kwargs).reshape((
            input_iid_dim,
            input_batch_dim,
        ))

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        """Return samples from the conditional categorical distribution.

        Args:
            sample_shape: Shape of samples.
            condition: Conditions. Of shape
                `(iid_dim_condition, batch_dim_condition, *event_shape_condition)`.

        Returns:
            Samples of shape (*sample_shape, batch_dim_condition). Note that the
            `CategoricalMassEstimator` is defined to have `event_shape=()` and
            therefore `.sample()` does not return a trailing dimension for
            `event_shape`.
        """
        return self.net.sample(sample_shape, condition, **kwargs)

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            condition: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Loss of shape (batch_size,)
        """

        return -self.log_prob(input, condition)
