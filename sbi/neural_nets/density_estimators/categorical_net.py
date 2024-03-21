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
        assert context.dim() == 2, "context needs to have a batch dimension."
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
        return Categorical(probs=ps).log_prob(input.squeeze())

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
        return (
            Categorical(probs=ps)
            .sample(sample_shape=sample_shape)
            .reshape(sample_shape[0], -1)
        )


class CategoricalMassEstimator(DensityEstimator):
    """Class to perform conditional density (mass) estimation
    for a categorical RV.
    """

    def __init__(self, net: CategoricalNet) -> None:
        super().__init__(net=net, condition_shape=torch.Size([]))
        self.net = net
        self.num_categories = net.num_categories

    def log_prob(self, input: Tensor, context: Tensor, **kwargs) -> Tensor:
        return self.net.log_prob(input, context, **kwargs)

    def sample(self, sample_shape: torch.Size, context: Tensor, **kwargs) -> Tensor:
        return self.net.sample(sample_shape, context, **kwargs)

    def loss(self, input: Tensor, context: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs to evaluate the loss on of shape (batch_size, input_size).
            context: Conditions of shape (batch_size, *condition_shape).

        Returns:
            Loss of shape (batch_size,)
        """

        return -self.log_prob(input, context)
