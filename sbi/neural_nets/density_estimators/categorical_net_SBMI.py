from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax


class CategoricalNet(nn.Module):
    """Class to perform conditional density (mass) estimation for a categorical RV.

    Takes as input data x and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    ---
    adapted from:
    SBI package
    https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/mnle.py#L209
    """

    def __init__(
        self,
        num_input: int = 4,
        num_categories: int = 2,
        num_hidden: int = 20,
        num_layers: int = 2,
        binary_vector_input: bool = False,
        embedding: Optional[nn.Module] = None,
    ):
        """Initialize the neural net.

        Args:
            num_input: number of input units, i.e., dimensionality of parameters.
            num_categories: number of output units, i.e., number of categories.
                (2^n for binary encoded vector of lenght n)
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            binary_vector_input: whether input is a binary vector instead of category.
            embedding: emebedding net for parameters, e.g., a z-scoring transform.
        """
        super(CategoricalNet, self).__init__()

        self.num_hidden = num_hidden
        self.num_input = num_input
        self.activation = Sigmoid()
        self.softmax = Softmax(dim=1)
        self.num_categories = num_categories
        self.binary_vector_input = binary_vector_input

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

    def forward(self, cat: Tensor, context: Tensor) -> Tensor:
        """
        alias for self.log_prob

        Args:
            cat: categories
            context: batch of input data for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """

        return self.log_prob(cat, context)

    def get_dist_params(self, x: Tensor) -> Tensor:
        """
        evaluates tes
        Args:
            x: batch of input data for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """
        assert x.dim() == 2, "input needs to have a batch dimension."
        assert (
            x.shape[1] == self.num_input
        ), f"input dimensions must match num_input {self.num_input}"

        # forward path
        x = self.activation(self.input_layer(x))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.softmax(self.output_layer(x))

    def log_prob(self, cat: Tensor, x: Tensor) -> Tensor:
        """Return categorical log probability of categories cat, given data x.

        Args:
            x: data.
            cat: categories to evaluate.

        Returns:
            Tensor: log probs with shape (cat.shape[0],)
        """
        # convert to decimal if necessary
        if self.binary_vector_input:
            cat = bin2dec(cat)

        # Predict categorical ps and evaluate.

        ps = self.get_dist_params(x)

        return Categorical(probs=ps).log_prob(cat.squeeze())

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.

        Args:
            context: batch of data for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.get_dist_params(context)
        prediction = (
            Categorical(probs=ps)
            .sample(torch.Size((num_samples,)))
            .reshape(num_samples, -1)
        )

        if self.binary_vector_input:
            bits = int(torch.log2(torch.tensor([self.num_categories])))
            prediction = dec2bin(prediction, bits)

        return prediction


def dec2bin(x, bits):
    """converts x to a binary vector

    Args:
        x (int tensor): if not int, it gets converted to int8 type
        bits (int): _description_

    Returns:
        _type_: _description_
    """
    x = x.type(torch.int8)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def bin2dec(b):
    """converts binary vector to integer value

    Args:
        b (tensor): binary vector

    Returns:
        tensor: int tensor of same shabe as b
    """
    bits = b.shape[-1]
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
