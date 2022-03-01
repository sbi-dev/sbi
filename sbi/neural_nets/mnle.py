# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from typing import Optional, Tuple

import torch
from pyknos.nflows import flows
from torch import Tensor, nn, unique
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax

from sbi.neural_nets.flow import build_nsf
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, standardizing_net
from sbi.utils.torchutils import atleast_2d
from sbi.utils.user_input_checks import check_data_device


def build_mnle(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    num_transforms: int = 2,
    num_bins: int = 5,
    hidden_features: int = 50,
    hidden_layers: int = 2,
    tail_bound: float = 10.0,
    log_transform_x: bool = True,
    **kwargs,
):
    """Returns a density estimator for mixed data types.

    Uses a categorical net to model the discrete part and a neural spline flow (NSF) to
    model the continuous part of the data.

    Args:
        batch_x: batch of data
        batch_y: batch of parameters
        z_score_x: whether to z-score x.
        z_score_y: whether to z-score y.
        num_transforms: number of transforms in the NSF
        num_bins: bins per spline for NSF.
        hidden_features: number of hidden features used in both nets.
        hidden_layers: number of hidden layers in the categorical net.
        tail_bound: spline tail bound for NSF.
        log_transform_x: whether to apply a log-transform to x to move it to unbounded
            space, e.g., in case x consists of reaction time data (bounded by zero).

    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE.
    """

    check_data_device(batch_x, batch_y)
    if z_score_y == "independent":
        embedding = standardizing_net(batch_y)
    else:
        embedding = None

    warnings.warn(
        """The mixed neural likelihood estimator assumes that x contains
        continuous data in the first n-1 columns (e.g., reaction times) and
        categorical data in the last column (e.g., corresponding choices). If
        this is not the case for the passed `x` do not use this function."""
    )
    # Separate continuous and discrete data.
    cont_x, disc_x = _separate_x(batch_x)

    # Infer input and output dims.
    dim_parameters = batch_y[0].numel()
    num_categories = unique(disc_x).numel()

    # Set up a categorical RV neural net for modelling the discrete data.
    disc_nle = CategoricalNet(
        num_input=dim_parameters,
        num_categories=num_categories,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding=embedding,
    )

    # Set up a NSF for modelling the continuous data, conditioned on the discrete data.
    cont_nle = build_nsf(
        batch_x=torch.log(cont_x)
        if log_transform_x
        else cont_x,  # log transform manually.
        batch_y=torch.cat((batch_y, disc_x), dim=1),  # condition on discrete data too.
        z_score_y=z_score_y,
        z_score_x=z_score_x,
        num_bins=num_bins,
        num_transforms=num_transforms,
        tail_bound=tail_bound,
        hidden_features=hidden_features,
    )

    return MixedDensityEstimator(
        discrete_net=disc_nle,
        continuous_net=cont_nle,
        log_transform_x=log_transform_x,
    )


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
            num_input: number of input units, i.e., dimensionality of parameters.
            num_categories: number of output units, i.e., number of categories.
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            embedding: emebedding net for parameters, e.g., a z-scoring transform.
        """
        super(CategoricalNet, self).__init__()

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

    def forward(self, theta: Tensor) -> Tensor:
        """Return categorical probability predicted from a batch of parameters.

        Args:
            theta: batch of input parameters for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """
        assert theta.dim() == 2, "input needs to have a batch dimension."
        assert (
            theta.shape[1] == self.num_input
        ), f"input dimensions must match num_input {self.num_input}"

        # forward path
        theta = self.activation(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation(layer(theta))

        return self.softmax(self.output_layer(theta))

    def log_prob(self, x: Tensor, theta: Tensor) -> Tensor:
        """Return categorical log probability of categories x, given parameters theta.

        Args:
            theta: parameters.
            x: categories to evaluate.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(theta)
        return Categorical(probs=ps).log_prob(x.squeeze())

    def sample(self, num_samples: int, theta: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(theta)
        return (
            Categorical(probs=ps)
            .sample(torch.Size((num_samples,)))
            .reshape(num_samples, -1)
        )


class MixedDensityEstimator(nn.Module):
    """Class performing Mixed Neural Likelihood Estimation.

    MNLE combines a Categorical net and a neural spline flow to model data with
    mixed types, e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        discrete_net: CategoricalNet,
        continuous_net: flows.Flow,
        log_transform_x: bool = False,
    ):
        """Initialize class for combining density estimators for MNLE.

        Args:
            discrete_net: neural net to model discrete part of the data.
            continuous_net: neural net to model the continuous data.
            log_transform_x: whether to transform the continous part of the data into
                logarithmic domain before training. This is helpful for bounded data, e.
                g.,for reaction times.
        """
        super(MixedDensityEstimator, self).__init__()

        self.discrete_net = discrete_net
        self.continuous_net = continuous_net
        self.log_transform_x = log_transform_x

    def forward(self, x: Tensor):
        raise NotImplementedError(
            """The forward method is not implemented for MNLE, use '.sample(...)' to
            generate samples though a forward pass."""
        )

    def sample(
        self, theta: Tensor, num_samples: int = 1, track_gradients: bool = False
    ) -> Tensor:
        """Return sample from mixed data distribution.

        Args:
            theta: parameters for which to generate samples.
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples with shape (num_samples, num_data_dimensions)
        """
        assert theta.shape[0] == 1, "Samples can be generated for a single theta only."

        with torch.set_grad_enabled(track_gradients):

            # Sample discrete data given parameters.
            discrete_x = self.discrete_net.sample(
                theta=theta,
                num_samples=num_samples,
            ).reshape(num_samples, 1)

            # Sample continuous data condition on parameters and discrete data.
            # Pass num_samples=1 because the choices in the context contains
            # num_samples elements already.
            continuous_x = self.continuous_net.sample(
                # repeat the single theta to match number of sampled choices.
                context=torch.cat((theta.repeat(num_samples, 1), discrete_x), dim=1),
                num_samples=1,
            ).reshape(num_samples, 1)

            # In case x was log-transformed, move them to linear space.
            if self.log_transform_x:
                continuous_x = continuous_x.exp()

        return torch.cat((continuous_x, discrete_x), dim=1)

    def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
        """Return log-probability of samples under the learned MNLE.

        For a fixed data point x this returns the value of the likelihood function
        evaluated at theta, L(theta, x=x).

        Alternatively, it can be interpreted as the log-prob of the density
        p(x | theta).

        It evaluates the separate density estimator for the discrete and continous part
        of the data and then combines them into one evaluation.

        Args:
            x: data (containing continuous and discrete data).
            context: parameters for which to evaluate the likelihod function, or for
                which to condition p(x | theta).

        Returns:
            Tensor: log_prob of p(x | theta).
        """
        assert (
            x.shape[0] == context.shape[0]
        ), "x and context must have same batch size."

        cont_x, disc_x = _separate_x(x)
        num_parameters = context.shape[0]

        disc_log_prob = self.discrete_net.log_prob(x=disc_x, theta=context).reshape(
            num_parameters
        )

        cont_log_prob = self.continuous_net.log_prob(
            # Transform to log-space if needed.
            torch.log(cont_x) if self.log_transform_x else cont_x,
            # Pass parameters and discrete x as context.
            context=torch.cat((context, disc_x), dim=1),
        )

        # Combine into joint lp.
        log_probs_combined = (disc_log_prob + cont_log_prob).reshape(num_parameters)

        # Maybe add log abs det jacobian of RTs: log(1/x) = - log(x)
        if self.log_transform_x:
            log_probs_combined -= torch.log(cont_x).squeeze()

        return log_probs_combined

    def log_prob_iid(self, x: Tensor, theta: Tensor) -> Tensor:
        """Return log prob given a batch of iid x and a different batch of theta.

        This is different from `.log_prob()` to enable speed ups in evaluation during
        inference. The speed up is achieved by exploiting the fact that there are only
        finite number of possible categories in the discrete part of the dat: one can
        just calculate the log probs for each possible category (given the current batch
        of theta) and then copy those log probs into the entire batch of iid categories.
        For example, for the drift-diffusion model, there are only two choices, but
        often 100s or 1000 trials. With this method a evaluation over trials then passes
        a batch of `2 (one per choice) * num_thetas` into the NN, whereas the normal
        `.log_prob()` would pass `1000 * num_thetas`.

        Args:
            x: batch of iid data, data observed given the same underlying parameters or
                experimental conditions.
            theta: batch of parameters to be evaluated, i.e., each batch entry will be
                evaluated for the entire batch of iid x.

        Returns:
            Tensor: log probs with shape (num_trials, num_parameters), i.e., the log
                prob for each theta for each trial.
        """

        theta = atleast_2d(theta)
        x = atleast_2d(x)
        batch_size = theta.shape[0]
        num_trials = x.shape[0]
        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, x)
        net_device = next(self.discrete_net.parameters()).device
        assert (
            net_device == x.device and x.device == theta.device
        ), f"device mismatch: net, x, theta: {net_device}, {x.device}, {theta.device}."

        x_cont_repeated, x_disc_repeated = _separate_x(x_repeated)
        x_cont, x_disc = _separate_x(x)

        log_prob_per_cat = torch.zeros(self.discrete_net.num_categories, batch_size)
        # repeat categories for parameters
        repeated_categories = torch.repeat_interleave(
            torch.arange(self.discrete_net.num_categories - 1), batch_size, dim=0
        )
        # repeat parameters for categories
        repeated_theta = theta.repeat(self.discrete_net.num_categories - 1, 1)
        log_prob_per_cat[:-1, :] = self.discrete_net.log_prob(
            repeated_categories,
            repeated_theta,
        ).reshape(-1, batch_size)
        # infer the last category logprob from sum to one.
        log_prob_per_cat[-1, :] = torch.log(1 - log_prob_per_cat[:-1, :].exp().sum(0))

        # fill in lps for each occurred category
        log_probs_discrete = log_prob_per_cat[
            x_disc.type_as(torch.zeros(1, dtype=torch.long)).squeeze()
        ].reshape(-1)

        # Get repeat discrete data and theta to match in batch shape for flow eval.
        log_probs_cont = self.continuous_net.log_prob(
            torch.log(x_cont_repeated) if self.log_transform_x else x_cont_repeated,
            context=torch.cat((theta_repeated, x_disc_repeated), dim=1),
        )

        # Combine into joint lp with first dim over trials.
        log_probs_combined = (log_probs_discrete + log_probs_cont).reshape(
            num_trials, batch_size
        )

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.log_transform_x:
            log_probs_combined -= torch.log(x_cont)

        # Return batch over trials as required by SBI potentials.
        return log_probs_combined


def _separate_x(x: Tensor, num_discrete_columns: int = 1) -> Tuple[Tensor, Tensor]:
    """Returns the continuous and discrete part of the given x.

    Assumes the discrete data to live in the last columns of x.
    """

    assert x.ndim == 2, f"x must have two dimensions but has {x.ndim}."

    return x[:, :-num_discrete_columns], x[:, -num_discrete_columns:]
