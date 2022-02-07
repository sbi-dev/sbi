# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import warnings
from typing import Optional

import torch
from torch import Tensor, nn, unique
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax

from sbi.neural_nets.flow import build_nsf
from sbi.utils.sbiutils import standardizing_net
from sbi.utils.user_input_checks import check_data_device


def build_mnle(
    batch_x,
    batch_y,
    z_score_x: bool = False,
    z_score_y: bool = False,
    log_transform_x: bool = True,
    num_transforms: int = 2,
    num_bins: int = 5,
    tail_bound: float = 10.0,
    hidden_features: int = 10,
    hidden_layers: int = 2,
    **kwargs,
):

    check_data_device(batch_x, batch_y)
    if z_score_y:
        embedding = standardizing_net(batch_y)
    else:
        embedding = None

    warnings.warn(
        """The mixed neural likelihood estimator assumes that x contains
                  continuous data in the first n-1 columns (e.g., reation times) and
                  categorical data in the last column (e.g., corresponding choices). If
                  this is not the case for the passed `x` do not use this function."""
    )
    disc_x = batch_x[:, -1:]
    cont_x = batch_x[:, :-1]

    dim_parameters = batch_y[0].numel()
    num_categories = unique(disc_x).numel()

    disc_nle = CategoricalNet(
        num_input=dim_parameters,
        num_categories=num_categories,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding=embedding,
    )

    cont_nle = build_nsf(
        batch_x=torch.log(cont_x)
        if log_transform_x
        else cont_x,  # log transform manually.
        batch_y=torch.cat((batch_y, disc_x), dim=1),
        z_score_y=z_score_y,
        z_score_x=z_score_x,
        num_bins=num_bins,
        num_transforms=num_transforms,
        tail_bound=tail_bound,
        hidden_features=hidden_features,
    )

    return MNLE(
        discrete_net=disc_nle,
        continuous_net=cont_nle,
        log_transform_x=log_transform_x,
    )


class MNLE(nn.Module):
    """Class for Mixed Neural Likelihood Estimation.

    MNLE combines a Categorical net and a flow over continuous data to model data with
    mixed types, e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        discrete_net: nn.Module,
        continuous_net: nn.Module,
        log_transform_x: bool = False,
    ):
        """Initializa synthetic likelihood class from a choice net and reaction time
        flow.

        Args:
            discrete_net: neural net to model discrete part of the data.
            continuous_net: neural net to model the continuous data.
            log_transform_x: whether to transform the continous part of the data into
                logarithmic domain before training. This is helpful for bounded data, e.
                g.,for reaction times.
        """
        super(MNLE, self).__init__()

        self.choice_net = discrete_net
        self.rt_net = continuous_net
        self.use_log_rts = log_transform_x

    def forward(self, x):
        # the input x consists of parameters theta
        return self.sample(theta=x, track_gradients=False)

    def sample(
        self,
        theta: Tensor,
        num_samples: int = 1,
        track_gradients: bool = False,
    ) -> Tensor:
        """Return choices and reaction times given DDM parameters.

        Args:
            theta: DDM parameters, shape (batch, 4)
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples (rt, choice) with shape (num_samples, 2)
        """
        assert theta.shape[0] == 1, "for samples, no batching in theta is possible yet."

        with torch.set_grad_enabled(track_gradients):

            # Sample choices given parameters, from BernoulliMN.
            choices = self.choice_net.sample(num_samples, context=theta).reshape(
                num_samples, 1
            )
            # Pass num_samples=1 because the choices in the context contains
            # num_samples elements already.
            rts = self.rt_net.sample(
                num_samples=1,
                # repeat the single theta to match number of sampled choices.
                context=torch.cat((theta.repeat(num_samples, 1), choices), dim=1),
            ).reshape(num_samples, 1)
            if self.use_log_rts:
                rts = rts.exp()

        return torch.cat((rts, choices), dim=1)

    def log_prob(
        self,
        x: Tensor,
        context: Tensor,
    ) -> Tensor:
        assert (
            x.shape[0] == context.shape[0]
        ), "x and context must have same batch size."
        assert x.shape[1] > 1

        cont_x = x[:, :-1]
        disc_x = x[:, -1:]
        theta = context
        num_parameters = theta.shape[0]

        disc_log_prob = self.choice_net.log_prob(x=disc_x, context=theta).reshape(
            num_parameters
        )

        # Get rt log probs from rt net.
        cont_log_prob = self.rt_net.log_prob(
            torch.log(cont_x) if self.use_log_rts else cont_x,
            context=torch.cat((theta, disc_x), dim=1),
        )

        # Combine into joint lp with first dim over trials.
        lp_combined = (disc_log_prob + cont_log_prob).reshape(num_parameters)

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.use_log_rts:
            lp_combined -= torch.log(cont_x).squeeze()

        return lp_combined

    def log_prob_iidtrials(self, x, context):

        # Extract unique values to undo trial-parameter-batch matching.
        theta = torch.unique(context, sorted=False, dim=0)
        num_parameters = theta.shape[0]
        num_categories = self.choice_net.num_categories
        x_unique = torch.unique(x, sorted=False, dim=0)
        num_trials = x_unique.shape[0]

        assert x_unique.ndim > 1
        assert (
            x_unique.shape[1] == 2
        ), "MNLE assumes x to have two columns: [rts; choices]"

        rts_repeated = x[:, 0:1]
        choices_repeated = x[:, 1:2]
        rts = x_unique[:, 0:1]
        choices = x_unique[:, 1:2]

        # Get choice log probs from categorical net.
        # Apply efficiency trick due to discreteness of categories:
        # For iid trials the log probs are identical for each category.
        # Calculate lp for all but one possible category once for each parameter.
        log_prob_per_cat = torch.zeros(num_categories, num_parameters)
        log_prob_per_cat[:-1, :] = self.choice_net.log_prob(
            # repeat categories for parameters
            torch.repeat_interleave(
                torch.arange(num_categories - 1), num_parameters, dim=0
            ),
            # repeat parameters for categories
            theta.repeat(num_categories - 1, 1),
        ).reshape(-1, num_parameters)
        # infer the last category lp from the softmax condition.
        log_prob_per_cat[-1, :] = torch.log(1 - log_prob_per_cat.exp().sum(0))

        # fill in lps for each occurred category
        lp_choices = log_prob_per_cat[
            choices.type_as(torch.zeros(1, dtype=torch.int)).squeeze()
        ].reshape(-1)

        # Get rt log probs from rt net.
        lp_rts = self.rt_net.log_prob(
            torch.log(rts_repeated) if self.use_log_rts else rts_repeated,
            context=torch.cat((context, choices_repeated), dim=1),
        )

        # Combine into joint lp with first dim over trials.
        lp_combined = (lp_choices + lp_rts).reshape(num_trials, num_parameters)

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.use_log_rts:
            lp_combined -= torch.log(rts)
        # Set to lower bound where reaction happend before non-decision time tau.
        log_likelihood_trial_batch = torch.where(
            # If rt < tau the likelihood should be zero (or at lower bound).
            rts.repeat(1, num_parameters) > theta[:, -1],
            lp_combined,
            1e-19 * torch.ones_like(lp_combined),
        )

        # Return batch over trials as required by SBI potentials.
        return log_likelihood_trial_batch


class CategoricalNet(nn.Module):
    """Net for learning a conditional Bernoulli mass function over choices given parameters.

    Takes as input parameters theta and learns the parameter p of a Bernoulli.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        num_input: int = 4,
        num_categories: int = 2,
        num_hidden: int = 20,
        num_layers: int = 2,
        embedding: nn.Module = None,
    ):
        """Initialize density estimator for categorical data.

        Args:

        """
        super(CategoricalNet, self).__init__()

        self.num_hidden = num_hidden
        self.num_input = num_input
        self.activation = Sigmoid()
        self.softmax = Softmax(dim=1)

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
            Tensor: batch of predicted Bernoulli probabilities.
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

    def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
        """Return categorical log probability of categories x, given parameters theta.

        Args:
            theta: parameters.
            x: choices to evaluate.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(context)
        return Categorical(probs=ps).log_prob(x.squeeze())

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural bet.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(context)
        return Categorical(probs=ps).sample((num_samples,)).reshape(num_samples, -1)
