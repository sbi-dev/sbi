# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional, Tuple

import torch
from torch import Tensor

from sbi.neural_nets.density_estimators import (
    CategoricalMassEstimator,
    DensityEstimator,
)
from sbi.neural_nets.density_estimators.nflows_flow import NFlowsFlow
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


class MixedDensityEstimator(DensityEstimator):
    """Class performing Mixed Neural Likelihood Estimation.

    MNLE combines a Categorical net and a neural spline flow to model data with
    mixed types, e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        discrete_net: CategoricalMassEstimator,
        continuous_net: NFlowsFlow,
        log_transform_x: bool = False,
        condition_shape: Optional[torch.Size] = None,
    ):
        """Initialize class for combining density estimators for MNLE.

        Args:
            discrete_net: neural net to model discrete part of the data.
            continuous_net: neural net to model the continuous data.
            log_transform_x: whether to transform the continous part of the data into
                logarithmic domain before training. This is helpful for bounded data, e.
                g.,for reaction times.
        """
        super(MixedDensityEstimator, self).__init__(
            net=None, condition_shape=condition_shape
        )

        self.discrete_net = discrete_net
        self.continuous_net = continuous_net
        self.log_transform_x = log_transform_x

    def forward(self, x: Tensor):
        raise NotImplementedError(
            """The forward method is not implemented for MNLE, use '.sample(...)' to
            generate samples though a forward pass."""
        )

    def sample(
        self, theta: Tensor, sample_shape: torch.Size, track_gradients: bool = False
    ) -> Tensor:
        """Return sample from mixed data distribution.

        Args:
            theta: parameters for which to generate samples.
            sample_shape number of samples to generate.

        Returns:
            Tensor: samples with shape (num_samples, num_data_dimensions)
        """
        assert theta.shape[0] == 1, "Samples can be generated for a single theta only."

        with torch.set_grad_enabled(track_gradients):
            # Sample discrete data given parameters.
            discrete_x = self.discrete_net.sample(
                sample_shape=sample_shape,
                context=theta,
                # num_samples=num_samples,
            )  # .reshape(num_samples, 1)

            # Sample continuous data condition on parameters and discrete data.
            # Pass num_samples=1 because the choices in the context contains
            # num_samples elements already.
            continuous_x = self.continuous_net.sample(
                # repeat the single theta to match number of sampled choices.
                # sample_shape[0] is the iid dimension.
                condition=torch.cat(
                    (theta.repeat(sample_shape[0], 1), discrete_x), dim=1
                ),
                sample_shape=sample_shape,
            )  # .reshape(num_samples, 1)

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

        disc_log_prob = self.discrete_net.log_prob(
            input=disc_x, context=context
        ).reshape(num_parameters)

        cont_log_prob = self.continuous_net.log_prob(
            # Transform to log-space if needed.
            torch.log(cont_x) if self.log_transform_x else cont_x,
            # Pass parameters and discrete x as context.
            condition=torch.cat((context, disc_x), dim=1),
        )

        # Combine into joint lp.
        log_probs_combined = (disc_log_prob + cont_log_prob).reshape(num_parameters)

        # Maybe add log abs det jacobian of RTs: log(1/x) = - log(x)
        if self.log_transform_x:
            log_probs_combined -= torch.log(cont_x).squeeze()

        return log_probs_combined

    def loss(self, x: Tensor, theta: Tensor, **kwargs) -> Tensor:
        return self.log_prob(x, theta)

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

        # repeat categories for parameters
        repeated_categories = torch.repeat_interleave(
            torch.arange(self.discrete_net.num_categories - 1), batch_size, dim=0
        )
        # repeat parameters for categories
        repeated_theta = theta.repeat(self.discrete_net.num_categories - 1, 1)
        log_prob_per_cat = torch.zeros(self.discrete_net.num_categories, batch_size).to(
            net_device
        )
        log_prob_per_cat[:-1, :] = self.discrete_net.log_prob(
            repeated_categories.to(net_device),
            repeated_theta.to(net_device),
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
            condition=torch.cat((theta_repeated, x_disc_repeated), dim=1),
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
