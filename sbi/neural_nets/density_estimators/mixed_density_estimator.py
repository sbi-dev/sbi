# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Tuple

import torch
from torch import Tensor

from sbi.neural_nets.density_estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.density_estimators.categorical_net import CategoricalMassEstimator
from sbi.neural_nets.density_estimators.nflows_flow import NFlowsFlow
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes
from sbi.utils.torchutils import atleast_2d


class MixedDensityEstimator(ConditionalDensityEstimator):
    """Class performing Mixed Neural Likelihood Estimation.

    MNLE combines a Categorical net and a neural spline flow to model data with
    mixed types, e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        discrete_net: CategoricalMassEstimator,
        continuous_net: NFlowsFlow,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        log_transform_input: bool = False,
    ):
        """Initialize class for combining density estimators for MNLE.

        Args:
            discrete_net: neural net to model discrete part of the data.
            continuous_net: neural net to model the continuous data.
            input_shape: Event shape of the input at which the density is being
                evaluated (and which is also the event_shape of samples).
            condition_shape: Shape of the condition. If not provided, it will assume a
                1D input.
            log_transform_input: whether to transform the continous part of the data
                into logarithmic domain before training. This is helpful for bounded
                data, e.g.,for reaction times.
        """
        super(MixedDensityEstimator, self).__init__(
            net=continuous_net, input_shape=input_shape, condition_shape=condition_shape
        )

        self.discrete_net = discrete_net
        self.continuous_net = continuous_net
        self.log_transform_input = log_transform_input

    def forward(self, input: Tensor):
        raise NotImplementedError(
            """The forward method is not implemented for MNLE, use '.sample(...)' to
            generate samples though a forward pass."""
        )

    def sample(
        self, sample_shape: torch.Size, condition: Tensor, track_gradients: bool = False
    ) -> Tensor:
        """Return sample from mixed data distribution.

        Args:
            sample_shape: Shape of samples to generate.
            condition: Condition of shape `(batch_dim, *event_shape_condition)`

        Returns:
            Samples of shape `(*sample_shape, batch_dim, event_dim_input)`
        """
        num_samples = torch.Size(sample_shape).numel()
        batch_dim = condition.shape[0]
        condition_event_dim = condition.dim() - 1

        with torch.set_grad_enabled(track_gradients):
            # Sample discrete data given parameters.
            discrete_input = self.discrete_net.sample(
                sample_shape=sample_shape,
                condition=condition,
            )
            # Trailing `1` because `Categorical` has event_shape `()`.
            discrete_input = discrete_input.reshape(num_samples * batch_dim, 1)

            ones_for_event_dims = (1,) * condition_event_dim
            repeated_condition = condition.repeat(num_samples, *ones_for_event_dims)

            # Sample continuous data condition on parameters and discrete data.
            # Pass num_samples=1 because the choices in the condition contains
            # num_samples elements already.
            continuous_input = self.continuous_net.sample(
                sample_shape=(),
                # repeat the single condition to match number of sampled choices.
                # sample_shape[0] is the sample dimension.
                condition=torch.cat((repeated_condition, discrete_input), dim=1),
            )

            # In case input was log-transformed, move them to linear space.
            if self.log_transform_input:
                continuous_input = continuous_input.exp()

            joined_input = torch.cat((continuous_input, discrete_input), dim=1)

            # `continuous_input` is of shape `(batch_dim * numel(sample_shape))`.
            return joined_input.reshape(*sample_shape, batch_dim, -1)

    def log_prob(self, input: Tensor, condition: Tensor) -> Tensor:
        """Return log-probability of samples under the learned MNLE.

        For a fixed data point input this returns the value of the likelihood function
        evaluated at condition, L(condition, input=input).

        Alternatively, it can be interpreted as the log-prob of the density
        p(input | condition).

        It evaluates the separate density estimator for the discrete and continous part
        of the data and then combines them into one evaluation.

        Args:
            input: data (containing continuous and discrete data).
            condition: parameters for which to evaluate the likelihod function, or for
                which to condition p(input | condition).

        Returns:
            Tensor: log_prob of p(input | condition).
        """
        cont_input, disc_input = _separate_input(input)

        disc_log_prob = self.discrete_net.log_prob(
            input=disc_input, condition=condition
        )

        # Pass parameters and discrete input as condition.
        repeats = disc_input.shape[0]
        disc_input_repeated = disc_input.reshape((repeats * disc_input.shape[1], -1))
        condition_repeated = condition.repeat((repeats, 1))
        condition_reshaped = torch.cat((condition_repeated, disc_input_repeated), dim=1)

        cont_input_reshaped = cont_input.reshape((
            1,
            cont_input.shape[0] * cont_input.shape[1],
            -1,
        ))
        cont_log_prob = self.continuous_net.log_prob(
            # Transform to log-space if needed.
            (
                torch.log(cont_input_reshaped)
                if self.log_transform_input
                else cont_input_reshaped
            ),
            condition=condition_reshaped,
        )
        cont_log_prob = cont_log_prob.reshape(disc_log_prob.shape)

        # Combine into joint lp.
        log_probs_combined = disc_log_prob + cont_log_prob

        # Maybe add log abs det jacobian of RTs: log(1/x) = - log(x)
        if self.log_transform_input:
            log_probs_combined -= torch.log(cont_input).sum(-1)

        return log_probs_combined

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the loss for training the density estimator.

        Args:
            input: Inputs of shape `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)`.

        Returns:
            Loss of shape `(batch_dim,)`
        """
        return -self.log_prob(input.unsqueeze(0), condition)[0]

    def log_prob_iid(self, input: Tensor, condition: Tensor) -> Tensor:
        """Return logprob given a batch of iid input and a different batch of condition.

        This is different from `.log_prob()` to enable speed ups in evaluation during
        inference. The speed up is achieved by exploiting the fact that there are only
        finite number of possible categories in the discrete part of the data: one can
        just calculate the log probs for each possible category (given the current batch
        of context) and then copy those log probs into the entire batch of iid
        categories.
        For example, for the drift-diffusion model, there are only two choices, but
        often 100s or 1000 trials. With this method a evaluation over trials then passes
        a batch of `2 (one per choice) * num_conditions` into the NN, whereas the normal
        `.log_prob()` would pass `1000 * num_conditions`.

        Args:
            input: batch of iid data, data observed given the same underlying parameters
                or experimental conditions.
            condition: batch of parameters to be evaluated, i.e., each batch entry will
                be evaluated for the entire batch of iid input.

        Returns:
            log probs with shape (num_trials, num_parameters), i.e., the log prob for
            each context for each trial.
        """

        condition = atleast_2d(condition)
        input = atleast_2d(input)
        batch_size = condition.shape[0]
        num_trials = input.shape[0]
        condition_repeated, input_repeated = match_theta_and_x_batch_shapes(
            condition, input
        )
        net_device = next(self.discrete_net.parameters()).device
        assert net_device == input.device and input.device == condition.device, (
            f"device mismatch: net, x, condition: "
            f"{net_device}, {input.device}, {condition.device}."
        )

        input_cont_repeated, input_disc_repeated = _separate_input(input_repeated)
        input_cont, input_disc = _separate_input(input)

        # repeat categories for parameters
        repeated_categories = torch.repeat_interleave(
            torch.arange(self.discrete_net.num_categories - 1), batch_size, dim=0
        )
        # repeat parameters for categories
        repeated_condition = condition.repeat(self.discrete_net.num_categories - 1, 1)
        log_prob_per_cat = torch.zeros(self.discrete_net.num_categories, batch_size).to(
            net_device
        )
        log_prob_per_cat[:-1, :] = self.discrete_net.log_prob(
            repeated_categories.to(net_device),
            repeated_condition.to(net_device),
        ).reshape(-1, batch_size)
        # infer the last category logprob from sum to one.
        log_prob_per_cat[-1, :] = torch.log(1 - log_prob_per_cat[:-1, :].exp().sum(0))

        # fill in lps for each occurred category
        log_probs_discrete = log_prob_per_cat[
            input_disc.type_as(torch.zeros(1, dtype=torch.long)).squeeze()
        ].reshape(-1)

        # Get repeat discrete data and condition to match in batch shape for flow eval.
        log_probs_cont = self.continuous_net.log_prob(
            (
                torch.log(input_cont_repeated)
                if self.log_transform_input
                else input_cont_repeated
            ),
            condition=torch.cat((condition_repeated, input_disc_repeated), dim=1),
        )

        # Combine into joint lp with first dim over trials.
        log_probs_combined = (log_probs_discrete + log_probs_cont).reshape(
            num_trials, batch_size
        )

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.log_transform_input:
            log_probs_combined -= torch.log(input_cont)

        # Return batch over trials as required by SBI potentials.
        return log_probs_combined


def _separate_input(
    input: Tensor, num_discrete_columns: int = 1
) -> Tuple[Tensor, Tensor]:
    """Returns the continuous and discrete part of the given input.

    Assumes the discrete data to live in the last columns of input.
    """
    return input[..., :-num_discrete_columns], input[..., -num_discrete_columns:]
