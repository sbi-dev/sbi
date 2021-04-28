# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from copy import deepcopy

from typing import Any, Callable, Dict, Optional, Union

from sbi.inference.posteriors.variational_posterior import VariationalPosterior
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.snle.snle_base import LikelihoodEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.types import TensorboardSummaryWriter
from sbi.types import TorchModule
from sbi.utils import (
    del_entries,
    check_estimator_arg,
    test_posterior_net_for_multi_d_x,
    validate_theta_and_x,
    x_shape_from_simulation,
)

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
import torch.optim as optim

from sbi.vi.divergence_optimizers import ElboOptimizer
from sbi.vi.pyro_flows import build_q


class SNLE_C(LikelihoodEstimator):
    def __init__(
        self,
        prior,
        simulator,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args,
    ):
        r"""Sequential Neural Likelihood [1].

        [1] Sequential Neural Likelihood: Fast Likelihood-free Inference with
        Autoregressive Flows_, Papamakarios et al., AISTATS 2019,
        https://arxiv.org/abs/1805.07226

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: torch device on which to compute, e.g. gpu, cpu.
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
            unused_args: Absorbs additional arguments. No entries will be used. If it
                is not empty, we warn. In future versions, when the new interface of
                0.14.0 is more mature, we will remove this argument.
        """
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "unused_args", "simulator")
        )
        super().__init__(**kwargs, **unused_args)
        self._simulator = simulator
        self._posterior = None

    def train(
        self,
        x_obs,
        learning_rate: float = 1e-3,
        max_num_simulation: int = 2000,
        simulations_per_round: int = 10,
        simulation_num_worker: int = 1,
        epochs_per_round: Optional[int] = 10,
        elbo_steps_per_round: Optional[int] = 20,
        prior_proposal_probability_decay_rate: Optional[float] = 0.9,
        stop_after_epochs: Optional[int] = 500,
        posterior_kwargs: Optional[Dict] = {},
        likelihood_train_kwargs: Optional[Dict] = {},
        posterior_train_kwargs: Optional[Dict] = {},
    ) -> NeuralPosterior:

        num_simulations = 0
        prior_proposal_probability = 0.9
        converged = False
        rounds = 0

        while rounds <= stop_after_epochs:
            print("Additional Simulation: ", num_simulations)
            # Train likelihood for a small number of iterations
            density_estimator = self.train_likelihood(
                learning_rate=learning_rate,
                max_num_epochs=epochs_per_round,
                **likelihood_train_kwargs,
            )

            # Build the variational posterior which is used during training of the
            # likelihood network
            if self._posterior is None:
                self._posterior = VariationalPosterior(
                    method_family="snle",
                    neural_net=density_estimator,
                    prior=self._prior,
                    x_shape=self._x_shape,
                    device=self._device,
                    flow_paras=posterior_kwargs,
                )
                self._posterior.set_default_x(x_obs)
            else:
                self._posterior.net = density_estimator

            # Train the posterior for some steps
            self._posterior.train(
                x_obs,
                learning_rate=learning_rate,
                min_num_iters=elbo_steps_per_round,
                max_num_iters=elbo_steps_per_round,
            )
            if num_simulations <= max_num_simulation:
                # Simulate new samples
                if torch.rand(1) < prior_proposal_probability:
                    proposal = self._prior
                else:
                    proposal = self._posterior
                # Redure prior proposal probability
                prior_proposal_probability *= prior_proposal_probability_decay_rate
                # Simulate new data
                theta, x = simulate_for_sbi(
                    simulator=self._simulator,
                    proposal=proposal,
                    num_simulations=simulations_per_round,
                    num_workers=simulation_num_worker,
                )
                _ = self.append_simulations(theta, x)
                num_simulations += simulations_per_round
            rounds += 1

    def train_likelihood(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def build_posterior(
        self, density_estimator: Optional[TorchModule] = None, **kwargs
    ) -> VariationalPosterior:

        if density_estimator is None:
            density_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device

        # Train posterior
        if self._posterior.default_x is not None:
            self._posterior.train(learning_rate=1e-3, n_particles=1024)

        return self._posterior

