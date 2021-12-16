# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Dict, Optional, Union

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.snle.snle_base import LikelihoodEstimator
from sbi.types import TensorboardSummaryWriter
from sbi.utils import del_entries
from torch import autograd, Tensor
import torch

from sbi.utils import validate_theta_and_x
from sbi.utils.sbiutils import mask_sims_from_prior


class SCANDAL(LikelihoodEstimator):
    def __init__(
        self,
        prior,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
        **unused_args,
    ):
        r"""SCANDAL [1].
        [1] Mining gold from implicit models to improve likelihood-free inference,
        Brehmer et al., PNAS 2020, https://www.pnas.org/content/117/10/5242.short
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

        kwargs = del_entries(locals(), entries=("self", "__class__", "unused_args"))
        super().__init__(**kwargs, **unused_args)

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        from_round: int = 0,
        score: Optional[Tensor] = None,
    ) -> "LikelihoodEstimator":
        r"""
        Store parameters and simulation outputs to use them for later training.
        Data are stored as entries in lists for each type of variable (parameter/data).
        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.
        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            score: Joint score $\Nabla_{\theta}(p(x,z|\theta))$. If passed, the joint
                score will be used during training to regularize the likelihood
                estimate (see Brehmer, Louppe, Cranmer 2020 PNAS).
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        validate_theta_and_x(theta, x)

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(mask_sims_from_prior(int(from_round), theta.size(0)))
        self._data_round_index.append(int(from_round))
        self._score_roundwise.append(score)

        return self

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        exclude_invalid_x: bool = True,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch_each_round: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        score_lambda: float = 1e-6,
    ) -> NeuralPosterior:
        r"""
        Return density estimator that approximates the distribution $p(x|\theta)$.
        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch_each_round: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).
            score_lambda: Weighing factor of the mean-squared-error loss imposed on the
                scores.
        Returns:
            Density estimator that approximates the distribution $p(x|\theta)$.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        return super().train(**kwargs)

    def _loss(self, theta: Tensor, x: Tensor, score: Tensor, score_lambda: float):
        r"""
        Returns the loss (mixture between MLE and score-matching).
        Args:
            theta:
            x:
            score:
            score_lambda:
        Returns:
            Loss.
        """

        # Evaluate on x with theta as context.
        self._neural_net.train()
        log_prob = self._neural_net.log_prob(x, context=theta)
        loss = -log_prob

        # Add the score-loss to the neural density estimator.
        if torch.any(score.bool()):
            loss += score_lambda * self._score_loss(theta, x, score)

        return loss

    def _score_loss(self, theta, x, score):
        r"""
        Returns the mean-squared error loss between the true and estimated score.
        Args:
            theta:
            x:
            score:
        Returns:
            Mean-squared error loss for the score.
        """
        self._neural_net.eval()
        with torch.enable_grad():
            theta = theta.clone().requires_grad_(True)
            log_prob = self._neural_net.log_prob(x, context=theta)

            estimated_score = autograd.grad(
                outputs=log_prob,
                inputs=theta,
                grad_outputs=torch.ones_like(log_prob),
                create_graph=True,
                only_inputs=True,
            )[0]

        # Compute MSE-loss of true score and estimated score.
        score_is_available = score.bool()

        estimated_score = (estimated_score - self.score_means) / self.score_stds
        score = (score - self.score_means) / self.score_stds

        score_loss = (
            estimated_score[score_is_available] - score[score_is_available].float()
        ) ** 2
        score_loss = torch.reshape(score_loss, score.shape).sum(dim=1)
        return score_loss
