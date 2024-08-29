# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm

from sbi.inference.potentials.score_based_potential import (
    PosteriorScoreBasedPotential,
)
from sbi.samplers.score import Corrector, Predictor, get_corrector, get_predictor


class Diffuser:
    predictor: Predictor
    corrector: Optional[Corrector]

    def __init__(
        self,
        score_based_potential: PosteriorScoreBasedPotential,
        predictor: Union[str, Predictor],
        corrector: Optional[Union[str, Corrector]] = None,
        predictor_params: Optional[dict] = None,
        corrector_params: Optional[dict] = None,
    ):
        """Diffusion-based sampler for score-based sampling i.e it requires the
        gradient of a family of distributions (for different times) characterized by the
        gradient of a potential function (i.e. the score function). The sampler uses a
        predictor to propagate samples forward in time. Optionally, a corrector can be
        used to refine the samples at the current time.

        Args:
            score_based_potential_gradient: A time-dependent score-based potential.
            predictor: A predictor to propagate samples forward in time.
            corrector (Ooptional): A corrector to refine the samples. Defaults to None.
            predictor_params (optional): Parameters passed to the predictor, if given as
                string. Defaults to None.
            corrector_params (optional): Parameters passed to the corrector, if given as
                string. Defaults to None.
        """
        # Set predictor and corrector
        self.set_predictor(predictor, score_based_potential, **(predictor_params or {}))
        self.set_corrector(corrector, **(corrector_params or {}))
        self.device = self.predictor.device

        # Extract time limits from the score function
        self.t_min = score_based_potential.score_estimator.t_min
        self.t_max = score_based_potential.score_estimator.t_max

        # Extract initial moments
        self.init_mean = score_based_potential.score_estimator.mean_t
        self.init_std = score_based_potential.score_estimator.std_t

        # Extract relevant shapes from the score function
        self.input_shape = score_based_potential.score_estimator.input_shape
        self.condition_shape = score_based_potential.score_estimator.condition_shape
        condition_dim = len(self.condition_shape)
        # TODO: this is the iid setting and we don't want to generate num_obs samples,
        # but only one sample given the condition.
        self.batch_shape = score_based_potential.x_o.shape[:-condition_dim]

    def set_predictor(
        self,
        predictor: Union[str, Predictor],
        score_based_potential: PosteriorScoreBasedPotential,
        **kwargs,
    ):
        """Set the predictor for the diffusion-based sampler."""
        if isinstance(predictor, str):
            self.predictor = get_predictor(predictor, score_based_potential, **kwargs)
        else:
            self.predictor = predictor

    def set_corrector(self, corrector: Optional[Union[str, Corrector]], **kwargs):
        """Set the corrector for the diffusion-based sampler."""
        if corrector is None:
            self.corrector = None
        elif isinstance(corrector, Corrector):
            self.corrector = corrector
        else:
            self.corrector = get_corrector(corrector, self.predictor, **kwargs)

    def initialize(self, num_samples: int) -> Tensor:
        """Initialize the sampler by drawing samples from the initial distribution.

        If we have to sample from a batch of distributions, we draw samples from each
        distribution in the batch i.e. of shape (num_batch, num_samples, input_shape).

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            Tensor: _description_
        """
        # TODO: for iid setting, self.batch_shape.numel() will be the iid-batch. But we
        # don't want to generate num_obs samples, but only one sample given the the iid
        # batch.
        # TODO: the solution will probably be to distinguish between the iid setting and
        # batched sampling setting with a flag.
        # TODO: this fixes the iid setting shape problems, but iid inference via
        # iid_bridge is not accurate.
        # num_batch = self.batch_shape.numel()
        # init_shape = (num_batch, num_samples) + self.input_shape
        init_shape = (
            num_samples,
        ) + self.input_shape  # just use num_samples, not num_batch
        # NOTE: for the IID setting we might need to scale the noise with iid batch
        # size, as in equation (7) in the paper.
        eps = torch.randn(init_shape, device=self.device)
        mean, std, eps = torch.broadcast_tensors(self.init_mean, self.init_std, eps)
        return mean + std * eps

    @torch.no_grad()
    def run(
        self,
        num_samples: int,
        ts: Tensor,
        show_progress_bars: bool = True,
        save_intermediate: bool = False,
    ) -> Tensor:
        """Samples from the distribution at the final time point by propagating samples
        forward in time using the predictor and optionally refining them using the a
        corrector.

        Args:
            num_samples: Number of samples to draw.
            ts: Time grid to propagate samples forward, or "solve" the SDE.
            show_progress_bars (optional): Shows a progressbar or not. Defaults to True.
            save_intermediate (optional): Returns samples at all time point, instead of
                only returning samples at the end. Defaults to False.

        Returns:
            Tensor: Samples from the distribution(s).
        """
        samples = self.initialize(num_samples)
        pbar = tqdm(
            range(1, ts.numel()),
            disable=not show_progress_bars,
            desc=f"Drawing {num_samples} posterior samples",
        )

        if save_intermediate:
            intermediate_samples = [samples]

        for i in pbar:
            t1 = ts[i - 1]
            t0 = ts[i]
            samples = self.predictor(samples, t1, t0)
            if self.corrector is not None:
                samples = self.corrector(samples, t0, t1)
            if save_intermediate:
                intermediate_samples.append(samples)

        if save_intermediate:
            return torch.cat(intermediate_samples, dim=0)
        else:
            return samples
