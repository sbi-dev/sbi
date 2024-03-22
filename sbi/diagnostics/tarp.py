from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, yhat: Tensor, y: Tensor):
        return torch.sqrt(self.mse(yhat, y))


class TARP:
    """Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

    This class implements the distance to random point as a diagnostic method for posterior estimators.

    """

    def __init__(
        self,
        references: Tensor = None,
        metric: str = "euclidean",
        num_alpha_bins: Union[int, None] = None,
        num_bootstrap: int = 100,
        norm: bool = False,
        bootstrap: bool = False,
        seed: Union[int, None] = None,
    ):
        """
        Args:
          references: the reference points to use for the DRP regions, with shape ``(n_references, n_sims)``, or ``None``. If the latter, then the reference points are chosen randomly from the unit hypercube over the parameter space.
          metric: the metric to use when computing the distance. Can be ``"euclidean"`` or ``"manhattan"``.
          norm : whether to normalize parameters before coverage test (Default = True)
          num_alpha_bins: number of bins to use for the credibility values. If ``None``, then ``n_sims // 10`` bins are used.
          seed: the seed to use for the random number generator. If ``None``, then no seed
        """
        self.references = references
        self.metric_name = metric
        self.n_bins = num_alpha_bins
        self.n_bootstrap = num_bootstrap
        self.do_norm = norm
        self.do_bootstrap = bootstrap
        self.seed = seed

    def run(
        self,
        samples: Tensor,
        theta: Tensor,
        # posterior: NeuralPosterior,  # TODO: not sure yet, if this is required
    ):
        """
        Estimates coverage with the TARP method a single time.

        Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

        Args:
            samples: the parameter samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``. Multiple samples for observation are encouraged.
            theta: the true parameter value theta, with shape ``(n_sims, n_dims)``.

        Returns:
            Expected coverage probability (``ecp``) and credibility values (``alpha``)

        """
        # DRP assumes that the predicted thetas are sampled from the posterior num_samples times
        theta = theta.detach()
        samples = samples.detach()

        num_samples = samples.shape[0]
        num_sims = samples.shape[1]
        num_dims = samples.shape[2]

        if self.n_bins is None:
            self.n_bins = num_sims // 10

        if theta.shape[0] != num_sims:
            raise ValueError("theta must have the same number of rows as samples")
        if theta.shape[1] != num_dims:
            raise ValueError("theta must have the same number of columns as samples")
        theta = theta.unsqueeze(0)  # add new axis in front

        if self.do_norm:
            lo = torch.min(theta, dim=1, keepdims=True)  # min along num_sims
            hi = torch.max(theta, dim=1, keepdims=True)  # max along num_sims
            samples = (samples - lo) / (hi - lo + 1e-10)
            theta = (theta - lo) / (hi - lo + 1e-10)

        assert len(theta.shape) == len(samples.shape)

        if not isinstance(self.references, Tensor):
            refpdf = torch.distributions.Uniform(low=0, high=1)
            self.references = refpdf.sample((1, num_sims, num_dims))
        else:
            if len(self.references.shape) == 2:
                self.references = self.references.unsqueeze(0)

            if len(self.references.shape) == 3 and self.references.shape[0] != 1:
                raise ValueError(
                    f"references must be a 2D array with a singular first dimension, received {self.references.shape}"
                )

            if self.references.shape[-2] != num_sims:
                raise ValueError(
                    f"references must have the same number samples as samples, received {self.references.shape[-2]} != {num_sims}"
                )

            if self.references.shape[-1] != num_dims:
                raise ValueError(
                    f"references must have the same number of dimensions as samples or theta, received {self.references.shape[-1]} != {num_dims}"
                )

        assert len(self.references.shape) == len(samples.shape)

        if self.metric_name.lower() in ["l2", "euclidean"]:
            distance = RMSELoss(reduction="sum")
        elif self.metric_name.lower() in ["l1", "manhattan"]:
            distance = torch.nn.L1Loss(reduction="sum")
        else:
            raise ValueError(
                f"metric must be either 'euclidean' or 'manhattan', received {metric}"
            )

        sample_dists = distance(self.references.expand(num_samples, -1, -1), samples)
        theta_dists = distance(self.references, theta)

        # compute coverage
        coverage_values = torch.sum(sample_dists < theta_dists, axis=0) / num_samples
        hist, bin_edges = torch.histogram(
            coverage_values, density=True, bins=self.n_bins
        )
        stepsize = bin_edges[1] - bin_edges[0]
        ecp = torch.cumsum(hist, dim=0) * stepsize

        return torch.cat([Tensor([0]), ecp]), bin_edges
