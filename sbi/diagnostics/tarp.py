import warnings
from typing import Tuple, Union

# import numpy as np
import torch
from joblib import Parallel, delayed
from torch import Tensor
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.simulators.simutils import tqdm_joblib


def l2(x: Tensor, y: Tensor, axis=-1) -> Tensor:
    """
    Calculates the L2 distance between two tensors.
    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.
        axis (int, optional): The axis along which to calculate the L2 distance.
                Defaults to -1.
    Returns:
        Tensor: A tensor containing the L2 distance between x and y along the
                specified axis.
    """
    return torch.sqrt(torch.sum((x - y) ** 2, axis=axis))


def l1(x: Tensor, y: Tensor, axis=-1) -> Tensor:
    """
    Calculates the L1 distance between two tensors.
    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.
        axis (int, optional): The axis along which to calculate the L1 distance.
                Defaults to -1.
    Returns:
        Tensor: A tensor containing the L1 distance between x and y along the
                specified axis.
    """
    return torch.sum(torch.abs(x - y), axis=axis)


def infer_posterior_on_batch(
    xs: Tensor,
    posterior: NeuralPosterior,
    num_posterior_samples: int = 1000,
) -> Tensor:
    """
    Infer samples of a posterior distribution on a batch of inputs.

    Parameters:
    ----------
    xs : Tensor
        The input data batch.
    posterior : NeuralPosterior
        The neural posterior to use for inference.
    num_posterior_samples : int, optional
        The number of posterior samples to draw for each input, by default 1000.

    Returns:
    -------
    Tensor
        A tensor of shape (num_posterior_samples, N, P) where N is the number of
        samples given by xs and P is the output dimension of the neural posterior.
    """

    samples = []

    for idx in range(xs.shape[0]):
        # unsqueeze for potential higher-dimensional data.
        xo = xs[idx].unsqueeze(0)
        # VI posterior needs to be trained on the current xo.
        if isinstance(posterior, VIPosterior):
            posterior.set_default_x(xo)
            posterior.train()

        # Draw posterior samples and save one for the data average posterior.
        ths = posterior.sample((num_posterior_samples,), x=xo, show_progress_bars=False)
        # Note: one could calculate coverage values here

        samples.append(ths.unsqueeze(1))

    return torch.cat(samples, dim=1)


class TARP:
    """
    Implementation taken from Lemos et al, 'Sampling-Based Accuracy Testing of
    Posterior Estimators for General Inference' https://arxiv.org/abs/2302.03026

    This class implements the distance to random point as a diagnostic method
    for samples of posterior estimators.

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
        """TARP diagnostics of posterior samples
        Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

        Args:
          references: the reference points to use for the DRP regions, with
                shape ``(n_references, n_sims)``, or ``None``. If the latter,
                then the reference points are chosen randomly from the unit
                hypercube over the parameter space.
          metric: the metric to use when computing the distance. Can be
                ``"euclidean"`` or ``"manhattan"``.
          norm : whether to normalize parameters before coverage test
                (Default = True)
          num_alpha_bins: number of bins to use for the credibility values.
                If ``None``, then ``n_sims // 10`` bins are used.
          bootstrap: perform bootstrapped TARP analysis (not implemented yet)
          seed: the seed to use for the random number generator. If ``None``,
                then no seed
        """
        self.references = references
        self.metric_name = metric
        self.n_bins = num_alpha_bins
        if self.n_bins and self.n_bins < 10:
            warnings.warn(
                f"""Number of bins to assess TARP coverage should be between
                20 to 50. {self.n_bins} is low. TARP will work, but the
                statistical assessment might fluctuate.""",
                stacklevel=2,
            )

        self.n_bootstrap = num_bootstrap
        self.do_norm = norm
        self.do_bootstrap = bootstrap
        if bootstrap:
            raise NotImplementedError(
                "The bootstrapped version of TARP is not implemented yet in SBI."
            )
        self.seed = seed

    # this function currently does not perform any TARP related operation
    # the purpose of the function is (a) to align with the sbc interface and
    # (b) to provide the data which is required to run TARP
    def run(
        self,
        xs: Tensor,
        posterior: NeuralPosterior,
        num_posterior_samples: int = 1000,
        num_workers: int = 1,
        infer_batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> Tensor:
        """perform inference on batched x values using the provided posterior

        Args:
            xs: observed data for tarp, simulated from thetas.
            posterior: a posterior obtained from sbi.
            num_posterior_samples: number of approximate posterior samples used
                for ranking.
            num_workers: number of CPU cores to use in parallel for running
                infer_batch_size inferences.
            infer_batch_size: batch size for workers.
            show_progress_bar: whether to display a progress bar

        Returns:
            samples: posterior samples obtained by performing inference on xs
                given the posterior

        """
        num_sim_samples = xs.shape[0]
        xs_batches = torch.split(xs, infer_batch_size, dim=0)

        if num_workers != 1:
            # Parallelize the sequence of batches across workers.
            # We use the solution proposed here: https://stackoverflow.com/a/61689175
            # to update the pbar only after the workers finished a task.
            with tqdm_joblib(
                tqdm(
                    xs_batches,
                    disable=not show_progress_bar,
                    desc=f"Performing {num_sim_samples} posterior runs in"
                    f"{len(xs_batches)} batches.",
                    total=len(xs_batches),
                )
            ) as _:
                samples: Tensor
                samples = Parallel(n_jobs=num_workers)(  # pyright: ignore[reportAssignmentType]
                    delayed(infer_posterior_on_batch)(
                        xs_batch, posterior, num_posterior_samples
                    )
                    for xs_batch in xs_batches
                )
        else:
            pbar = tqdm(
                total=num_sim_samples,
                disable=not show_progress_bar,
                desc=f"Running {num_sim_samples} samples for tarp analysis.",
            )

            with pbar:
                samples = []
                for xs_batch in xs_batches:
                    samples.append(
                        infer_posterior_on_batch(
                            xs_batch, posterior, num_posterior_samples
                        )
                    )
                    pbar.update(infer_batch_size)
                samples = torch.cat(samples, dim=1)

        return samples

    def check(
        self,
        samples: Tensor,
        theta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Estimates coverage with the TARP method.
        Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

        Args:
            samples: The predicted parameter samples to compute the coverage of,
                     with shape ``(n_samples, n_sims, n_dims)``. These are
                     obtained by sampling a trained posterior.  Multiple
                     (posterior) samples for one observation are encouraged.
            theta: The true parameter value theta, with shape ``(n_sims, n_dims)``.

        Returns:
            ecp: Expected coverage probability (``ecp``)
            alpha: credibility values

        """
        # TARP assumes that the predicted thetas are sampled from the "true"
        # PDF num_samples times
        theta = theta.detach() if len(theta.shape) != 2 else theta.detach().unsqueeze(0)
        samples = samples.detach()

        num_samples = samples.shape[0]  # samples per simulation
        num_sims = samples.shape[-2]
        num_dims = samples.shape[-1]

        if self.n_bins is None:
            self.n_bins = num_sims // 10

        if theta.shape[-2] != num_sims:
            raise ValueError("theta must have the same number of rows as samples")
        if theta.shape[-1] != num_dims:
            raise ValueError("theta must have the same number of columns as samples")

        if self.do_norm:
            lo = torch.min(theta, dim=-2, keepdims=True).values  # min along num_sims
            hi = torch.max(theta, dim=-2, keepdims=True).values  # max along num_sims
            samples = (samples - lo) / (hi - lo + 1e-10)
            theta = (theta - lo) / (hi - lo + 1e-10)

        assert len(theta.shape) == len(samples.shape)

        if not isinstance(self.references, Tensor):
            # obtain min/max per dimension of theta
            lo = (
                torch.min(theta, dim=-2).values.min(axis=0).values
            )  # should be 0 if normalized
            hi = (
                torch.max(theta, dim=-2).values.max(axis=0).values
            )  # should be 1 if normalized

            refpdf = torch.distributions.Uniform(low=lo, high=hi)
            self.references = refpdf.sample((1, num_sims))
        else:
            if len(self.references.shape) == 2:
                # add singleton dimension in front
                self.references = self.references.unsqueeze(0)

            if len(self.references.shape) == 3 and self.references.shape[0] != 1:
                raise ValueError(
                    f"""references must be a 2D array with a singular first
                    dimension, received {self.references.shape}"""
                )

            if self.references.shape[-2] != num_sims:
                raise ValueError(
                    f"references must have the same number samples as samples,"
                    f"received {self.references.shape[-2]} != {num_sims}"
                )

            if self.references.shape[-1] != num_dims:
                raise ValueError(
                    "references must have the same number of dimensions as "
                    f"samples or theta, received {self.references.shape[-1]}"
                    f"!= {num_dims}"
                )

        assert len(self.references.shape) == len(
            samples.shape
        ), f"references {self.references.shape} != samples {samples.shape}"

        if self.metric_name.lower() in ["l2", "euclidean"]:
            distance = l2
        elif self.metric_name.lower() in ["l1", "manhattan"]:
            distance = l1
        else:
            raise ValueError(
                "metric must be either 'euclidean' or 'manhattan',"
                f"received {self.metric_name}"
            )

        sample_dists = distance(self.references.expand(num_samples, -1, -1), samples)
        theta_dists = distance(self.references, theta)

        # compute coverage, f in algorithm 2
        coverage_values = torch.sum(sample_dists < theta_dists, axis=0) / num_samples
        hist, bin_edges = torch.histogram(
            coverage_values, density=True, bins=self.n_bins
        )
        stepsize = bin_edges[1] - bin_edges[0]
        ecp = torch.cumsum(hist, dim=0) * stepsize

        return torch.cat([Tensor([0]), ecp]), bin_edges
