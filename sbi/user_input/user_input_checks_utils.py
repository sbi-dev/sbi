# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


import warnings
from typing import Callable, Optional, Union, Dict, Any, Tuple, Union, cast, List, Sequence, TypeVar

import torch
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen
from torch import Tensor, float32
from torch.distributions import Distribution


class CustomPytorchWrapper(Distribution):
    """Wrap custom prior object to PyTorch distribution object.

    Note that the prior must have .sample and .log_prob methods.
    """

    def __init__(
        self,
        custom_prior,
        return_type: Optional[torch.dtype] = float32,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self.custom_prior = custom_prior
        self.return_type = return_type

        self._set_mean_and_variance()

    def log_prob(self, value) -> Tensor:
        return torch.as_tensor(
            self.custom_prior.log_prob(value), dtype=self.return_type
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return torch.as_tensor(
            self.custom_prior.sample(sample_shape), dtype=self.return_type
        )

    def _set_mean_and_variance(self):
        """Set mean and variance if available, else estimate from samples."""

        if hasattr(self.custom_prior, "mean"):
            pass
        else:
            self.custom_prior.mean = torch.mean(
                torch.as_tensor(self.custom_prior.sample((1000,))), dim=0
            )
            warnings.warn(
                "Prior is lacking mean attribute, estimating prior mean from samples...",
                UserWarning,
            )
        if hasattr(self.custom_prior, "variance"):
            pass
        else:
            self.custom_prior.variance = (
                torch.std(torch.as_tensor(self.custom_prior.sample((1000,))), dim=0)
                ** 2
            )
            warnings.warn(
                "Prior is lacking variance attribute, estimating prior variance from samples...",
                UserWarning,
            )

    @property
    def mean(self):
        return torch.as_tensor(self.custom_prior.mean, dtype=self.return_type)

    @property
    def variance(self):
        return torch.as_tensor(self.custom_prior.variance, dtype=self.return_type)


class ScipyPytorchWrapper(Distribution):
    """Wrap scipy.stats prior as a PyTorch Distribution object."""

    def __init__(
        self,
        prior_scipy: Union[rv_frozen, multi_rv_frozen],
        return_type: Optional[torch.dtype] = float32,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self.prior_scipy = prior_scipy
        self.return_type = return_type

    def log_prob(self, value) -> Tensor:
        return torch.as_tensor(self.prior_scipy.logpdf(x=value), dtype=self.return_type)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return torch.as_tensor(
            self.prior_scipy.rvs(size=sample_shape), dtype=self.return_type
        )

    @property
    def mean(self):
        return self.prior_scipy.mean()

    @property
    def variance(self):
        return self.prior_scipy.var()


class PytorchReturnTypeWrapper(Distribution):
    """Wrap PyTorch Distribution to return a given return type."""

    def __init__(
        self,
        prior: Distribution,
        return_type: Optional[torch.dtype] = float32,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self.prior = prior
        self.return_type = return_type

    def log_prob(self, value) -> Tensor:
        return torch.as_tensor(self.prior.log_prob(value), dtype=self.return_type)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return torch.as_tensor(self.prior.sample(sample_shape), dtype=self.return_type)

    @property
    def mean(self):
        return torch.as_tensor(self.prior.mean, dtype=self.return_type)

    @property
    def variance(self):
        return torch.as_tensor(self.prior.variance, dtype=self.return_type)


class MultipleIndependent(Distribution):
    """Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.

    Every element of the sequence is treated as independent from the other elements.
    Single elements can be multivariate with dependent dimensions, e.g.,:
        - [
            Gamma(torch.zeros(1), torch.ones(1)),
            Beta(torch.zeros(1), torch.ones(1)),
            MVG(torch.ones(2), torch.tensor([[1, .1], [.1, 1.]]))
        ]
        - [
            Uniform(torch.zeros(1), torch.ones(1)),
            Uniform(torch.ones(1), 2.0 * torch.ones(1))]    
    """

    def __init__(
        self, dists: Sequence[Distribution], validate_args=None,
    ):
        self._check_distributions(dists)

        self.dists = dists
        # numel() instead of event_shape because for all dists both is possible,
        # event_shape=[1] or batch_shape=[1]
        self.dims_per_dist = torch.as_tensor([d.sample().numel() for d in self.dists])
        self.ndims = torch.sum(torch.as_tensor(self.dims_per_dist)).item()

        super().__init__(
            batch_shape=torch.Size([]),  # batch size was ensured to be <= 1 above.
            event_shape=torch.Size(
                [self.ndims]
            ),  # Event shape is the sum of all ndims.
            validate_args=validate_args,
        )

    def _check_distributions(self, dists):
        """Check if dists is Sequence and longer 1 and check every member."""
        assert isinstance(
            dists, Sequence
        ), f"""The combination of independent priors must be of type Sequence, is 
               {type(dists)}."""
        assert len(dists) > 1, "Provide at least 2 distributions to combine."
        # Check every element of the sequence.
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        """Check type and shape of a single input distribution."""

        assert not isinstance(
            dist, MultipleIndependent
        ), "Nesting of combined distributions is not possible."
        assert isinstance(
            dist, Distribution
        ), "Distribution must be a PyTorch distribution."
        # Make sure batch shape is smaller or equal to 1.
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution must be smaller or equal to 1."

        assert (
            len(dist.batch_shape) > 0 or len(dist.event_shape) > 0
        ), """One of the distributions you passed is defined over a scalar only. Make
        sure pass distributions with one of event_shape or batch_shape > 0: For example
            - instead of Uniform(0.0, 1.0) pass Uniform(torch.zeros(1), torch.ones(1))
            - instead of Beta(1.0, 2.0) pass Beta(tensor([1.0]), tensor([2.0])).
        """

    def sample(self, sample_shape=torch.Size()) -> Tensor:

        # Sample from every sub distribution and concatenate samples.
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # This reshape is needed to cover the case .sample() vs. .sample((n, )).
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:

        value = self._prepare_value(value)

        # Evaluate value per distribution, taking into account that individual
        # distributions can be multivariate.
        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0
        for idx, d in enumerate(self.dists):
            ndims = self.dims_per_dist[idx].item()
            v = value[:, dims_covered : dims_covered + ndims]
            # Reshape here to ensure all returned log_probs are 2D for concatenation.
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum accross last dimension to get joint log prob over all distributions.
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        """Return input value with fixed shape.

        Raises: 
            AssertionError: if value has more than 2 dimensions or invalid size in
                2nd dimension.
        """

        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert (
            value.ndim == 2
        ), f"value in log_prob must have ndim <= 2, it is {value.ndim}."

        batch_shape, num_value_dims = value.shape

        assert (
            num_value_dims == self.ndims
        ), f"Number of dimensions must match dimensions of this joint: {self.ndims}."

        return value

    @property
    def mean(self) -> Tensor:
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        return torch.cat([d.variance for d in self.dists])
