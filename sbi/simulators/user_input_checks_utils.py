from __future__ import annotations

import warnings
from typing import Optional, Union, List

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


class IndependentJoint(Distribution):
    def __init__(
        self, dists: List[Distribution], validate_args=None,
    ):
        # reject all batch_shape>1, assert Distribution.
        self._check_distributions(dists)

        self.dists = dists
        # numel() instead of event_shape because all scalar dists usually have
        # event_shape=[] and batch_shape=[1]
        self.dims_per_dist = torch.as_tensor([d.sample().numel() for d in self.dists])
        self.ndims = torch.sum(torch.as_tensor(self.dims_per_dist)).item()

        super().__init__(
            batch_shape=torch.Size([]),  # batch size was ensured to be <= 1 above.
            event_shape=torch.Size(
                [self.ndims]
            ),  # Event shape is the sum of all ndims.
            validate_args=validate_args,
        )

    def _check_distributions(self, dists: List[Distribution]):
        """Check validity of input distributions."""
        assert isinstance(dists, List)
        assert len(dists) > 1
        for d in dists:
            assert not isinstance(d, IndependentJoint)  # Nesting is not allowed.
            assert isinstance(d, Distribution)
            # Make sure batch shape is smaller or equal to 1.
            assert d.batch_shape in (torch.Size([1]), torch.Size([0]), torch.Size([]))

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

    def _prepare_value(self, value):

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
