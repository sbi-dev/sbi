import torch
from torch import nn
from torch.distributions import Independent, Categorical, Distribution, biject_to
from torch.distributions.constraints import Constraint, real

import numpy as np
import warnings
from typing import Optional, Iterable

from pyro.nn import AutoRegressiveNN, DenseNN
from pyro.distributions import transforms
from pyro.distributions.transforms import ComposeTransform


from .first_second_order_helpers import jacobian_in_batch
from .flows import (
    AffineTransform,
    LowerCholeskyAffine,
    AffineAutoregressive,
    SplineAutoregressive,
    TransformedDistribution,
    build_flow,
)


class StackedAutoRegressiveNN(nn.Module):
    """This is implements several independent autoregressive networks, which is usefull
        for mixture distributions.

        Args:
            num_nn: Number of independent networks
            input_dim: Input dimension of each network, the total dimension of this
            network will be (num_nn, input_dim)
            hidden_dim: Hidden dimension of the autoregressive neural network
            param_dim: The shape and the number of paramters for each network. For
            Affine Autoregressive flow [1,1] is default as we want to estimate one mean
            and one scale.
    """

    def __init__(
        self,
        num_nn: int,
        input_dim: int,
        hidden_dim: Iterable = None,
        param_dims: Iterable = [1, 1],
        **kwargs,
    ):
        super().__init__()
        self.num_nn = num_nn
        self.input_dim = input_dim
        self.param_dims = param_dims
        self.nets = []
        self._kwargs = kwargs
        self.permutation = []
        if hidden_dim is None:
            self.hidden_dim = [input_dim * 5 + 10]
        else:
            self.hidden_dim = hidden_dim
        for i in range(num_nn):
            net = AutoRegressiveNN(
                input_dim,
                hidden_dims=self.hidden_dim,
                param_dims=param_dims,
                permutation=torch.arange(input_dim),  # This is important for imp. grads
                **kwargs,
            )
            self.add_module("AutoRegressiveNN" + str(i), net)
            self.nets.append(net)
            self.permutation = list(net.permutation)

    def forward(self, x):
        """ Forward pass through each network and stack it """
        x = x.reshape(-1, self.num_nn, self.input_dim)
        out = list(zip(*[net(x[:, i]) for i, net in enumerate(self.nets)]))
        result = [
            torch.hstack(o)
            .reshape(-1, self.num_nn, self.param_dims[i], self.input_dim)
            .squeeze()
            for i, o in enumerate(out)
        ]
        return tuple(result)


class StackedDenseNN(nn.Module):
    """This is implements several independent dense networks, which is usefull
        for mixture distributions of coupling flows.

        Args:
            num_nn: Number of independent networks
            split_dim: Input dimension of each network, which corresponds to the split
            dimension in coupling flows.
            hidden_dim: Hidden dimension of the autoregressive neural network
            param_dim: The shape and the number of paramters for each network. For
            Affine flow [1,1] is default as we want to estimate one mean
            and one scale.
    """

    def __init__(
        self,
        num_nn: int,
        split_dim: int,
        hidden_dim: Iterable = None,
        param_dims: Iterable = [1, 1],
        **kwargs,
    ):
        super().__init__()
        self.num_nn = num_nn
        self.split_dim = split_dim
        self.param_dims = param_dims
        self.nets = []
        self._kwargs = kwargs
        if hidden_dim is None:
            self.hidden_dim = [split_dim * 5 + 10]
        else:
            self.hidden_dim = hidden_dim
        for i in range(num_nn):
            net = DenseNN(
                split_dim, hidden_dims=self.hidden_dim, param_dims=param_dims, **kwargs
            )
            self.add_module("DenseNN" + str(i), net)
            self.nets.append(net)

    def forward(self, x):
        """ Forward pass through each network and stack it """
        x = x.reshape(-1, self.num_nn, self.split_dim)
        out = list(zip(*[net(x[:, i]) for i, net in enumerate(self.nets)]))
        result = [
            torch.hstack(o)
            .reshape(-1, self.num_nn, self.param_dims[i], self.split_dim)
            .squeeze(-1)
            .squeeze(0)
            for i, o in enumerate(out)
        ]
        return tuple(result)


class MixtureSameTransform(torch.distributions.MixtureSameFamily):
    """Trainable MixtureSameFamily using transformed distributions.  The component
    distribution should be of tpye TransformedDistribution!
    
    We implement rsample for component distributions that satisfy the autoregressive
    property. If your are not sure if your model is correct use "check_rsample" method

    """

    def parameters(self):
        """ Returns the learnable paramters of the model """
        if not self._mixture_distribution.logits.requires_grad:
            self._mixture_distribution.logits.requires_grad_(True)
        yield self._mixture_distribution.logits
        if hasattr(self._component_distribution, "parameters"):
            yield from self._component_distribution.parameters()
        elif hasattr(self._component_distribution, "transforms"):
            for t in self._component_distribution.transforms:
                if isinstance(t, nn.Module) or hasattr(t, "parameters"):
                    for para in t.parameters():
                        yield para
        else:
            pass

    def modules(self):
        """ Returns the modules of the model """
        if hasattr(self._component_distribution, "modules"):
            yield from self._component_distribution.modules()
        elif hasattr(self._component_distribution, "transforms"):
            for t in self._component_distribution.transforms:
                if isinstance(t, nn.Module):
                    yield t
        else:
            pass

    def clear_cache(self):
        if hasattr(self._component_distribution, "clear_cache"):
            self._component_distribution.clear_cache()

    def conditional_logprobs(self, x):
        """ Logprobs for each component and dimension."""
        x_pad = self._pad(x)
        link_transform = self._component_distribution.transforms[-1]
        transforms = self._component_distribution.transforms[:-1]
        x_delinked = link_transform.inv(x_pad)
        x = x_delinked
        eps = torch.zeros_like(x)
        jac = torch.zeros_like(x)
        for t in reversed(transforms):
            eps = t.inv(x)
            jac += t.log_abs_jacobian_diag(eps, x)
            x = eps

        base_dist = self._component_distribution.base_dist
        if isinstance(base_dist, Independent):
            log_prob = (
                base_dist.base_dist.log_prob(eps)
                - jac
                - link_transform.log_abs_det_jacobian(x_delinked, x_pad).squeeze()
            )
        else:
            log_prob = (
                base_dist.log_prob(eps)
                - jac
                - link_transform.log_abs_det_jacobian(x_delinked, x_pad).squeeze()
            )

        return log_prob

    def _pad(self, x):
        """ Pads the input, by repeating in "_num_component" times """
        x = x.reshape(-1, self._event_shape[0])
        return x.unsqueeze(1).repeat(1, self._num_component, 1)

    def conditional_cdf(self, x):
        """ Cdfs for each component and dimension. """
        x_pad = self._pad(x)
        transform = ComposeTransform(self._component_distribution.transforms)
        eps = transform.inv(x_pad)
        base_dist = self._component_distribution.base_dist
        if isinstance(base_dist, Independent):
            cdf = base_dist.base_dist.cdf(eps)
        else:
            cdf = base_dist.cdf(eps)
        return cdf

    def standardize(self, x):
        """ This transform converts samples from the distributions to Unif[0,1]
        samples. This works only if the autoregressive property holds."""
        log_prob_x = self.conditional_logprobs(x)
        cdf_x = self.conditional_cdf(x)

        cum_sum_logq_k = log_prob_x.cumsum(2).roll(1, 2)
        cum_sum_logq_k[:, :, 0] = 0
        cum_sum_logq_k

        logits_mix_prob = torch.log_softmax(self._mixture_distribution.logits, 0)
        self._mixture_distribution.probs = logits_mix_prob.exp().detach()
        log_posterior_weights_x = logits_mix_prob.unsqueeze(1) + cum_sum_logq_k
        posterior_weights = torch.softmax(log_posterior_weights_x, 1)

        return torch.sum(posterior_weights * cdf_x, 1)

    def rsample(self, shape=(), eps=1e-8):
        """ Implicit reparamterization """
        x = self.sample(shape).detach()
        x.requires_grad = True
        z = self.standardize(x)
        batch_jacobian = jacobian_in_batch(z, x) + (
            torch.eye(self.event_shape[-1]) * eps
        )
        surrogate = -torch.triangular_solve(z[..., None], batch_jacobian, upper=False)[
            0
        ].squeeze(-1)
        return x.detach() + (surrogate - surrogate.detach())

    def check_rsample(self, num_samples=100):
        """ The jacobian must be lower triangular such that rsample works. This is
        checked here """
        x = self.sample((num_samples,)).detach()
        x.requires_grad = True
        z = self.standardize(x)
        batch_jacobian = jacobian_in_batch(z, x).sum(0)
        if float(torch.triu(batch_jacobian, diagonal=1).sum()) != 0.0:
            warnings.warn(
                "The implicitly reparameterized samples require that the model satisfies the autoregressive property, which is not the case here!"
            )


class MixtureDiagGaussians(MixtureSameTransform):
    """ This implements a learnable Mixture of Gaussians with diagonal covariance.
    
        Args:
        num_components: Number of mixture components
        event_dim: Dimension of the event.
        loc: Starting location, as default this will be drawn randomly according to a
        standard normal distribution.
        scale: Scale, default is one.
        support: The support of the distribution
        """

    def __init__(
        self,
        num_components: int,
        event_dim: int,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        base_dist: Optional[Distribution] = None,
        support: Optional[Constraint] = real,
        check_rsample: bool = False,
        **kwargs,
    ):
        if base_dist is None:
            base_dist = Independent(
                torch.distributions.Normal(
                    torch.zeros(num_components, event_dim),
                    torch.ones(num_components, event_dim),
                ),
                1,
            )
        # Random mean
        if loc is None:
            loc = torch.randn(num_components, event_dim)
        if scale is None:
            scale = torch.ones(num_components, event_dim)

        t = AffineTransform(loc, scale).with_cache()
        link = biject_to(support).with_cache()
        qs = TransformedDistribution(base_dist, [t, link])

        # Create mixture distribution
        mix = Categorical(logits=torch.ones(num_components))
        super().__init__(mix, qs, **kwargs)

        self.transforms = [t, link]
        self.has_rsample = True
        if check_rsample:
            self.check_rsample()
            self.component_distribution.clear_cache()

    @property
    def mean(self):
        component_mean = self.component_distribution.transforms[0].loc
        mix = self.mixture_distribution.probs
        return mix @ component_mean

    @property
    def variance(self):
        component_var = self.component_distribution.transforms[0].scale ** 2
        component_mean2 = self.component_distribution.transforms[0].loc ** 2
        mean2 = self.mean ** 2
        mix = self.mixture_distribution.probs
        return mix @ (component_var + component_mean2 - mean2)


class MixtureFullGaussians(MixtureSameTransform):
    """This implements a learnable Mixture of Gaussians with full covariance.
    
    
    
    Args:
        num_components: Number of mixture components
        event_dim: Dimension of the event.
        loc: Starting location, as default this will be drawn randomly according to a
        standard normal distribution.
        scale_tril: Trinagular matrix as scale. As default this will have ones everywhere.
        support: The support of the distribution
    """

    def __init__(
        self,
        num_components: int,
        event_dim: int,
        loc: Optional[torch.Tensor] = None,
        scale_tril: Optional[torch.Tensor] = None,
        base_dist: Optional[torch.distributions.Distribution] = None,
        support: Optional[Constraint] = real,
        check_rsample: bool = False,
        **kwargs,
    ):
        if base_dist is None:
            base_dist = Independent(
                torch.distributions.Normal(
                    torch.zeros(num_components, event_dim),
                    torch.ones(num_components, event_dim),
                ),
                1,
            )
        if loc is None:
            loc = torch.randn(num_components, event_dim)
        if scale_tril is None:
            id = torch.eye(event_dim)
            b_id = id.reshape((1, event_dim, event_dim))
            scale_tril = torch.tril(b_id.repeat(num_components, 1, 1))

        t = LowerCholeskyAffine(loc, scale_tril).with_cache()
        link = biject_to(support).with_cache()
        qs = TransformedDistribution(base_dist, [t, link])
        mix = Categorical(logits=torch.ones(num_components))
        super().__init__(mix, qs, **kwargs)

        self.transforms = [t, link]
        self.has_rsample = True
        if check_rsample:
            self.check_rsample()
            self.component_distribution.clear_cache()

    @property
    def mean(self):
        component_mean = self.component_distribution.transforms[0].loc
        mix = self.mixture_distribution.probs
        return mix @ component_mean

    @property
    def variance(self):
        component_var = (
            torch.diagonal(
                self.component_distribution.transforms[0].scale_tril, dim1=-2, dim2=-1
            )
            ** 2
        )
        component_mean2 = self.component_distribution.transforms[0].loc ** 2
        mean2 = self.mean ** 2
        mix = self.mixture_distribution.probs
        return mix @ (component_var + component_mean2 - mean2)


class MixtureAffineAutoregressive(MixtureSameTransform):
    """A learnable Mixture of affine autoregressive flows 
    Args:
        num_components: Number of Mixture components.
        event_dim: Shape of events.
        base_dist: Base distribution used to construct the flows
        num_flows: Number of flows per component.
        support: The support of the distribution
    """

    def __init__(
        self,
        num_components: int,
        event_dim: int,
        base_dist: Optional[torch.distributions.Distribution] = None,
        num_flows: int = 1,
        support: Optional[Constraint] = real,
        check_rsample: bool = False,
        **kwargs,
    ):
        if base_dist is None:
            base_dist = Independent(
                torch.distributions.Normal(
                    torch.randn(num_components, event_dim),
                    torch.ones(num_components, event_dim),
                ),
                1,
            )
        self.nets = []
        self.transforms = []
        for _ in range(num_flows):
            net = StackedAutoRegressiveNN(num_components, event_dim, **kwargs)
            transform = AffineAutoregressive(net, log_scale_min_clip=-2).with_cache()
            self.nets.append(net)
            self.transforms.append(transform)
        link = biject_to(support).with_cache()
        self.transforms.append(link)
        qs = TransformedDistribution(base_dist, self.transforms)
        mix = Categorical(logits=torch.ones(num_components))
        super().__init__(mix, qs, **kwargs)

        self.has_rsample = True
        if check_rsample:
            self.check_rsample()
            self.component_distribution.clear_cache()


class MixtureSplineAutoregressive(MixtureSameTransform):
    """A learnable Mixture of spline autoregressive flows 
    Args:
        num_components: Number of Mixture components.
        event_dim: Shape of events.
        base_dist: Base distribution used to construct the flows
        num_flows: Number of flows per component.
        support: The support of the distribution
    
    """

    def __init__(
        self,
        num_components: int,
        event_dim: int,
        base_dist: Optional[torch.distributions.Distribution] = None,
        num_flows: int = 1,
        support: Optional[Constraint] = real,
        check_rsample: bool = False,
        **kwargs,
    ):
        if base_dist is None:
            base_dist = Independent(
                torch.distributions.Normal(
                    torch.randn(num_components, event_dim),
                    torch.ones(num_components, event_dim),
                ),
                1,
            )
        self.nets = []
        self.transforms = []
        count_bins = kwargs.pop("count_bins", 8)
        bound = kwargs.pop("bound", 3.0)
        order = kwargs.pop("order", "linear")
        param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
        for _ in range(num_flows):
            net = StackedAutoRegressiveNN(
                num_components, event_dim, param_dims=param_dims, **kwargs
            )
            transform = SplineAutoregressive(
                event_dim, net, count_bins=count_bins, bound=bound, order=order
            ).with_cache()
            self.nets.append(net)
            self.transforms.append(transform)
        link = biject_to(support).with_cache()
        self.transforms.append(link)
        qs = TransformedDistribution(base_dist, self.transforms)
        mix = Categorical(logits=torch.ones(num_components))
        super().__init__(mix, qs, **kwargs)

        self.has_rsample = True
        if check_rsample:
            self.check_rsample()
            self.component_distribution.clear_cache()


class Mixture(Distribution):
    """ This is a general Mixture distribution. """

    def __init__(self, cat: Categorical, components: Iterable[Distribution]):
        super().__init__()
        if isinstance(cat, Categorical):
            self._mixture_distribution = cat
            self._mixture_distribution
        else:
            raise ValueError("Mixture distribution must be Categorical")

        self._components = components
        self._validate_components
        self._num_components = len(components)
        self._event_shape = components[0].event_shape
        self._batch_shape = components[0].batch_shape
        self.has_rsample = False

    @property
    def transforms(self):
        transforms = []
        for comp in self._components:
            if hasattr(comp, "transforms"):
                transforms.extend(comp.transforms)
        return transforms

    def parameters(self):
        """ Returns the learnable paramters of the model """
        self._mixture_distribution.logits.requires_grad_(True)
        yield self._mixture_distribution.logits
        for comp in self.components:
            if hasattr(comp, "parameters"):
                yield from comp.parameters()
            elif hasattr(comp, "transforms"):
                for t in comp.transforms:
                    if hasattr(t, "parameters"):
                        for para in t.parameters():
                            yield para
            else:
                pass

    def modules(self):
        """ Returns the learnable modules of the model """
        for comp in self.components:
            if hasattr(comp, "modules"):
                yield from comp.modules()
            elif hasattr(comp, "transforms"):
                for t in comp.transforms:
                    if isinstance(t, nn.Module):
                        yield t
            else:
                pass

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return self._num_components

    def clear_cache(self):
        for comp in self._components:
            if hasattr(comp, "clear_cache"):
                comp.clear_cache()

    def log_prob(self, x):
        logmix = torch.log_softmax(self._mixture_distribution.logits, -1)
        logcomprobs = torch.stack(
            [self._components[k].log_prob(x) for k in range(self._num_components)]
        ).transpose(0, 1)

        return torch.logsumexp(logcomprobs + logmix, dim=-1)

    def sample(self, shape=torch.Size()):
        shape = torch.Size(shape)
        with torch.no_grad():
            samples = []
            num_samples = shape.numel()
            self._mixture_distribution.probs = torch.softmax(
                self._mixture_distribution.logits, -1
            )
            cat_samples = self._mixture_distribution.sample((num_samples,)).flatten()
            comps, counts = torch.unique(cat_samples, return_counts=True)
            for k in range(len(comps)):
                samples.append(self.components[comps[k]].sample((counts[k],)))
            samples = torch.vstack(samples)
            return samples.reshape(shape + (samples.shape[-1],))

    def rsample_components(self, shape=torch.Size()):
        return torch.stack([p.rsample(shape) for p in self._components]).transpose(0, 1)

    def _validate_components(self):
        if all((isinstance(p, Distribution)) for p in self.components):
            event_shapes = [p.event_shape for p in self.components]
            batch_shapes = [p.event_shape for p in self.components]
            assert all(ele == event_shapes[0] for ele in event_shapes)
            assert all(ele == batch_shapes[0] for ele in batch_shapes)
        else:
            for p in self.components:
                assert hasattr(p, "log_prob")
                assert hasattr(p, "sample")

    def __repr__(self):
        return f"Mixture({self.components})"


def build_mixture(
    event_shape: torch.Size,
    support: Constraint,
    num_components: int = 4,
    rsample=False,
    **kwargs,
) -> Mixture:
    """ This builds a mixture of normalizing flows.

    Args:
        event_shape: Dimension of the events
        support: Support of the distribution.
        num_components: Number of mixture components
        kwargs: Arguments for the type of flow
    
    Returns:
        MixtureOfFlows: Pytorch module that implements a trainable mixture of flows.
    
    """
    if not rsample:
        components = []
        cat = Categorical(torch.ones(num_components))
        for k in range(num_components):
            flow = build_flow(event_shape, support, **kwargs)
            components.append(flow)
        return Mixture(cat, components)
    else:
        flow = kwargs.pop("type")
        if flow == "affine_diag":
            return MixtureDiagGaussians(
                num_components, event_shape[-1], support=support, **kwargs
            )
        elif flow == "affine_tril":
            return MixtureFullGaussians(
                num_components, event_shape[-1], support=support, **kwargs
            )
        elif flow == "affine_autoregressive":
            return MixtureAffineAutoregressive(
                num_components, event_shape[-1], support=support, **kwargs
            )
        elif flow == "spline_autoregressive":
            return MixtureSplineAutoregressive(
                num_components, event_shape[-1], support=support, **kwargs
            )
        else:
            raise NotImplementedError(
                "Currently only mixtures whos component distributions satisfy the autoregressive property have implicit rsample methods."
            )


# Deprecated may remove
# class MixtureOfFlows(nn.Module):
#     """ This is a general class for Mixture distributions for arbitrary components.
#     However this class  """

#     def __init__(self, components: Iterable[Distribution]):
#         super().__init__()
#         self.num_components = len(components)
#         self.components = components
#         self.event_dim = components[0].event_shape[-1]
#         self.logit_mixtures = torch.nn.Parameter(torch.ones(self.K), requires_grad=True)

#         modules = []
#         for k, comp in enumerate(components):
#             module_compk = nn.ModuleList(
#                 [t for t in comp.transforms if isinstance(t, nn.Module)]
#             )
#             modules.append(module_compk)
#             self.add_module("component_" + str(k), module_compk)
#         self.modules = modules

#     def log_prob(self, x):
#         logmix = torch.log_softmax(self.logit_mixtures, 0)
#         logcomprobs = torch.vstack(
#             [self.components[k].log_prob(x).T for k in range(self.K)]
#         ).T

#         return torch.logsumexp(logcomprobs + logmix, -1)

#     def sample(self, shape):
#         with torch.no_grad():
#             num_samples = np.prod(shape)
#             mix = torch.softmax(self.logit_mixtures, 0)
#             ks = torch.multinomial(mix, num_samples, replacement=True)
#             comps, counts = torch.unique(ks, return_counts=True)
#             samples = torch.vstack(
#                 [self.components[k].sample((nums,)) for k, nums in zip(comps, counts)]
#             )
#             return samples[torch.randperm(num_samples)].reshape(
#                 shape + (self.event_dim,)
#             )

#     def rsample_comp(self, shape, k):
#         return self.components[k].rsample(shape)

#     def build_loss_elbo(self, optimizer):
#         n_particels = optimizer.n_particles
#         prior = optimizer.posterior._prior
#         ll = optimizer.posterior.net

#         def loss(x_obs):
#             x_obs = x_obs.repeat(n_particels * self.K, 1)
#             mix = torch.softmax(self.logit_mixtures, 0)
#             samples = torch.vstack(
#                 [self.rsample_comp((n_particels,), k) for k in range(self.K)]
#             )
#             log_q = self.log_prob(samples)
#             log_prior = prior.log_prob(samples)
#             log_ll = ll.log_prob(x_obs, context=samples)
#             elbo = log_ll + log_prior - log_q
#             loss = -mix @ elbo.reshape(self.K, -1).mean(-1)
#             return loss, loss.clone().detach()

#         return loss

#     def build_loss_renjey(self, optimizer):
#         n_particels = optimizer.n_particles
#         prior = optimizer.posterior._prior
#         ll = optimizer.posterior.net
#         alpha = optimizer.alpha

#         def loss(x_obs):
#             x_obs = x_obs.repeat(n_particels * self.K, 1)
#             mix = torch.softmax(self.logit_mixtures, 0)
#             samples = torch.vstack(
#                 [self.rsample_comp((n_particels,), k) for k in range(self.K)]
#             )
#             log_q = self.log_prob(samples)
#             log_prior = prior.log_prob(samples)
#             log_ll = ll.log_prob(x_obs, context=samples)
#             elbo_particles = log_ll + log_prior - log_q
#             elbo_particles = elbo_particles.reshape(self.K, -1)
#             # Weights
#             logweights = (1 - alpha) * elbo_particles.T.clone().detach()
#             mean_log_weights = torch.logsumexp(logweights, 0) - np.log(n_particels)
#             logweights = logweights - mean_log_weights
#             weights = logweights.exp()

#             surrogate_loss = -torch.mean(mix @ (elbo_particles * weights.T))
#             loss = -mix.clone().detach() @ mean_log_weights * 1 / (1 - alpha)
#             return surrogate_loss, loss

#         return loss
