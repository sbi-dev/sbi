import torch
from torch import nn

import numpy as np


class MixtureOfFlows(nn.Module):
    def __init__(self, components, train_mixtures=False):
        super().__init__()
        self.K = len(components)
        self.logit_mixtures = torch.nn.Parameter(torch.ones(self.K), requires_grad=train_mixtures)
        self.components = components
        self.event_dim = components[0].shape()[0]
        modules = []
        for k, comp in enumerate(components):
            module_compk = nn.ModuleList(
                [t for t in comp.transforms if isinstance(t, nn.Module)]
            )
            modules.append(module_compk)
            self.add_module("component_" + str(k), module_compk)
        self.modules = modules

    def log_prob(self, x):
        logmix = torch.log_softmax(self.logit_mixtures, 0)
        logcomprobs = torch.vstack(
            [self.components[k].log_prob(x).T for k in range(self.K)]
        ).T

        return torch.logsumexp(logcomprobs + logmix, -1)

    def sample(self, shape):
        with torch.no_grad():
            num_samples = np.prod(shape)
            mix = torch.softmax(self.logit_mixtures, 0)
            ks = torch.multinomial(mix, num_samples, replacement=True)
            comps, counts = torch.unique(ks, return_counts=True)
            samples = torch.vstack(
                [self.components[k].sample((nums,)) for k, nums in zip(comps, counts)]
            )
            return samples[torch.randperm(num_samples)].reshape(
                shape + (self.event_dim,)
            )

    def rsample_comp(self, shape, k):
        return self.components[k].rsample(shape)

    def build_loss_elbo(self, optimizer):
        n_particels = optimizer.n_particles
        prior = optimizer.posterior._prior
        ll = optimizer.posterior.net

        def loss(x_obs):
            x_obs = x_obs.repeat(n_particels * self.K, 1)
            mix = torch.softmax(self.logit_mixtures, 0)
            samples = torch.vstack(
                [self.rsample_comp((n_particels,), k) for k in range(self.K)]
            )
            log_q = self.log_prob(samples)
            log_prior = prior.log_prob(samples)
            log_ll = ll.log_prob(x_obs, context=samples)
            elbo = log_ll + log_prior - log_q
            loss = -mix @ elbo.reshape(self.K, -1).mean(-1)
            return loss, loss.clone().detach()

        return loss

    def build_loss_renjey(self, optimizer):
        n_particels = optimizer.n_particles
        prior = optimizer.posterior._prior
        ll = optimizer.posterior.net
        alpha = optimizer.alpha

        def loss(x_obs):
            x_obs = x_obs.repeat(n_particels * self.K, 1)
            mix = torch.softmax(self.logit_mixtures, 0)
            samples = torch.vstack(
                [self.rsample_comp((n_particels,), k) for k in range(self.K)]
            )
            log_q = self.log_prob(samples)
            log_prior = prior.log_prob(samples)
            log_ll = ll.log_prob(x_obs, context=samples)
            elbo_particles = log_ll + log_prior - log_q
            elbo_particles = elbo_particles.reshape(self.K, -1)
            # Weights
            logweights = (1 - alpha) * elbo_particles.T.clone().detach()
            mean_log_weights = torch.logsumexp(logweights, 0) - np.log(n_particels)
            logweights = logweights - mean_log_weights
            weights = logweights.exp()

            surrogate_loss = -torch.mean(mix @ (elbo_particles * weights.T))
            loss = -mix.clone().detach() @ mean_log_weights * 1 / (1 - alpha)
            return surrogate_loss, loss

        return loss

