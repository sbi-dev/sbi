import torch
from torch import nn

import numpy as np  

class ElboOptimizer():
    def __init__(self, posterior, elbo_particles=64, optimizer=torch.optim.Adam, clip_value=10., **kwargs):
        self.posterior = posterior
        self.elbo_particles = elbo_particles
        self.clip_value = clip_value
        self._kwargs = kwargs
        # TODO change for more general usage
        self.modules = nn.ModuleList([t for t in posterior.q.transforms[0].parts if isinstance(t, nn.Module)])
        opt_kwargs = self.__filter_kwrags_for_func(optimizer.__init__)
        self.optimizer = optimizer(self.modules.parameters(), **opt_kwargs)

    def step(self, x_obs):
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        loss = - elbo_particles.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        return loss.detach()


    def generate_elbo_particles(self, x_obs, n_samples=1):
        x_obs = x_obs.repeat(n_samples,1)
        samples = self.posterior.q.rsample((n_samples,))
        log_q = self.posterior.q.log_prob(samples)
        log_ll = self.posterior.net.log_prob(x_obs, context=samples)
        log_prior = self.posterior._prior.log_prob(samples)
        elbo = log_ll + log_prior - log_q
        return elbo
        

    def __filter_kwrags_for_func(self, f):
        args = f.__code__.co_varnames
        new_kwargs = dict([(key,val) for key,val in self._kwargs.items() if key in args])
        return new_kwargs


class RenjeyDivergenceOptimizer(ElboOptimizer):
    def __init__(self, posterior, alpha, elbo_particles=512, optimizer=torch.optim.Adam, clip_value=10., **kwargs):
        super().__init__(posterior, elbo_particles, optimizer, clip_value, **kwargs)
        self.alpha = alpha 

    def step(self, x_obs):
        if isinstance(self.alpha, float):
            return self.step_surrogated(x_obs)
        elif self.alpha == "max":
            return self.step_max(x_obs) 
        elif self.alpha == "min":
            return self.step_min(x_obs)
    
    def step_surrogated(self, x_obs):
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        logweights = (1-self.alpha)*elbo_particles.clone().detach()
        mean_log_weights = torch.logsumexp(logweights,0) - np.log(self.elbo_particles)
        normed_logweights = logweights - mean_log_weights
        weights = normed_logweights.exp()
        surrogate_loss = -torch.mean(weights*elbo_particles)
        surrogate_loss.backward() 
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        loss = -mean_log_weights/(1-self.alpha)
        return loss 

    def step_full(self, x_obs):
        # TODO may remove
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        weighted_elbos = elbo_particles * (1-self.alpha)
        loss = (torch.logsumexp(weighted_elbos) - np.log(elbo_particels))/(1-self.alpha)
        loss.backward() 
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        return loss.detach()

    def step_max(self, x_obs):
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        loss = -elbo_particles.max()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        return loss.detach()

    def step_min(self, x_obs):
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        loss = -elbo_particles.min()
        loss.backward() 
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        return loss.detach()

class TailAdaptivefDivergenceOptimizer(ElboOptimizer):

    def __init__(self, posterior, beta= -1, elbo_particles=512, optimizer=torch.optim.Adam, clip_value=10., **kwargs):
        super().__init__(posterior, elbo_particles, optimizer, clip_value, **kwargs)
        self.beta = beta 

    def step(self,x_obs):
        self.optimizer.zero_grad()
        elbo_particles = self.generate_elbo_particles(x_obs, n_samples=self.elbo_particles)
        gammas = self.get_tail_adaptive_weights(elbo_particles)
        
        surrogate_loss = -torch.sum(torch.unsqueeze(gammas*elbo_particles,1))
        surrogate_loss.backward() 
        nn.utils.clip_grad_norm_(self.modules.parameters(),self.clip_value)
        self.optimizer.step()
        return surrogate_loss.detach() 

    def get_tail_adaptive_weights(self, elbo_particles):
        weights = torch.exp(elbo_particles - elbo_particles.max())
        prob = torch.sign(weights.unsqueeze(1) - weights.unsqueeze(0))
        prob = torch.greater(prob,0.5).float() 
        F = 1-prob.sum(1)/self.elbo_particles  
        gammas = F**self.beta 
        gammas /= gammas.sum()  
        return gammas.clone().detach()





    
