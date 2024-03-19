from typing import Tuple, Union, Optional, Callable

import torch
from torch import Tensor, nn

from sbi.neural_nets.vf_estimators import VectorFieldEstimator
from sbi.types import Shape

class ScoreEstimator(VectorFieldEstimator):
    r"""Score estimator for score-based generative models (e.g., denoising diffusion).
    """

    def __init__(
            self, 
            net: nn.Module,
            condition_shape: torch.Size,
            sde_type: str='VP',
            noise_minmax: Tuple[float, float]=(0.1, 20.),
            weight_fn: Union[str, Callable]='variance',
            embedding_net: nn.Module = nn.Identity(),
            ) -> None:
        """
        Class for score estimators with variance exploding (NCSN), variance preserving (DDPM), or sub-variance preserving SDEs.
        """
        super().__init__(net, condition_shape)
        if net is None:
            # Define a simple torch MLP network if not provided.
            nn.MLP()



            

        elif isinstance(net, nn.Module):
            self.net = net

        self.condition_shape = condition_shape
        # Set mean and standard deviation functions based on the type of SDE and noise bounds.
        self._set_mean_std_fn(sde_type, noise_minmax)
        # Set lambdas (variance weights) function
        self._set_weight_fn(weight_fn)

    def forward(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        score = self.net(input, condition, times)
        # TO DO: divide by std here also
        return score
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        """Denoising score matching loss (Song et al., ICLR 2021)."""
        # Sample diffusion times.
        times = torch.rand((input.shape[0],))

        # Sample noise
        eps = torch.randn_like(input)

        # # Compute variances from noise schedule.
        # sigma = self.noise_schedule(times)

        # Compute mean and standard deviation.
        mean = self.mean_fn(input, times)
        std = self.std_fn(times)

        # Get noised input, i.e., p(xt|x0)
        noised_input = mean + std * eps

        # Compute true score: -(mean - noised_input) / (std**2)
        score_target = - eps/std
        
        # Predict score.
        score_pred = self.forward(noised_input, condition, times)

        # Compute weights over time.
        weights = self.weight_fn(std)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_target - score_pred).pow(2.), axis=-1)
        loss = torch.mean(weights * loss)

        return loss
    
    def likelihood(self):
        """Return input likelihood (under the ODE flow formulation)."""
        raise NotImplementedError        

    def _set_weight_fn(self, weight_fn):
        """Get the weight function."""
        if weight_fn=="identity":
           self.weight_fn = lambda sigma: 1
        elif weight_fn=="variance":
            # From Song & Ermon, NeurIPS 2019.
            self.weight_fn = lambda sigma: sigma.pow(2.)
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")                

    def _set_mean_std_fn(self, sde_type, noise_minmax):
        """Get the mean and standard deviation functions based on the type of SDE."""
        self.sde_type = sde_type
        if self.sde_type=="VE":
            # Variance exploding.
            self.sigma_min, self.sigma_max = noise_minmax
            mean_fn, std_fn = self._get_VE_mean_std_fn()
        elif self.sde_type=="VP":
            # Variance preserving.
            self.beta_min, self.beta_max = noise_minmax
            mean_fn, std_fn = self._get_VP_mean_std_fn()
        
        elif self.sde_type=="subVP":
            # Sub-variance preserving.
            self.beta_min, self.beta_max = noise_minmax
            mean_fn, std_fn = self._get_subVP_mean_std_fn()
                
        self.mean_fn = mean_fn
        self.std_fn = std_fn
        
    def _beta_schedule(self, times):
        return self.beta_min + (self.beta_max - self.beta_min) * times
    
    def _sigma_schedule(self, times):
        return self.sigma_min * (self.sigma_max / self.sigma_min).pow(times)
        
    def _get_VE_mean_std_fn(self):
        # Variance exploding, i.e., SMLD/NCSN.        
        def mean_fn(self, x0, times):
            return x0
        def std_fn(self, times):
            return self.sigma_min.pow(2.) * (self.sigma_max / self.sigma_min).pow(2.*times)
        return mean_fn, std_fn
    
    def _get_VP_mean_std_fn(self):
        # Variance preserving, i.e., DDPM.        
        def mean_fn(self, x0, times):
            return torch.exp(-0.25*times.pow(2.)*(self.beta_max-self.beta_min)-0.5*times*self.beta_min)*x0
        def std_fn(self, times):            
            return 1.-torch.exp(-0.5*times.pow(2.)*(self.beta_max-self.beta_min)-times*self.beta_min)
        return mean_fn, std_fn
    
    def _get_subVP_mean_std_fn(self):        
        # Sub-variance preserving.        
        def mean_fn(self, x0, times):
            return torch.exp(-0.25*times.pow(2.)*(self.beta_max-self.beta_min)-0.5*times*self.beta_min)*x0
        def std_fn(self, times):
            return (1.-torch.exp(-0.5*times.pow(2.)*(self.beta_max-self.beta_min)-times*self.beta_min)).power(2.)
        return mean_fn, std_fn
        