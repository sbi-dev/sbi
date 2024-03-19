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
            sde_type: Optional[str]='VP',
            noise_minmax: Tuple[float, float]=(0.1, 20.),
            weight_fn: Union[str, Callable]='variance',
            embedding_net: nn.Module = nn.Identity(),
            ) -> None:
        """Class for score estimators with variance exploding (NCSN), variance preserving (DDPM), or sub-variance preserving SDEs.
        """        
        super().__init__(net, condition_shape)
    
        self.net = net
        self.condition_shape = condition_shape        
        # Set mean and standard deviation functions based on the type of SDE and noise hyperparameter bounds.
        self._set_mean_std_fn(sde_type, noise_minmax)
        # Set lambdas (variance weights) function
        self._set_weight_fn(weight_fn)

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        score = self.net(input, condition)
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
        std = self.std_fn(input, times)

        # Compute true score.
        score_target = - (mean - input) / (std**2)
        
        # Predict score.
        score_pred = self.forward(input, condition, times)

        # Compute weights over time.
        lams = self.weight_fn(std)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_target - score_pred).pow(2), axis=-1)
        loss = torch.mean(lams * loss)

        return loss        
    
    def likelihood(self):
        """Return input likelihood (under the ODE flow formulation)."""
        raise NotImplementedError

    # def _set_noise_schedule(self, noise_schedule):
    #     """Set noise schedule."""
    #     if noise_schedule=="cosine":
    #         self.noise_schedule = self._cosine_noise_schedule()
    #     elif noise_schedule=="linear":
    #         self.noise_schedule = self._linear_noise_schedule()
    #     elif noise_schedule=="exponential":
    #         self.noise_schedule = self._exp_noise_schedule()
    #     elif callable(noise_schedule):
    #         self.noise_schedule = noise_schedule
    #     else:
    #         raise ValueError(f"Noise schedule {noise_schedule} not recognized.")        

    def _set_weight_fn(self, weight_fn):
        """Get the weight function."""
        if weight_fn=="identity":
           self.weight_fn = lambda sigma: 1
        elif weight_fn=="variance":
            # From Song & Ermon, NeurIPS 2019.
            self.weight_fn = lambda sigma: sigma**2.
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")                


    def _set_mean_std_fn(self, sde_type, noise_minmax):
        """Get the mean and standard deviation functions based on the type of SDE."""
        if self.sde_type=="VE":
            # Variance exploding.
            sigma_min, sigma_max = noise_minmax
            mean_fn, std_fn = self._VE_mean_std_fn(sigma_min, sigma_max)
        elif self.sde_type=="VP":
            # Variance preserving.
            beta_min, beta_max = noise_minmax
            mean_fn, std_fn = self._VP_mean_std_fn(beta_min, beta_max)
        
        elif self.sde_type=="subVP":
            # Sub-variance preserving.
            mean_fn, std_fn = self._subVP_mean_std_fn(noise_minmax)
        
        self.sde_type = sde_type
        self.mean_fn = mean_fn
        self.std_fn = std_fn

        
    def _VE_mean_std_fn(self, sigma_min, sigma_max):
        # Variance exploding, i.e., SMLD/NCSN.
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        def mean_fn(self, input, times):
            return input
        def std_fn(self, input, times):
            return self.sigma_min.pow(2.) * (self.sigma_max / self.sigma_min).pow(2.*times)
        return mean_fn, std_fn
    
    def _VP_mean_std_fn(self, beta_min, beta_max):
        # Variance preserving, i.e., DDPM.
        self.beta_min = beta_min
        self.beta_max = beta_max        
        def mean_fn(self, input, times):
            return torch.exp(-0.25*times.pow(2.)*(self.beta_max-self.beta_min)-0.5*times*self.beta_min)*input
        def std_fn(self, input, times):
            return 
        return None, None
    
    def _subVP_mean_std_fn(self):
        # Sub-variance preserving
        return None, None