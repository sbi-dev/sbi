from typing import Tuple, Union, Optional, Callable

import torch
from torch import Tensor, nn

from sbi.neural_nets.vf_estimators import VectorFieldEstimator
from sbi.types import Shape
from torch. import Distribution

class ScoreEstimator(VectorFieldEstimator):
    r"""Score estimator for denoising diffusion probabilistic models (and similar).    
    """

    def __init__(self, net: nn.Module, 
                 condition_shape: torch.Size, 
                 noise_schedule: Union[str, Callable]='cosine',
                 sde_type: Optional[str]=None, 
                 mean_func: Optional[Union[str, Callable]]=None, 
                 var_func: Optional[Union[str, Callable]]=None
                 ) -> None:
        """Base class for score estimators.

        

        """
        # Need mean, variance, and weighting functions (defined as strings?)
        super().__init__(net, condition_shape)

    
        self.net = net


    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        score = self.net(input, condition)
        return score
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        # Sample a diffusion time, plug in the alpha and betas to get the noise
        # and return the MSE loss between the noise estimate and the true noise

        # Sample diffusion time.
        t = torch.rand((input.shape[0], 1))

        # Compute beta from noise schedule.

        # Compute mean and variance.

        # Compute true score.
        true_score = None

        # Compute MSE loss between network output and true score.
        loss = (self.forward(input, condition) - true_score)**2


        raise NotImplementedError
    
    def likelihood():
        """Return the likelihood of the input under the ODE formulation."""
        raise NotImplementedError

    def _get_noise_schedule(self):


    def _get_mean_var_func(self):
        """Get the mean and variance functions based on the type of SDE."""
        if self.sde_type=="VP":
            # Variance preserving.
            mean_func, var_func = self._VP_mean_var_func()            
        elif self.sde_type=="VE":
            # Variance exploding.
            mean_func, var_func = self._VE_mean_var_func()
        elif self.sde_type=="subVP":
            # Sub-Variance preserving.
            mean_func, var_func = self._subVP_mean_var_func()
        
        self.mean_func = mean_func
        self.var_func = var_func
        
    def _VE_mean_var_func(self):
        # Variance exploding
        def mean_func(input, t):
            return x
        def std_func(input, t):
            return x
        return mean_func, std_func
    
    def _VP_mean_var_func(self):
        # Variance preserving
        return None, None
    
    def _subVP_mean_var_func(self):
        # Sub-Variance preserving
        return None, None