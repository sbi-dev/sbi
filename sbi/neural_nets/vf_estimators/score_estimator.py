from typing import Union, Callable

import torch
from torch import Tensor, nn
from sbi.neural_nets.vf_estimators.base import VectorFieldEstimator



class ScoreEstimator(VectorFieldEstimator):
    r"""Score matching for score-based generative models (e.g., denoising diffusion).
    The estimator neural network (this class) learns the score function, i.e., gradient of 
    the conditional probability density with respect to the input, which can be used to
    generate samples from the target distribution by solving the SDE starting from the 
    base (Gaussian) distribution.

    Relevant literature: 
    - Score-based generative modeling through SDE: https://arxiv.org/abs/2011.13456
    - Denoising diffusion probabilistic models: https://arxiv.org/abs/2006.11239
    - Noise conditional score networks: https://arxiv.org/abs/1907.05600
    """

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable],
    ) -> None:
        r""" Score estimator class that estimates the conditional score function, i.e., gradient of the density p(xt|x0).

        Args:
            net: Score estimator neural network, should take a list [input, condition, and time (in [0,1])].
            condition_shape: Shape of the conditioning variable.
            weight_fn: Function to compute the weights over time. Can be one of the following:
                - "identity": constant weights (1.),
                - "max_likelihood": weights proportional to the diffusion function, or
                - a custom function that returns a Callable.

        """
        super().__init__(net, condition_shape)

        # Set lambdas (variance weights) function.
        self._set_weight_fn(weight_fn)
        
        # Min time for diffusion (0 can be numerically unstable).
        self.T_min = 1e-3

        # These still need to be computed (mean and std of the noise distribution).
        self.mean = 0.0          
        self.std = 1.0

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Conditional mean function, E[xt|x0], specifying the "mean factor" at a given time, 
        which is always multiplied by x0 to get the mean of the noise distribution,
        
        i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].
        
        Raises:
            NotImplementedError: This method is implemented in each individual SDE classes.
        """
        raise NotImplementedError
    
    def mean_fn(self, x0: Tensor, times: Tensor) -> Tensor:
        r"""Mean function of the SDE, which just multiplies the specific "mean factor" by 
        the original input x0, to get the mean of the noise distribution,
        
        i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            x0: Initial input data.
            times: SDE time variable in [0,1].

        Returns:
            Mean of the noise distribution at a given time.
        """
        return self.mean_t_fn(times) * x0

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function of the noise distribution at a given time,
        
        i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].
        
        Raises:
            NotImplementedError: This method is implemented in each individual SDE classes.
        """
        raise NotImplementedError

    def drift_fn(self, input: Tensor, times: Tensor)-> Tensor:
        r"""Drift function, f(x,t), of the SDE described by dx = f(x,t)dt + g(x,t)dW.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE classes.
        """
        raise NotImplementedError
    
    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function, g(x,t), of the SDE described by dx = f(x,t)dt + g(x,t)dW.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE classes.
        """
        raise NotImplementedError

    def forward(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        r"""Forward pass of the score estimator network to compute the conditional score at a given time.

        Args:
            input: Original data, x0.
            condition: Conditioning variable.
            times: SDE time variable in [0,1].
        
        Returns:
            Score (gradient of the density) at a given time, matches input shape.
        """
        # Expand times if it's a scalar.
        if times.ndim == 1:
            times = times.expand(input.shape[0])            

        # Predict noise and divide by standard deviation to mirror target score.
        # TODO Replace with Michaels magic shapeing function
        #print(input.shape, condition.shape, times.shape)
        if times.shape.numel() == 1:
            times = torch.repeat_interleave(times[None], input.shape[0], dim=0)
            times = times.reshape((input.shape[0],))

        input_shape = input.shape
        input = input.reshape((-1, input.shape[-1]))
        condition = condition.reshape(-1, condition.shape[-1])
        condition = torch.repeat_interleave(condition, input.shape[0]//condition.shape[0], dim=0)
        #print(input.shape, condition.shape, times.shape)
        eps_pred = self.net([input, condition, times])
        std = self.std_fn(times)
        eps_pred = eps_pred
        score =  eps_pred / std
        return score.reshape(input_shape)
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:        
        r"""Defines the denoising score matching loss (e.g., from Song et al., ICLR 2021).
        A random diffusion time is sampled from [0,1], and the network is trained to predict the
        score of the true conditional distribution given the noised input, which is equivalent
        to predicting the (scaled) Gaussian noise added to the input.

        Args:
            input: Original data, x0.
            condition: Conditioning variable.

        Returns:
            MSE between target score and network output, scaled by the weight function.
        
        """
        # Sample diffusion times.
        times = torch.clip(torch.rand((input.shape[0],)), self.T_min, 1.0)

        # Sample noise.
        eps = torch.randn_like(input)

        # Compute mean and standard deviation.
        mean = self.mean_fn(input, times)
        std = self.std_fn(times)

        # Get noised input, i.e., p(xt|x0).
        input_noised = mean + std * eps

        # Compute true score: -(mean - noised_input) / (std**2).
        score_target = -eps / std

        # Predict score from noised input and diffusion time.
        score_pred = self.forward(input_noised, condition, times)

        # Compute weights over time.
        weights = self.weight_fn(times)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_target - score_pred)**2.0, dim=-1)        

        return weights*loss

    def _set_weight_fn(self, weight_fn: Union[str, Callable]):
        """Set the weight function.

        Args:
            weight_fn: Function to compute the weights over time. Can be one of the following:
                - "identity": constant weights (1.),
                - "max_likelihood": weights proportional to the diffusion function, or
                - a custom function that returns a Callable.        
        """
        if weight_fn == "identity":
            self.weight_fn = lambda times: 1
        elif weight_fn == "max_likelihood":
            self.weight_fn = lambda times: self.diffusion_fn(torch.ones((1,)),times)**2        
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")


class VPScoreEstimator(ScoreEstimator):
    """Class for score estimators with variance preserving SDEs (i.e., DDPM)."""
    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 20.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        a = torch.exp(
                -0.25 * times**2.0 * (self.beta_max - self.beta_min)
                - 0.5 * times * self.beta_min
            )
        return a.unsqueeze(-1)
    
    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std =  1.0 - torch.exp(
            -0.5 * times**2.0 * (self.beta_max - self.beta_min)
            - times * self.beta_min
        )
        return torch.sqrt(std.unsqueeze(-1))

    def _beta_schedule(self, times: Tensor) -> Tensor:
        """Linear beta schedule for mean scaling in variance preserving SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Beta schedule at a given time.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for variance preserving SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self._beta_schedule(times)
        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)
        return phi * input

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for variance preserving SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        g = torch.sqrt(
            self._beta_schedule(times)
        )
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        return g


class subVPScoreEstimator(ScoreEstimator):
    """Class for score estimators with sub-variance preserving SDEs."""
    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 20.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for sub-variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        a = torch.exp(
                -0.25 * times**2.0 * (self.beta_max - self.beta_min)
                - 0.5 * times * self.beta_min
        )                        
        return a.unsqueeze(-1)

    def std_fn(self, times: Tensor) -> Tensor:   
        """Standard deviation function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """     
        std = 1.0 - torch.exp(
                -0.5 * times**2.0 * (self.beta_max - self.beta_min)
                - times * self.beta_min
            )        
        return std.unsqueeze(-1)

    def _beta_schedule(self, times: Tensor) -> Tensor:
        """Linear beta schedule for mean scaling in sub-variance preserving SDEs.
        (Same as for variance preserving SDEs.)

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Beta schedule at a given time.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input: Tensor, times:Tensor) -> Tensor:
        """Drift function for sub-variance preserving SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self._beta_schedule(times) 
        
        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)
        
        return phi * input

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for sub-variance preserving SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        g = torch.sqrt(
            self._beta_schedule(times)
            * (-torch.exp(-2 * self.beta_min * times - (self.beta_max - self.beta_min) * times**2)))
        
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        
        return g
        

class VEScoreEstimator(ScoreEstimator):
    """Class for score estimators with variance exploding SDEs (i.e., NCSN / SMLD)."""

    def __init__(
        self,
        net: nn.Module,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for variance exploding SDEs, which is always 1.
        
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        return torch.tensor([1.0])

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance exploding SDEs.
        
        Args:
            times: SDE time variable in [0,1].
        
        Returns:
            Standard deviation at a given time.
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** times
        return std.unsqueeze(-1)

    def _sigma_schedule(self, times: Tensor) -> Tensor:
        """Geometric sigma schedule for variance exploding SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Sigma schedule at a given time.
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** times

    def drift_fn(self, input: Tensor, times: Tensor)-> Tensor:
        """Drift function for variance exploding SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        return torch.tensor([0.0])

    def diffusion_fn(self, input:Tensor, times: Tensor)-> Tensor:
        """Diffusion function for variance exploding SDEs.
        
        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        g = self._sigma_schedule(times) * torch.sqrt(2 * torch.log(torch.Tensor(self.sigma_max / self.sigma_min)))
        
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        
        return g
