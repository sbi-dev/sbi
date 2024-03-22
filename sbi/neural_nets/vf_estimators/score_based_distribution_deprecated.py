# BaseClass is a class for conditionalDistributions sample() and log_prob() methods, e.g torch.distributions.Distribution,
from sbi.neural_nets.vf_estimators.score_estimator import ScoreEstimator
import torch
from torch.distributions.normal import Normal
from zuko.utils import odeint
from torch import Tensor
from torch.distributions import Distribution


class ScoreDistribution(Distribution):
    """Wrapper around ScoreEstimator to have objects with sample function"""

    def __init__(self, 
                 score_estimator: ScoreEstimator, 
                 condition=None, 
                 sample_with="sde",
                 event_shape = torch.Size([]),
        ):
        super().__init__()
        self.score_estimator = score_estimator
        self.drift_fn = score_estimator.drift_fn
        self.diffusion_fn = score_estimator.diffusion_fn
        self._event_shape = event_shape
        #self.condition_shape = score_estimator.condition_shape
        self.step_size = 1000
        self.noise_distribution = Normal(
            loc=score_estimator.mean * torch.ones(event_shape), scale=score_estimator.std * torch.ones(event_shape)
        )
        self.condition = condition
        self.sample_with = sample_with
        

    def log_prob(self, inputs):
        raise NotImplementedError()

    def sample_with_sde(self, sample_shape, condition):
        if condition is None:
            raise ValueError("No condition is passed to SBDistribution.sample.")
        else:
            theta = self.noise_distribution.sample(sample_shape=sample_shape)
            delta_t = (
                1 / self.step_size
            )  # depends if we want to ode and sde term by step_size, probably right?
            condition = condition[None,...]
            condition = condition.repeat(sample_shape[0], 1)

            for step in range(self.step_size):
                t = (step + 1) / self.step_size
                t = t* torch.ones(sample_shape)
                print(theta.shape, condition.shape, t.shape)
                theta = theta + (self.drift_fn(input=theta, t=t)
                        - (self.diffusion_fn(t=t)) ** 2
                        * self.score_estimator(input=theta, condition=condition, times=t)) * delta_t + self.diffusion_fn(t=t) * torch.randn(sample_shape + self._event_shape) * delta_t

                return theta

    def sample_with_ode(self, sample_shape, condition):
        def f(theta: Tensor, t) -> Tensor:
            return self.drift_fn(input=theta, t=t) - 0.5*(
                self.diffusion_fn(t=t)
            ) ** 2 * self.score_estimator(theta, condition=condition, times=t)

        theta0 = self.noise_distribution.sample(
            sample_shape=sample_shape
        )
        theta1 = odeint(f, theta0, 0.0, 1.0)

        return theta1

    def sample(self, sample_shape: torch.Size, condition) -> torch.Tensor:
        if self.condition == None:
            raise ValueError("sampling from the posterior is only possible when specifying a condition")
        if self.sample_with == "sde":
            return self.sample_with_sde(sample_shape = sample_shape, condition = self.condition)
        if self.sample_with == "ode":
            return self.sample_with_ode(sample_shape=sample_shape, condition = self.condition)