import torch
from torch import Tensor

from sbi import utils as utils


class ExamplePosterior:
    """
    Class that builds a density with broad marginals and narrow conditionals.

    This is used only for the tutorial on conditional correlations.

    It has the key function that a `NeuralPosterior` object would also have. This is
    because, in the tutorial, we pretend that the posterior was obtained with SNPE --
    even though in fact it is just an `ExamplePosterior`.

    The distribution is a plane in 8-dimensional space with a bit of uniform noise
    around it. The plane is tilted, which makes the marginals broad but the conditionals
    narrow.
    """

    def __init__(self):
        # Vector that is orthogonal to the plane that has high probability.
        self.normal_vec = torch.tensor([0.9, 1.1, -1.0])
        self.noise_factor = 1.0

    def sample(self, sample_shape: torch.Size):
        """
        Return samples from the toy density.

        We first sample from a box uniform and then compute their L1-distance to a
        hyperplane in the 8D parameter space. We then accept with probability
        (1.-distance). If the distance is larger than 1.0, we never accept.
        """
        num_dim = 3
        all_samples = torch.empty(0, num_dim)
        num_accepted = 0
        while num_accepted < sample_shape[0]:
            proposed_samples = utils.BoxUniform(
                -2 * torch.ones(num_dim), 2 * torch.ones(num_dim)
            ).sample(sample_shape)
            vec_prior_samples = proposed_samples * self.normal_vec
            dist_to_zero = torch.abs(torch.sum(vec_prior_samples, dim=1))
            accept_or_not = (
                self.noise_factor * torch.rand(dist_to_zero.shape) > dist_to_zero
            )
            accepted_samples = proposed_samples[accept_or_not]

            num_accepted += accepted_samples.shape[0]
            all_samples = torch.cat((all_samples, accepted_samples), dim=0)
        return all_samples

    def log_prob(self, theta: Tensor):
        """
        Compute the unnormalized log-probability of the toy density.

        This is done by computing the acceptance probability (see `.sample()` method.
        Because the samples were proposed by a box uniform distribution, the acceptance
        probability is proportional to the joint density.
        """
        vec_prior_samples = theta * self.normal_vec
        dist_to_zero = torch.abs(torch.sum(vec_prior_samples, dim=1))
        acceptance_prob = torch.max(
            torch.zeros(dist_to_zero.shape), self.noise_factor - dist_to_zero
        )
        return torch.log(acceptance_prob)

    def set_default_x(self, x: Tensor):
        """
        Does not do anything. This function only exists because we pretend that this
        class is a `NeuralPosterior` in the tutorial. Calling `set_default_x()` is a
        required step when analysing conditional correlations.
        """
        pass
