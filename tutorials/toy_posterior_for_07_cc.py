import torch
import sbi.utils as utils


class example_posterior:
    def __init__(self):
        self.normal_vec = torch.tensor([1, 0.5, -1, -2, 1, 1.5, 1, -1])

    def sample(self, num_samples):
        num_dim = 8
        all_samples = torch.empty(0, 8)
        num_accepted = 0
        while num_accepted < num_samples[0]:
            prior_samples = utils.BoxUniform(
                -2 * torch.ones(num_dim), 2 * torch.ones(num_dim)
            ).sample(num_samples)
            vec_prior_samples = prior_samples * self.normal_vec
            dist_to_zero = torch.abs(torch.sum(vec_prior_samples, dim=1))
            acceptance_prob = torch.rand(dist_to_zero.shape) > dist_to_zero
            accepted_samples = prior_samples[acceptance_prob]
            num_accepted += accepted_samples.shape[0]
            all_samples = torch.cat((all_samples, accepted_samples), dim=0)
        return all_samples

    def log_prob(self, samples):
        vec_prior_samples = samples * self.normal_vec
        dist_to_zero = torch.abs(torch.sum(vec_prior_samples, dim=1))
        acceptance_prob = torch.max(torch.zeros(dist_to_zero.shape), 1 - dist_to_zero)
        return torch.log(acceptance_prob)

    def set_default_x(self, x):
        pass
