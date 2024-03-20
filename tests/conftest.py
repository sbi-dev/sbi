import pyro
import pyro.distributions as dist
import pytest
import torch

from sbi.utils.sbiutils import seed_all_backends

# Seed for `set_seed` fixture. Change to random state of all seeded tests.
seed = 1


# Use seed automatically for every test function.
@pytest.fixture(autouse=True)
def set_seed():
    seed_all_backends(seed)


@pytest.fixture(scope="session", autouse=True)
def set_default_tensor_type():
    torch.set_default_tensor_type("torch.FloatTensor")


class LogisticRegressionModelPyro:
    """
    Simple logistic regression model in Pyro, intended to be only used during testing.
    """

    def __init__(self, labels: torch.Tensor, data: torch.Tensor):
        """
        Args:
            labels: Samples from a Bernoulli distribution.
            data: Samples from a normal distribution.
        """
        self.data = data
        self.dim = data.size(-1)
        self.labels = labels

    def __call__(self, *args, **kwargs):
        coefs_mean = torch.zeros(self.dim)
        coefs = pyro.sample("beta", dist.Normal(coefs_mean, torch.ones(self.dim)))
        y = pyro.sample(
            "y", dist.Bernoulli(logits=(coefs * self.data).sum(-1)), obs=self.labels
        )
        return y
